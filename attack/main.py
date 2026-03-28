"""
CAAP Main: Entry point for attack training
"""
import argparse
import json
import os
import time
import sys
import numpy as np
import torch
import random
from torchvision.utils import save_image



CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

from data.data import get_data
from utils.net_factory import get_net

from attack.attack import CAAP_Attack


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    """For reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _resolve_num_class(dataset: str) -> int:
    """Resolve number of classes for dataset.

    Note: AISEC labels are 1-49, so we need 50 classes (0-49) to match checkpoint.
    """
    dataset = dataset.lower()
    if dataset == "tongji":
        return 600
    if dataset == "iitd":
        return 460
    if dataset == "aisec":
        return 50
    raise ValueError(f"Unsupported dataset: {dataset}")


def main():
    parser = argparse.ArgumentParser(
        description="CAAP: Capture-Aware Adversarial Patch Attacks on Palmprint Recognition"
    )

    # Dataset & Mode
    parser.add_argument("--dataset", type=str, default="tongji",
                        choices=["tongji", "iitd", "aisec"])
    parser.add_argument("--attack_mode", type=str, default="untargeted",
                        choices=["targeted", "untargeted"])
    parser.add_argument("--target_class", type=int, default=0)
    parser.add_argument("--target_bank_size", type=int, default=64)

    # Training Hyperparams
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--stn_lr", type=float, default=5e-4)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--grad_accum_steps", type=int, default=1)


    # Loss Weights (lambda_adv is fixed at 1.0)
    parser.add_argument("--lambda_adv", type=float, default=1.0) # Fix lambda_adv to 1.0 (for ablation study control)
    parser.add_argument("--lambda_vis", type=float, default=0.004)
    parser.add_argument("--lambda_id", type=float, default=0.20)
    parser.add_argument("--lambda_tv", type=float, default=2e-5)
    parser.add_argument("--identity_margin", type=float, default=0.5)
    parser.add_argument("--kappa", type=float, default=0.0)

    # Advanced Loss Options
    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--lambda_focal", type=float, default=0.1)
    parser.add_argument("--use_diversity_loss", action="store_true")
    parser.add_argument("--lambda_diversity", type=float, default=0.05)

    # Curriculum Learning
    parser.add_argument("--use_curriculum", action="store_true")
    parser.add_argument("--curriculum_epochs", type=int, default=10)

    # MS-DIFE Training Options
    parser.add_argument("--ms_dife_joint_training", action="store_true")
    parser.add_argument("--ms_dife_lr", type=float, default=1e-5)
    parser.add_argument("--ms_dife_distillation", action="store_true")

    # Patch & Image Settings
    parser.add_argument("--patch_size", type=int, default=40)
    parser.add_argument("--patch_shape", type=str, default="cross",
                        choices=["cross","patch"])
    parser.add_argument("--cross_thickness_ratio", type=float, default=0.25)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--input_channels", type=int, default=1)
    parser.add_argument("--norm", type=str, default="aug_0.5")
    parser.add_argument("--loc", type=str, default="center", choices=["center", "topleft"])
    parser.add_argument("--location_strategy", type=str, default=None,
                        choices=["center", "random", "attention", "top_left", "topleft", "center_random"])

    # ASIT Parameter Preset
    parser.add_argument("--preset", type=str, default=None,
                        choices=["conservative", "balanced", "aggressive", "max_asr"])

    parser.add_argument("--disable_asit", action="store_true")
    parser.add_argument("--disable_ms_dife", action="store_true")

    # ASIT Transformation Bounds
    parser.add_argument("--max_rotation_deg", type=float, default=20.0)
    parser.add_argument("--max_rotation", type=float, default=None)
    parser.add_argument("--max_translate_px", type=float, default=15.0)
    parser.add_argument("--max_translate", type=float, default=None)
    parser.add_argument("--max_scale_delta", type=float, default=0.2)
    parser.add_argument("--max_gain_delta", type=float, default=0.3)
    parser.add_argument("--max_bias_delta", type=float, default=0.3)
    parser.add_argument("--max_bias", type=float, default=None)

    # ASIT Ablation Flags
    parser.add_argument("--enable_rotation", action="store_true", default=True)
    parser.add_argument("--no_enable_rotation", action="store_false", dest="enable_rotation")
    parser.add_argument("--enable_translation", action="store_true", default=True)
    parser.add_argument("--no_enable_translation", action="store_false", dest="enable_translation")
    parser.add_argument("--enable_scale", action="store_true", default=True)
    parser.add_argument("--no_enable_scale", action="store_false", dest="enable_scale")
    parser.add_argument("--enable_gain", action="store_true", default=True)
    parser.add_argument("--no_enable_gain", action="store_false", dest="enable_gain")
    parser.add_argument("--enable_bias", action="store_true", default=True)
    parser.add_argument("--no_enable_bias", action="store_false", dest="enable_bias")

    # RaS (EOT) Settings
    parser.add_argument("--use_ras_eot", action="store_true", default=True)
    parser.add_argument("--no_ras_eot", action="store_false", dest="use_ras_eot")
    parser.add_argument("--n_eot_samples", type=int, default=1)
    parser.add_argument("--jitter_brightness", type=float, default=0.25)
    parser.add_argument("--jitter_contrast", type=float, default=0.2)
    parser.add_argument("--noise_std", type=float, default=0.02)
    parser.add_argument("--use_affine_eot", action="store_true", default=False)
    parser.add_argument("--affine_rotation", type=float, default=5.0)
    parser.add_argument("--affine_translate", type=float, default=3.0)

    # Evaluation Settings
    parser.add_argument("--eot_eval", action="store_true")
    parser.add_argument("--eot_eval_samples", type=int, default=10)

    # Misc
    parser.add_argument("--net", type=str, default="compnet")
    parser.add_argument("--model_ckpt", type=str, default=None)
    parser.add_argument("--co3net", action="store_true", default=False)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--disable_scheduler", action="store_true")
    parser.add_argument("--disable_edge_smoothing", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./results")

    args = parser.parse_args()


    # Handle deprecated parameter names
    if args.max_rotation is not None:
        args.max_rotation_deg = args.max_rotation
    if args.max_translate is not None:
        args.max_translate_px = args.max_translate
    if args.max_bias is not None:
        args.max_bias_delta = args.max_bias

    # Create Save Directory
    mode_suffix = f"{args.attack_mode}"
    if args.attack_mode == 'targeted':
        mode_suffix += f"_t{args.target_class}"
    exp_name = f"{args.dataset}_{args.net}_{mode_suffix}_ps{args.patch_size}"
    save_path = os.path.join(args.save_dir, exp_name)
    os.makedirs(save_path, exist_ok=True)

    # --- 1. Load Data & Model ---
    print("=" * 80)
    print(f"🚀 CAAP: Grayscale Radiometric Adaptive Spatial Patch Attack")
    print(f"   Dataset: {args.dataset}")
    print(f"   Model: {args.net}")
    print(f"   Attack Mode: {args.attack_mode}")
    if args.attack_mode == 'targeted':
        print(f"   Target Class: {args.target_class}")
    print(f"   Patch Size: {args.patch_size}×{args.patch_size}")
    print(f"   Patch Shape: {args.patch_shape}")
    print(f"   λ_vis={args.lambda_vis}, λ_id={args.lambda_id}, λ_tv={args.lambda_tv}")
    print("=" * 80)

    args.num_class = _resolve_num_class(args.dataset)

    trainloader, _, testloader = get_data(args)
    args.trainloader = trainloader
    args.testloader = testloader

    # Infer image size and channels
    dataset = getattr(trainloader, "dataset", None)
    if dataset is not None:
        args.img_size = getattr(dataset, "imside", args.img_size)
        args.input_channels = getattr(dataset, "chs", getattr(dataset, "outchannels", args.input_channels))

    print(f"\n📦 Data loaded: {len(trainloader)} train batches, {len(testloader)} test batches")
    print(f"   Image size: {args.img_size}×{args.img_size}, Channels: {args.input_channels}")

    # Load target model
    model = get_net(args).to(device)
    if args.model_ckpt:
        custom_ckpt = os.path.expanduser(args.model_ckpt)
        print(f"🔁 Loading custom checkpoint: {custom_ckpt}")
        state = torch.load(custom_ckpt, map_location=device)
        model.load_state_dict(state)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"✅ Target model {args.net} loaded and FROZEN")

    # --- 2. Initialize Attacker ---
    attacker = CAAP_Attack(model, args, device)
    print(f"   ➤ Effective mask area: {attacker.mask_area:.0f} px (fill={attacker.mask_fill_ratio:.3f})")


    best_asr = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    patience = args.early_stop_patience

    # --- 3. Training Loop ---
    print(f"\n🔄 Start Training for {args.epochs} epochs (patience={patience})...")
    training_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train
        loss_avg, atk_loss_avg, kept, total = attacker.train_epoch(epoch + 1)

        # Evaluate
        if args.eot_eval:
            asr, eval_total = attacker.evaluate_with_eot(args.testloader, n_samples=args.eot_eval_samples)
        else:
            asr, eval_total = attacker.evaluate(args.testloader)

        epoch_time = time.time() - epoch_start

        # Logging
        mode_label = "Target Success" if args.attack_mode == 'targeted' else "ASR"
        eval_method = f"(EOT×{args.eot_eval_samples})" if args.eot_eval else ""
        print(f"\n📊 Epoch {epoch+1}/{args.epochs} | Time: {epoch_time:.1f}s")
        print(f"   Loss: {loss_avg:.4f} | AtkLoss: {atk_loss_avg:.4f}")
        print(f"   {mode_label}{eval_method}: {asr:.2f}% ({eval_total} samples)")
        print(f"   Correctly classified samples used: {kept}/{total}")

       
        # Save Best
        if asr > best_asr:
            best_asr = asr
            best_epoch = epoch + 1
            epochs_no_improve = 0
            print(f"   ✨ New Best {mode_label}: {best_asr:.2f}%! Saving...")

            # Save Patch
            torch.save(attacker.patch.data, os.path.join(save_path, "best_patch.pth"))
            save_image(attacker.patch.data, os.path.join(save_path, "best_patch.png"))

            # Save ASIT Model
            if attacker.use_asit:
                torch.save(attacker.asit.state_dict(), os.path.join(save_path, "best_asit.pth"))

            # Save visualization
            with torch.no_grad():
                for images, labels in args.testloader:
                    if args.co3net and isinstance(images, list):
                        images = images[0]
                    images = images.to(device)[:8]

                    if attacker.use_asit:
                        theta, intensity = attacker.asit(images)
                    else:
                        theta, intensity = None, None

                    # Denormalize for patch synthesis / visualization
                    if args.norm == 'aug_0.5':
                        images_vis = images * 0.5 + 0.5
                    else:
                        images_vis = images

                    loc = args.loc
                    demo_adv, demo_mask = attacker.synthesizer.synthesize_attack(
                        images_vis, theta, intensity, attacker.patch, attacker.patch_mask,
                        attacker.patch_size, canvas_loc=loc
                    )

                    # Stack: original | mask | adversarial
                    n_channels = images_vis.size(1)
                    if n_channels == 1:
                        vis_mask = demo_mask
                    else:
                        vis_mask = demo_mask.repeat(1, n_channels, 1, 1)

                    save_image(
                        torch.cat([images_vis, vis_mask, demo_adv], dim=0),
                        os.path.join(save_path, f"vis_epoch_{epoch+1}.png"),
                        nrow=8
                    )
                    break
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                print(f"\n⏹️ Early stopping triggered (no improvement for {patience} epochs)")
                break

    # Training complete
    total_time = time.time() - training_start
    print(f"\n{'='*80}")
    print(f"✅ Training Complete!")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Best {mode_label}: {best_asr:.2f}% (epoch {best_epoch})")
    print(f"📂 Results saved to: {save_path}")
    print(f"{'='*80}")

    # Save final metrics
    final_metrics = {
        "best_asr": best_asr,
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "attack_mode": args.attack_mode,
        "target_class": args.target_class if args.attack_mode == 'targeted' else None,
        "dataset": args.dataset,
        "net": args.net,
        "patch_size": args.patch_size,
        "patch_shape": attacker.patch_shape,
        "mask_area": attacker.mask_area,
        "mask_fill_ratio": attacker.mask_fill_ratio,
        "location_strategy": attacker.location_strategy,
        "lambda_adv": args.lambda_adv,
        "lambda_vis": args.lambda_vis,
        "lambda_id": args.lambda_id,
        "lambda_tv": args.lambda_tv,
        "total_time_minutes": total_time / 60,
    }
    with open(os.path.join(save_path, "metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)


if __name__ == "__main__":
    set_seed()

    main()
