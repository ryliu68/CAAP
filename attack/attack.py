"""
CAAP_Attack: Core attack class
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import sys
import numpy as np
from tqdm import tqdm

from .modules import ASIT, MS_DIFE
from .synthesizer import PatchSynthesizer
from .losses import focal_loss, cw_like_loss, identity_loss, feature_diversity_loss
from .utils import (
    create_shape_mask, ssim_loss, _normalize_for_model, PATCH_INIT_LOW, PATCH_INIT_HIGH,
    ASIT_PRESETS
)


class CAAP_Attack:
    """
    CAAP: Grayscale Radiometric Adaptive Spatial Patch Attack.
    Core attack framework with support for targeted and untargeted modes.
    """
    def __init__(self, model, args, device):
        self.model = model
        self.args = args
        self.device = device
        self.in_channels = args.input_channels
        self.lambda_adv = args.lambda_adv
        self.lambda_vis = args.lambda_vis
        self.lambda_adv = args.lambda_adv

        self.patch_shape = args.patch_shape.lower()
        self.cross_thickness_ratio = args.cross_thickness_ratio
        self.patch_size = args.patch_size
        self.patch_mask = create_shape_mask(self.patch_shape, self.patch_size, self.cross_thickness_ratio).to(device)
        self.mask_area = float(self.patch_mask.sum().item())
        self.mask_fill_ratio = float(self.patch_mask.mean().item())
        self.location_strategy = (args.location_strategy or args.loc).lower()
        if self.location_strategy == 'topleft':
            self.location_strategy = 'top_left'
        self.canvas_loc = args.loc if args.location_strategy is None else 'center'

        # 1. Initialize Patch
        self.patch = nn.Parameter(
            torch.rand(1, self.in_channels, self.patch_size, self.patch_size).to(device)
            * (PATCH_INIT_HIGH - PATCH_INIT_LOW) + PATCH_INIT_LOW
        )

        # 2. Initialize ASIT
        self.use_asit = not args.disable_asit
        if self.use_asit:
            asit_params = self._get_asit_params(args)
            self.asit = ASIT(
                in_channels=self.in_channels,
                img_size=args.img_size,
                **asit_params
            ).to(device)
        else:
            self.asit = None
            print("⚠️ ASIT disabled")

        # 3. Initialize MS-DIFE
        self.use_ms_dife = not args.disable_ms_dife
        self.ms_dife = MS_DIFE(in_channels=self.in_channels).to(device) if self.use_ms_dife else None
        self.use_joint_training = self.use_ms_dife and args.ms_dife_joint_training
        self.use_distillation = self.use_ms_dife and args.ms_dife_distillation

        if self.use_ms_dife and self.use_joint_training:
            self.ms_dife.train()
            for p in self.ms_dife.parameters():
                p.requires_grad = True
            print("✅ MS-DIFE: JOINT TRAINING")
        elif self.use_ms_dife and self.use_distillation:
            self.ms_dife_frozen = MS_DIFE(in_channels=self.in_channels).to(device)
            self.ms_dife_frozen.eval()
            for p in self.ms_dife_frozen.parameters():
                p.requires_grad = False
            self.ms_dife.train()
            for p in self.ms_dife.parameters():
                p.requires_grad = True
            print("✅ MS-DIFE: SELF-DISTILLATION")
        elif self.use_ms_dife:
            self.ms_dife.eval()
            for p in self.ms_dife.parameters():
                p.requires_grad = False
            print("✅ MS-DIFE: FROZEN")
        else:
            print("⚠️ MS-DIFE disabled")

        # 4. Target bank for targeted mode
        self.target_bank = None
        self.require_target_bank = args.attack_mode == 'targeted' and self.use_ms_dife
        if self.require_target_bank:
            bank = self._build_target_bank(args.target_class, args.target_bank_size)
            self.target_bank = bank.to(device) if bank is not None else None
            if self.target_bank is None:
                raise RuntimeError(f"Unable to collect target references for class {args.target_class}")
            print(f"✅ Target bank: {self.target_bank.size(0)} samples for class {args.target_class}")

        # 5. Optimizer
        optimizer_params = [{'params': [self.patch], 'lr': args.lr}]
        if self.use_asit:
            optimizer_params.append({'params': self.asit.parameters(), 'lr': args.stn_lr})
        if self.use_joint_training or self.use_distillation:
            ms_dife_lr = args.ms_dife_lr
            optimizer_params.append({'params': self.ms_dife.parameters(), 'lr': ms_dife_lr})

        self.optimizer = optim.AdamW(optimizer_params, weight_decay=1e-4)

        # 6. Scheduler
        self.scheduler = None
        if not args.disable_scheduler and hasattr(args, 'trainloader'):
            max_lrs = [args.lr]
            if self.use_asit:
                max_lrs.append(args.stn_lr)
            if self.use_joint_training or self.use_distillation:
                max_lrs.append(args.ms_dife_lr)
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lrs,
                epochs=args.epochs,
                steps_per_epoch=len(args.trainloader),
                pct_start=0.1,
                anneal_strategy='cos'
            )

        # 7. Curriculum learning
        self.curriculum_epoch = 0
        self.curriculum_max_epochs = args.curriculum_epochs

        # 8. Gradient accumulation
        self.grad_accum_steps = args.grad_accum_steps
        self.accum_counter = 0

        # 9. Synthesizer
        self.synthesizer = PatchSynthesizer(args, device)


    def _get_asit_params(self, args) -> dict:
        """Get ASIT parameters from preset or manual specification."""
        preset_name = args.preset
        if preset_name and preset_name in ASIT_PRESETS:
            params = ASIT_PRESETS[preset_name].copy()
            print(f"✅ Using ASIT preset: {preset_name}")
        else:
            params = {
                'max_rotation_deg': args.max_rotation_deg,
                'max_translate_px': args.max_translate_px,
                'max_scale_delta': args.max_scale_delta,
                'max_gain_delta': args.max_gain_delta,
                'max_bias_delta': args.max_bias_delta,
            }

        params['enable_rotation'] = args.enable_rotation
        params['enable_translation'] = args.enable_translation
        params['enable_scale'] = args.enable_scale
        params['enable_gain'] = args.enable_gain
        params['enable_bias'] = args.enable_bias

        return params

    def _build_target_bank(self, target_class: int, bank_size: int):
        """Collect target-class samples from test gallery."""
        collected = []
        remaining = bank_size
        with torch.no_grad():
            for batch_imgs, batch_labels in self.args.testloader:
                mask = batch_labels == target_class
                if mask.any():
                    collected.append(batch_imgs[mask])
                    remaining -= mask.sum().item()
                    if remaining <= 0:
                        break
        if not collected:
            print(f"⚠️ No samples found for target class {target_class}")
            return None
        bank = torch.cat(collected, dim=0)
        return bank[:bank_size].clone()

    def _sample_target_batch(self, batch_size: int) -> torch.Tensor:
        """Sample from target bank."""
        if self.target_bank is None:
            raise RuntimeError("Target bank unavailable")
        idx = torch.randint(0, self.target_bank.size(0), (batch_size,), device=self.device)
        return self.target_bank[idx]

    def _masked_patch(self) -> torch.Tensor:
        """Return patch content masked to the desired geometry."""
        mask = self.patch_mask
        if mask.size(1) != self.patch.size(1):
            mask = mask.expand(1, self.patch.size(1), self.patch_size, self.patch_size)
        return self.patch * mask

    def _forward_asit(self, images: torch.Tensor) -> tuple:
        """Forward through ASIT or return identity transforms."""
        if self.use_asit:
            theta, intensity = self.asit(images)
            theta = {k: v.clone().contiguous() for k, v in theta.items()}
            intensity = {k: v.clone().contiguous() for k, v in intensity.items()}
            return theta, intensity
        B = images.size(0)
        dev = images.device
        theta = {
            'rot': torch.zeros(B, 1, device=dev),
            'trans': torch.zeros(B, 2, device=dev),
            'scale': torch.ones(B, 1, device=dev),
        }
        intensity = {
            'gain': torch.ones(B, 1, device=dev),
            'bias': torch.zeros(B, 1, device=dev),
        }
        return theta, intensity

    def _clamp_offsets(self, offsets: torch.Tensor, max_x: float, max_y: float) -> torch.Tensor:
        """Clamp translation offsets."""
        if offsets is None:
            return None
        offsets[:, 0] = torch.clamp(offsets[:, 0], -max_x, max_x)
        offsets[:, 1] = torch.clamp(offsets[:, 1], -max_y, max_y)
        return offsets

    def _compute_location_offsets(self, images_norm: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute per-sample translation offsets for placement strategies."""
        strategy = getattr(self, 'location_strategy', 'center')
        B, _, H, W = images_norm.shape
        device_ = images_norm.device
        offsets = torch.zeros(B, 2, device=device_)
        if strategy in (None, 'center'):
            return offsets

        max_x = max(0.0, W / 2.0 - self.patch_size / 2.0)
        max_y = max(0.0, H / 2.0 - self.patch_size / 2.0)

        if strategy == 'random':
            offsets[:, 0] = (torch.rand(B, device=device_) - 0.5) * 2 * max_x
            offsets[:, 1] = (torch.rand(B, device=device_) - 0.5) * 2 * max_y
        elif strategy in {'top_left', 'topleft'}:
            offsets[:, 0] = -max_x
            offsets[:, 1] = -max_y
        elif strategy == 'attention':
            offsets = self._attention_offsets(images_norm, labels, max_x, max_y)
        elif strategy == 'center_random':
            offsets[:, 0] = (torch.randn(B, device=device_) * max_x * 0.25)
            offsets[:, 1] = (torch.randn(B, device=device_) * max_y * 0.25)

        return offsets

    def _attention_offsets(self, images_norm: torch.Tensor, labels: torch.Tensor,
                           max_x: float, max_y: float) -> torch.Tensor:
        """Compute attention-based placement offsets."""
        if labels is None:
            return torch.zeros(images_norm.size(0), 2, device=images_norm.device)
        with torch.enable_grad():
            imgs = images_norm.detach().clone().requires_grad_(True)
            logits = self.model(_normalize_for_model(imgs, self.args.norm))
            loss = F.cross_entropy(logits, labels)
            grads = torch.autograd.grad(loss, imgs, retain_graph=False)[0]
        attn = grads.abs().mean(dim=1, keepdim=True)
        attn = F.avg_pool2d(attn, kernel_size=5, stride=1, padding=2)
        B, _, H, W = attn.shape
        flat = attn.view(B, -1)
        idx = torch.argmax(flat, dim=1)
        ys = (idx // W).float()
        xs = (idx % W).float()
        center_x = (W - 1) / 2.0
        center_y = (H - 1) / 2.0
        offsets = torch.zeros(B, 2, device=images_norm.device)
        offsets[:, 0] = xs - center_x
        offsets[:, 1] = ys - center_y
        return self._clamp_offsets(offsets, max_x, max_y)

    def compute_loss(self, clean_imgs: torch.Tensor, base_adv: torch.Tensor,
                     adv_eot_list: list, labels: torch.Tensor,
                     target_imgs: torch.Tensor = None,
                     curriculum_factor: float = 1.0) -> tuple:
        """Compute total loss with multi-sample EOT and advanced techniques."""
        # Visual loss
        l_vis = F.mse_loss(base_adv, clean_imgs) + ssim_loss(base_adv, clean_imgs)
        patch_content = self._masked_patch()
        tv = (torch.mean(torch.abs(patch_content[:, :, 1:, :] - patch_content[:, :, :-1, :])) +
              torch.mean(torch.abs(patch_content[:, :, :, 1:] - patch_content[:, :, :, :-1])))

        # Attack loss
        l_atk, l_id, l_focal, l_div = 0.0, 0.0, 0.0, 0.0
        use_focal = getattr(self.args, 'use_focal_loss', False)
        use_div = getattr(self.args, 'use_diversity_loss', False) and self.use_ms_dife

        for adv_eot in adv_eot_list:
            if self.args.norm == 'aug_0.5':
                adv_norm = (adv_eot - 0.5) / 0.5
            else:
                adv_norm = _normalize_for_model(adv_eot, self.args.norm)

            logits = self.model(adv_norm)
            feat = self.ms_dife(adv_eot) if self.use_ms_dife else None

            if self.args.attack_mode == 'targeted':
                target_labels = torch.full_like(labels, self.args.target_class)
                l_atk += cw_like_loss(logits, target_labels, 'targeted', self.args.kappa)
                if use_focal:
                    l_focal += focal_loss(logits, target_labels)
                if self.use_ms_dife and target_imgs is not None:
                    with torch.no_grad():
                        feat_ref = self.ms_dife(target_imgs)
                    l_id += identity_loss(feat, feat_ref, 'targeted', self.args.identity_margin)
            else:
                l_atk += cw_like_loss(logits, labels, 'untargeted', self.args.kappa)
                if use_focal:
                    l_focal += focal_loss(logits, (labels + 1) % self.args.num_class)
                if self.use_ms_dife:
                    with torch.no_grad():
                        feat_ref = self.ms_dife(clean_imgs)
                    l_id += identity_loss(feat, feat_ref, 'untargeted', self.args.identity_margin)

            if use_div and feat is not None:
                l_div += feature_diversity_loss(feat)

        n = max(1, len(adv_eot_list))
        loss_adv = curriculum_factor * (l_atk / n)
        loss_id = l_id / n
        loss_focal = l_focal / n if use_focal else 0.0
        loss_div = l_div / n if use_div else 0.0
        total = (
            self.lambda_adv * loss_adv +
            self.args.lambda_id * loss_id +
            self.args.lambda_tv * tv +
            self.lambda_vis * l_vis +
            getattr(self.args, 'lambda_focal', 0.1) * loss_focal +
            getattr(self.args, 'lambda_diversity', 0.05) * loss_div
        )

        # Self-distillation loss
        l_distill = 0.0
        if self.use_distillation:
            for adv_eot in adv_eot_list:
                feat_student = self.ms_dife(adv_eot)
                feat_teacher = self.ms_dife_frozen(adv_eot).detach()
                l_distill += F.mse_loss(feat_student, feat_teacher)
            total += 0.1 * l_distill / n

        return total, {
            "l_atk": (l_atk / n).item(),
            "l_id": loss_id.item() if torch.is_tensor(loss_id) else float(loss_id),
            "l_vis": l_vis.item(),
            "l_tv": tv.item(),
            "l_focal": loss_focal if isinstance(loss_focal, float) else loss_focal.item() if torch.is_tensor(loss_focal) else float(loss_focal),
            "l_distill": (l_distill / n).item() if self.use_distillation else 0.0,
        }

    def _curriculum_factor(self, epoch: int) -> float:
        """Curriculum learning: ramp from 0.3 to 1.0."""
        if not getattr(self.args, 'use_curriculum', False):
            return 1.0
        max_ep = self.curriculum_max_epochs
        return min(1.0, 0.3 + 0.7 * (epoch / max_ep))

    def train_epoch(self, epoch: int) -> tuple:
        """Train for one epoch."""
        self.model.eval()
        if self.use_asit:
            self.asit.train()
        if self.use_ms_dife:
            if self.use_joint_training or self.use_distillation:
                self.ms_dife.train()
            else:
                self.ms_dife.eval()
        if self.use_distillation:
            self.ms_dife_frozen.eval()
        self.patch.requires_grad = True

        epoch_metrics = []
        total_samples = 0
        correct_filtered = 0

        curriculum_factor = self._curriculum_factor(epoch)
        n_eot_samples = getattr(self.args, 'n_eot_samples', 1)
        grad_accum_steps = self.grad_accum_steps
        self.accum_counter = 0

        use_progress = sys.stdout.isatty()
        pbar = tqdm(self.args.trainloader, desc=f"Epoch {epoch}", disable=not use_progress)

        for batch_idx, (images, labels) in enumerate(pbar):
            if self.args.co3net and isinstance(images, list):
                images = images[0]

            images = images.to(self.device)
            labels = labels.to(self.device)

            # Filter: only attack correctly classified samples
            with torch.no_grad():
                clean_logits = self.model(_normalize_for_model(images, self.args.norm))
                preds = torch.argmax(clean_logits, dim=1)
                mask = preds.eq(labels)

                if self.args.attack_mode == 'targeted':
                    mask = mask & labels.ne(self.args.target_class)

            if mask.sum() == 0:
                continue

            images = images[mask]
            labels = labels[mask]
            correct_filtered += images.size(0)
            total_samples += mask.size(0)

            # For targeted, sample from target bank
            target_imgs = None
            if self.args.attack_mode == 'targeted' and self.require_target_bank:
                target_imgs = self._sample_target_batch(images.size(0))

            # Handle Normalization
            if self.args.norm == 'aug_0.5':
                images_denorm = images * 0.5 + 0.5
            else:
                images_denorm = images

            # Forward ASIT
            theta, intensity = self._forward_asit(images)

            # Synthesize Base Attack
            offsets_px = self._compute_location_offsets(images, labels)
            base_adv, mask_warped = self.synthesizer.synthesize_attack(
                images_denorm, theta, intensity, self.patch, self.patch_mask,
                self.patch_size, canvas_loc=self.canvas_loc, offsets_px=offsets_px
            )

            # Apply RaS (EOT)
            adv_eot_list = self.synthesizer.ras_eot_multi_sample(base_adv, n_samples=n_eot_samples,
                                                                  curriculum_factor=curriculum_factor)

            # Compute Loss
            loss, metrics = self.compute_loss(images_denorm, base_adv, adv_eot_list, labels,
                                              target_imgs, curriculum_factor)

            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps
            loss.backward()
            self.accum_counter += 1

            # Optimizer step (with gradient accumulation)
            if self.accum_counter >= grad_accum_steps:
                grad_params = [self.patch]
                if self.use_asit:
                    grad_params += list(self.asit.parameters())
                if (self.use_joint_training or self.use_distillation) and self.use_ms_dife:
                    grad_params += list(self.ms_dife.parameters())
                clip_grad_norm_(grad_params, self.args.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.accum_counter = 0

                self.patch.data.clamp_(0, 1)

                if self.scheduler:
                    self.scheduler.step()

            metrics['loss'] = loss.item() * grad_accum_steps
            epoch_metrics.append(metrics)

            pbar.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                atk=f"{metrics['l_atk']:.4f}",
                cf=f"{curriculum_factor:.2f}"
            )

        # Handle remaining gradients
        if self.accum_counter > 0:
            grad_params = [self.patch]
            if self.use_asit:
                grad_params += list(self.asit.parameters())
            if (self.use_joint_training or self.use_distillation) and self.use_ms_dife:
                grad_params += list(self.ms_dife.parameters())
            clip_grad_norm_(grad_params, self.args.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.patch.data.clamp_(0, 1)
            self.accum_counter = 0

        avg_loss = np.mean([m['loss'] for m in epoch_metrics]) if epoch_metrics else 0.0
        avg_atk = np.mean([m['l_atk'] for m in epoch_metrics]) if epoch_metrics else 0.0

        return avg_loss, avg_atk, correct_filtered, total_samples

    def evaluate(self, testloader) -> tuple:
        """Evaluate ASR on test set."""
        self.model.eval()
        if self.use_asit:
            self.asit.eval()

        succ = 0
        total = 0

        with torch.no_grad():
            for images, labels in testloader:
                if self.args.co3net and isinstance(images, list):
                    images = images[0]
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Filter, Note:　here mask is used for obtain the images can be recognized successfullu, not the topology mask $M$
                clean_logits = self.model(_normalize_for_model(images, self.args.norm))
                preds = torch.argmax(clean_logits, dim=1)
                mask = preds.eq(labels)

                if self.args.attack_mode == 'targeted':
                    mask = mask & labels.ne(self.args.target_class)

                if mask.sum() == 0:
                    continue

                images = images[mask]
                labels = labels[mask]

                # Forward ASIT
                theta, intensity = self._forward_asit(images)

                # Denorm for synthesis
                if self.args.norm == 'aug_0.5':
                    images_denorm = images * 0.5 + 0.5
                else:
                    images_denorm = images

                # Synthesize attack
                offsets_px = self._compute_location_offsets(images, labels)
                adv_images, _ = self.synthesizer.synthesize_attack(
                    images_denorm, theta, intensity, self.patch, self.patch_mask,
                    self.patch_size, canvas_loc=self.canvas_loc, offsets_px=offsets_px
                )

                # Model prediction
                if self.args.norm == 'aug_0.5':
                    adv_norm = (adv_images - 0.5) / 0.5
                else:
                    adv_norm = _normalize_for_model(adv_images, self.args.norm)

                logits = self.model(adv_norm)
                adv_preds = torch.argmax(logits, dim=1)

                # Count success
                if self.args.attack_mode == 'targeted':
                    succ += (adv_preds == self.args.target_class).sum().item()
                else:
                    succ += (adv_preds != labels).sum().item()

                total += labels.size(0)

        asr = 100.0 * succ / total if total > 0 else 0.0
        return asr, total

    def evaluate_with_eot(self, testloader, n_samples: int = 10) -> tuple:
        """Evaluate ASR with EOT and majority voting."""
        self.model.eval()
        prev_asit_training = self.asit.training if self.use_asit else False
        if self.use_asit:
            self.asit.eval()

        succ = 0
        total = 0

        with torch.no_grad():
            for images, labels in testloader:
                if self.args.co3net and isinstance(images, list):
                    images = images[0]
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Filter
                clean_logits = self.model(_normalize_for_model(images, self.args.norm))
                preds = torch.argmax(clean_logits, dim=1)
                mask = preds.eq(labels)

                if self.args.attack_mode == 'targeted':
                    mask = mask & labels.ne(self.args.target_class)

                if mask.sum() == 0:
                    continue

                images = images[mask]
                labels = labels[mask]
                B = images.size(0)

                # Forward ASIT
                theta, intensity = self._forward_asit(images)

                # Denorm
                if self.args.norm == 'aug_0.5':
                    images_denorm = images * 0.5 + 0.5
                else:
                    images_denorm = images

                # Synthesize base attack
                offsets_px = self._compute_location_offsets(images, labels)
                base_adv, _ = self.synthesizer.synthesize_attack(
                    images_denorm, theta, intensity, self.patch, self.patch_mask,
                    self.patch_size, canvas_loc=self.canvas_loc, offsets_px=offsets_px
                )

                # Multiple EOT evaluations with majority voting
                all_preds = []
                for _ in range(n_samples):
                    adv_eot = self.synthesizer.ras_eot_augmentation(base_adv.clone(), curriculum_factor=1.0)

                    if self.args.norm == 'aug_0.5':
                        adv_norm = (adv_eot - 0.5) / 0.5
                    else:
                        adv_norm = _normalize_for_model(adv_eot, self.args.norm)

                    logits = self.model(adv_norm)
                    all_preds.append(torch.argmax(logits, dim=1))

                # Stack predictions and majority voting
                all_preds = torch.stack(all_preds, dim=0)
                final_preds = torch.mode(all_preds, dim=0).values

                # Count success
                if self.args.attack_mode == 'targeted':
                    succ += (final_preds == self.args.target_class).sum().item()
                else:
                    succ += (final_preds != labels).sum().item()

                total += B

        # Restore training state
        if self.use_asit and prev_asit_training:
            self.asit.train()

        asr = 100.0 * succ / total if total > 0 else 0.0
        return asr, total
