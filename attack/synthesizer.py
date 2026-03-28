"""
Synthesizer: Patch synthesis, EOT augmentation, affine transformations
"""
import math
import torch
import torch.nn.functional as F
from .utils import EDGE_SMOOTH_KERNEL


class PatchSynthesizer:
    """Handles adversarial patch synthesis and EOT augmentation."""

    def __init__(self, args, device):
        self.args = args
        self.device = device

    def synthesize_attack(self, images: torch.Tensor, theta: dict, intensity: dict,
                          patch: torch.Tensor, patch_mask: torch.Tensor, patch_size: int,
                          canvas_loc: str = 'center', offsets_px: torch.Tensor = None) -> tuple:
        """
        Generates base adversarial examples (Clean, no EOT).

        Args:
            images: Clean input images [B, C, H, W]
            theta: Geometric parameters {rot, trans, scale}
            intensity: Intensity parameters {gain, bias}
            patch: Patch tensor [1, C, P, P]
            patch_mask: Patch mask [1, 1, P, P]
            patch_size: Patch size
            canvas_loc: Patch placement ('center' or 'topleft')
            offsets_px: Per-sample translation offsets

        Returns:
            adv_images: Adversarial images [B, C, H, W]
            mask_warped: Transformed mask [B, 1, H, W]
        """
        B, C, H, W = images.size()

        # Handle None inputs
        if theta is None:
            theta = {
                'rot': torch.zeros(B, device=self.device),
                'trans': torch.zeros(B, 2, device=self.device),
                'scale': torch.ones(B, device=self.device)
            }
        if intensity is None:
            intensity = {
                'gain': torch.ones(B, device=self.device),
                'bias': torch.zeros(B, device=self.device)
            }

        # 1. Prepare Patch and Mask
        patch_full = torch.zeros(B, C, H, W, device=self.device)
        mask_full = torch.zeros(B, 1, H, W, device=self.device)

        # Compute patch placement
        if canvas_loc == 'center':
            top = (H - patch_size) // 2
            left = (W - patch_size) // 2
        elif canvas_loc == 'topleft':
            top = 0
            left = 0
        else:
            raise ValueError(f"Unknown loc: {canvas_loc}. Use 'center' or 'topleft'.")

        masked_patch = patch * patch_mask
        patch_full[:, :, top:top+patch_size, left:left+patch_size] = masked_patch
        mask_full[:, :, top:top+patch_size, left:left+patch_size] = patch_mask

        # 2. Apply Geometric Transformation
        angle = theta['rot'] * math.pi / 180.0
        offsets = offsets_px.to(self.device) if offsets_px is not None else torch.zeros(B, 2, device=self.device)
        max_x = max(0.0, W / 2.0 - patch_size / 2.0)
        max_y = max(0.0, H / 2.0 - patch_size / 2.0)
        trans = theta['trans'] + offsets
        trans_x = torch.clamp(trans[:, 0], -max_x, max_x)
        trans_y = torch.clamp(trans[:, 1], -max_y, max_y)
        tx = trans_x / (W / 2)
        ty = trans_y / (H / 2)
        scale = theta['scale'].view(B)
        cos_a = torch.cos(angle).view(B)
        sin_a = torch.sin(angle).view(B)

        # Build affine matrix [B, 2, 3]
        affine_mat = torch.zeros(B, 2, 3, device=self.device)
        affine_mat[:, 0, 0] = cos_a * scale
        affine_mat[:, 0, 1] = -sin_a * scale
        affine_mat[:, 0, 2] = tx
        affine_mat[:, 1, 0] = sin_a * scale
        affine_mat[:, 1, 1] = cos_a * scale
        affine_mat[:, 1, 2] = ty

        grid = F.affine_grid(affine_mat, images.size(), align_corners=False)
        patch_warped = F.grid_sample(patch_full, grid, align_corners=False)
        mask_warped = F.grid_sample(mask_full, grid, align_corners=False)

        # 3. Apply Intensity Modulation
        gain = intensity['gain'].view(B, 1, 1, 1)
        bias = intensity['bias'].view(B, 1, 1, 1)
        patch_warped = patch_warped * gain + bias

        # 4. Fusion with optional edge smoothing
        if not self.args.disable_edge_smoothing:
            mask_warped = F.avg_pool2d(
                mask_warped,
                kernel_size=EDGE_SMOOTH_KERNEL,
                stride=1,
                padding=EDGE_SMOOTH_KERNEL // 2
            )

        adv_images = images * (1 - mask_warped) + patch_warped * mask_warped
        return torch.clamp(adv_images, 0, 1), mask_warped

    def ras_eot_augmentation(self, images: torch.Tensor,
                              curriculum_factor: float = 1.0) -> torch.Tensor:
        """
        Radiometric-aware Synthesis (EOT) for grayscale robustness.
        """
        if not self.args.use_ras_eot:
            return images

        B = images.size(0)
        cf = curriculum_factor

        # 1. Brightness jitter
        jitter_b = self.args.jitter_brightness * cf
        if jitter_b > 0:
            b_factor = 1.0 + (torch.rand(B, 1, 1, 1, device=self.device) - 0.5) * jitter_b * 2
            images = images * b_factor

        # 2. Contrast jitter
        jitter_c = self.args.jitter_contrast * cf
        if jitter_c > 0:
            c_factor = 1.0 + (torch.rand(B, 1, 1, 1, device=self.device) - 0.5) * jitter_c * 2
            mean = images.mean(dim=[2, 3], keepdim=True)
            images = (images - mean) * c_factor + mean

        # 3. Gaussian noise
        noise_std = getattr(self.args, 'noise_std', 0.02) * cf
        if noise_std > 0:
            noise = torch.randn_like(images) * noise_std
            images = images + noise

        # 4. Optional affine jitter
        if getattr(self.args, 'use_affine_eot', False):
            images = self._apply_affine_jitter(images, curriculum_factor=cf)

        return torch.clamp(images, 0, 1)

    def ras_eot_multi_sample(self, images: torch.Tensor, n_samples: int = 1,
                             curriculum_factor: float = 1.0) -> list:
        """Generate multiple EOT-augmented versions for diversity."""
        if n_samples <= 1 or not self.args.use_ras_eot:
            return [self.ras_eot_augmentation(images, curriculum_factor)]

        return [self.ras_eot_augmentation(images.clone(), curriculum_factor)
                for _ in range(n_samples)]

    def _apply_affine_jitter(self, images: torch.Tensor,
                             curriculum_factor: float = 1.0) -> torch.Tensor:
        """Apply small random affine transformations."""
        B, C, H, W = images.size()
        cf = curriculum_factor
        max_rot = getattr(self.args, 'affine_rotation', 5.0) * cf
        max_trans = getattr(self.args, 'affine_translate', 3.0) * cf

        angles = (torch.rand(B, device=self.device) - 0.5) * 2 * max_rot * math.pi / 180.0
        tx = (torch.rand(B, device=self.device) - 0.5) * 2 * max_trans / (W / 2)
        ty = (torch.rand(B, device=self.device) - 0.5) * 2 * max_trans / (H / 2)

        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)

        # Build affine matrix
        theta = torch.zeros(B, 2, 3, device=self.device)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a
        theta[:, 1, 2] = ty

        grid = F.affine_grid(theta, images.size(), align_corners=False)
        return F.grid_sample(images, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
