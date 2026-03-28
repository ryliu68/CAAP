"""
Utils: Constants, SSIM, mask creation, normalization
"""
import math
import torch
import torch.nn.functional as F

# --- Constants ---
SSIM_C1 = 0.01 ** 2
SSIM_C2 = 0.03 ** 2
PATCH_INIT_LOW = 0.25
PATCH_INIT_HIGH = 0.75
EDGE_SMOOTH_KERNEL = 3



# ASIT Parameter Presets
ASIT_PRESETS = {
    "conservative": {
        "max_rotation_deg": 10.0,
        "max_translate_px": 8.0,
        "max_scale_delta": 0.1,
        "max_gain_delta": 0.2,
        "max_bias_delta": 0.2,
    },
    "balanced": {
        "max_rotation_deg": 20.0,
        "max_translate_px": 15.0,
        "max_scale_delta": 0.2,
        "max_gain_delta": 0.3,
        "max_bias_delta": 0.3,
    },
    "aggressive": {
        "max_rotation_deg": 30.0,
        "max_translate_px": 20.0,
        "max_scale_delta": 0.3,
        "max_gain_delta": 0.4,
        "max_bias_delta": 0.4,
    },
    "max_asr": {
        "max_rotation_deg": 25.0,
        "max_translate_px": 18.0,
        "max_scale_delta": 0.25,
        "max_gain_delta": 0.35,
        "max_bias_delta": 0.35,
    },
}


def _shape_fill_ratio(shape: str, thickness_ratio: float = 0.3) -> float:
    """Estimated ratio of active pixels for each geometric mask."""
    shape = shape.lower()
    if shape == 'square':
        return 1.0
    if shape == 'circle':
        return math.pi / 4.0
    if shape == 'triangle':
        return 0.5
    if shape == 'cross':
        ratio = max(1e-3, thickness_ratio)
        return max(1e-3, 2 * ratio - ratio ** 2)
    return 1.0


def _triangle_mask(size: int) -> torch.Tensor:
    """Create an isosceles triangle mask anchored at the top center."""
    coords = torch.linspace(0, size - 1, steps=size)
    yy, xx = torch.meshgrid(coords, coords)
    v0 = torch.tensor([(size - 1) / 2.0, 0.0])
    v1 = torch.tensor([0.0, size - 1.0])
    v2 = torch.tensor([size - 1.0, size - 1.0])
    denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
    denom = denom if denom != 0 else 1.0
    a = ((v1[1] - v2[1]) * (xx - v2[0]) + (v2[0] - v1[0]) * (yy - v2[1])) / denom
    b = ((v2[1] - v0[1]) * (xx - v2[0]) + (v0[0] - v2[0]) * (yy - v2[1])) / denom
    c = 1.0 - a - b
    mask = ((a >= 0) & (b >= 0) & (c >= 0)).float()
    return mask.unsqueeze(0).unsqueeze(0)


def create_shape_mask(shape: str, size: int, thickness_ratio: float = 0.25) -> torch.Tensor:
    """Create a binary mask (1 × 1 × size × size) for the requested shape."""
    shape = shape.lower()

    if shape == 'square':
        mask = torch.ones(1, 1, size, size)
    elif shape == 'cross':
        mask = torch.zeros(1, 1, size, size)
        thickness = max(1, int(round(size * thickness_ratio)))
        center = size // 2
        half = max(0, thickness // 2)
        mask[:, :, center - half:center + half + 1, :] = 1.0
        mask[:, :, :, center - half:center + half + 1] = 1.0
    elif shape == 'circle':
        coords = torch.linspace(0, size - 1, steps=size)
        yy, xx = torch.meshgrid(coords, coords)
        center = (size - 1) / 2.0
        radius = size / 2.0
        mask = (((xx - center) ** 2 + (yy - center) ** 2) <= radius ** 2).float().unsqueeze(0).unsqueeze(0)
    else:  # triangle
        mask = _triangle_mask(size)

    return mask


# --- SSIM ---
_SSIM_WINDOW_CACHE = {}


def _get_ssim_window(window_size: int, channel: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Get cached SSIM window or create new one."""
    key = (window_size, channel, device, dtype)
    if key not in _SSIM_WINDOW_CACHE:
        _SSIM_WINDOW_CACHE[key] = _create_ssim_window(window_size, channel).to(device=device, dtype=dtype)
    return _SSIM_WINDOW_CACHE[key]


def _create_ssim_window(window_size: int, channel: int) -> torch.Tensor:
    """Create Gaussian window for SSIM."""
    def gaussian(size, sigma):
        gauss = torch.tensor([
            math.exp(-(x - size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(size)
        ])
        return gauss / gauss.sum()

    _1d = gaussian(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return _2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim_loss(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Compute SSIM loss (1 - SSIM) with cached window."""
    channel = img1.size(1)
    window = _get_ssim_window(window_size, channel, img1.device, img1.dtype)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + SSIM_C1) * (2 * sigma12 + SSIM_C2)) / \
               ((mu1_sq + mu2_sq + SSIM_C1) * (sigma1_sq + sigma2_sq + SSIM_C2))

    return 1.0 - ssim_map.mean()


def _normalize_for_model(tensor: torch.Tensor, norm: str) -> torch.Tensor:
    """Normalize tensor for model input based on norm type."""
    if norm == 'aug_0.5':
        return tensor
    return (tensor - 0.5) / 0.5
