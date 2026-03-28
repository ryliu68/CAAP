"""
Losses: Focal Loss, CW Loss, Identity Loss, Diversity Loss
"""
import torch
import torch.nn.functional as F


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0,
               reduction: str = 'mean') -> torch.Tensor:
    """
    Focal Loss for hard sample mining.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_weight = (1 - pt) ** gamma
    loss = focal_weight * ce_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def cw_like_loss(logits: torch.Tensor, labels: torch.Tensor,
                  mode: str, kappa: float = 0.0) -> torch.Tensor:
    """
    Compute margin-based loss following C&W style for stability.

    For targeted: minimize (max_other - target + kappa)
    For untargeted: minimize (target - max_other + kappa)
    """
    target_logits = logits.gather(1, labels.view(-1, 1))
    logits_masked = logits.clone()
    logits_masked.scatter_(1, labels.view(-1, 1), float('-inf'))
    max_other, _ = logits_masked.max(dim=1, keepdim=True)

    if mode == 'targeted':
        loss = torch.clamp(max_other - target_logits + kappa, min=0.0)
    else:
        loss = torch.clamp(target_logits - max_other + kappa, min=0.0)

    return loss.mean()


def identity_loss(feat_adv: torch.Tensor, feat_ref: torch.Tensor,
                  mode: str, identity_margin: float = 0.5) -> torch.Tensor:
    """
    Cosine distance-based identity loss for feature-level attack.

    For targeted: minimize distance to target features
    For untargeted: maximize distance from source features (with margin)
    """
    dist = 1 - F.cosine_similarity(feat_adv, feat_ref, dim=1)
    if mode == 'targeted':
        return dist.mean()
    return F.relu(identity_margin - dist).mean()


def feature_diversity_loss(feat_adv: torch.Tensor) -> torch.Tensor:
    """Minimize feature similarity to improve generalization."""
    if feat_adv.size(0) < 2:
        return torch.tensor(0.0, device=feat_adv.device)
    feat_norm = F.normalize(feat_adv, p=2, dim=1)
    sim = torch.mm(feat_norm, feat_norm.t())
    mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    return sim[mask].mean()
