import numpy as np
import torch
from torch.nn import functional as F


def balance(labels, classes):
    counts = np.bincount(labels, minlength=classes)
    if (counts == 0).any():
        raise ValueError(f'Cannot compute inverse-root weights for Train class counts {counts.tolist()}')
    weights = counts.astype(np.float64) ** -0.5
    return {'counts': counts.tolist(), 'weights': (weights / weights.mean()).tolist()}


def loss_terms(logits, expr, intensity, weights):
    expr_logits, int_logits = logits['expression_logits'], logits['intensity_logits']
    expr_weight, int_weight = weights
    expr_sum = F.cross_entropy(expr_logits, expr, weight=expr_weight, reduction='sum')
    mass = expr_sum.new_tensor(expr.numel()) if expr_weight is None else expr_weight[expr].sum()
    active = expr != 0
    targets = intensity[active]
    int_sum = expr_sum.new_zeros(())
    if targets.numel():
        int_sum = F.cross_entropy(int_logits[active], targets, weight=int_weight, reduction='sum')
    return torch.stack((expr_sum, mass, int_sum, expr_sum.new_tensor(targets.numel())))


def mean_losses(terms, scale):
    expr_sum, mass, int_sum, count = terms.unbind()
    expr_loss, int_loss = expr_sum / mass, int_sum / count.clamp_min(1)
    return expr_loss + scale * int_loss, expr_loss, int_loss


def baseline_loss(logits, expr, intensity, weights, scale):
    expr_logits, int_logits = logits['expression_logits'], logits['intensity_logits']
    expr_loss = F.cross_entropy(expr_logits, expr, weight=weights[0])
    active = expr != 0
    int_loss = expr_loss.new_zeros(())
    if active.any():
        int_loss = F.cross_entropy(int_logits[active], intensity[active], weight=weights[1],
                                   reduction='none').mean()
    return expr_loss + scale * int_loss
