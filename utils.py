"""
Shared utilities for P2L.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_bce_per_example(model, X, y, device="cuda"):
    """Per-example BCE for any single-logit binary model."""
    model.eval()
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
    logits = model(X_t).squeeze(-1)
    return F.binary_cross_entropy_with_logits(
        logits, y_t, reduction="none"
    ).cpu().numpy()


@torch.no_grad()
def evaluate_model_binary(model, X_test, y_test, device="cuda"):
    """
    Returns (error_rate, accuracy) for a single-logit binary model.
    """
    model.eval()
    X_t = torch.FloatTensor(X_test).to(device)
    logits = model(X_t).squeeze(-1)
    preds = (logits >= 0.0).long()
    labels = torch.LongTensor(y_test).to(device)
    acc = (preds == labels).float().mean().item()
    return 1.0 - acc, acc


def fmt_time(s):
    """Format seconds as human-readable string."""
    if s < 60:
        return f"{s:.1f}s"
    return f"{int(s // 60)}m {s % 60:.1f}s"
