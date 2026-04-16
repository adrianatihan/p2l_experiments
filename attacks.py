"""
PGD attack for binary BCE models.

Works with any nn.Module that outputs a single logit per sample.
"""

import torch
import torch.nn as nn


def pgd_attack_bce(model, X, y, epsilon, pgd_steps=20, pgd_restarts=5,
                   step_size=None, device="cuda", prev_deltas=None):
    """
    Batched PGD on GPU. Maximises BCE within L∞ ball of radius epsilon.

    Returns
    -------
    best_bce    : (N,) tensor — worst-case BCE per example
    best_inputs : (N, ...) tensor — adversarial inputs
    best_deltas : (N, ...) tensor — perturbations (for warm-starting)
    """
    if step_size is None:
        step_size = 2.5 * epsilon / pgd_steps

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    X = X.to(device).detach()
    y_f = y.float().to(device).detach()
    N = X.shape[0]

    best_bce = torch.full((N,), -float("inf"), device=device)
    best_inputs = X.clone()
    best_deltas = torch.zeros_like(X)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    total_restarts = pgd_restarts + (1 if prev_deltas is not None else 0)

    for r in range(total_restarts):
        if r == 0 and prev_deltas is not None:
            x_adv = (X + prev_deltas.clone().to(device)).detach()
        else:
            x_adv = (X + torch.empty_like(X).uniform_(-epsilon, epsilon)).detach()

        for _ in range(pgd_steps):
            x_adv = x_adv.detach().requires_grad_(True)
            loss = criterion(model(x_adv).squeeze(-1), y_f)
            grad = torch.autograd.grad(loss.sum(), x_adv)[0]
            x_adv = x_adv.detach() + step_size * grad.detach().sign()
            x_adv = torch.min(torch.max(x_adv, X - epsilon), X + epsilon)

        with torch.no_grad():
            bce = criterion(model(x_adv.detach()).squeeze(-1), y_f)
            improved = bce > best_bce
            best_bce[improved] = bce[improved]
            best_inputs[improved] = x_adv[improved]
            best_deltas[improved] = (x_adv - X)[improved]

    for p in model.parameters():
        p.requires_grad_(True)

    return best_bce.detach(), best_inputs.detach(), best_deltas.detach()

def pgd_attack_ce(model, X, y, epsilon, pgd_steps=20, pgd_restarts=5,
                  step_size=None, device="cuda", prev_deltas=None):
    """
    Batched PGD on GPU. Maximises cross-entropy within L∞ ball of radius epsilon.

    Returns
    -------
    best_ce     : (N,) tensor — worst-case CE per example
    best_inputs : (N, ...) tensor — adversarial inputs
    best_deltas : (N, ...) tensor — perturbations (for warm-starting)
    """
    if step_size is None:
        step_size = 2.5 * epsilon / pgd_steps

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    X = X.to(device).detach()
    y_t = y.long().to(device).detach()
    N = X.shape[0]

    best_ce = torch.full((N,), -float("inf"), device=device)
    best_inputs = X.clone()
    best_deltas = torch.zeros_like(X)

    total_restarts = pgd_restarts + (1 if prev_deltas is not None else 0)

    for r in range(total_restarts):
        if r == 0 and prev_deltas is not None:
            x_adv = (X + prev_deltas.clone().to(device)).detach()
        else:
            x_adv = (X + torch.empty_like(X).uniform_(-epsilon, epsilon)).detach()

        for _ in range(pgd_steps):
            x_adv = x_adv.detach().requires_grad_(True)
            loss = F.cross_entropy(model(x_adv), y_t, reduction="none")
            grad = torch.autograd.grad(loss.sum(), x_adv)[0]
            x_adv = x_adv.detach() + step_size * grad.detach().sign()
            x_adv = torch.min(torch.max(x_adv, X - epsilon), X + epsilon)

        with torch.no_grad():
            ce = F.cross_entropy(model(x_adv.detach()), y_t, reduction="none")
            improved = ce > best_ce
            best_ce[improved] = ce[improved]
            best_inputs[improved] = x_adv[improved]
            best_deltas[improved] = (x_adv - X)[improved]

    for p in model.parameters():
        p.requires_grad_(True)

    return best_ce.detach(), best_inputs.detach(), best_deltas.detach()