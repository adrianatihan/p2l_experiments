"""
Training utilities for P2L.

All functions accept any nn.Module with single-logit output.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR


# ═══════════════════════════════════════════════════════════════════════════════
#  Standard BCE training
# ═══════════════════════════════════════════════════════════════════════════════

def train_model_bce(model, X_train, y_train, epochs=100, lr=1e-3,
                    batch_size=128, device="cuda", verbose=False, **_ignored):
    """Clean BCE training with Adam + cosine LR."""
    model = model.to(device).train()

    X_t = _to_float_tensor(X_train)
    y_t = _to_float_tensor(y_train)

    loader = _make_loader(X_t, y_t, batch_size, device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    for epoch in range(epochs):
        total = 0.0
        for bx, by in loader:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx).squeeze(-1), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        scheduler.step()
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, BCE: {total/len(loader):.4f}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Pretraining with cosine LR + early stopping
# ═══════════════════════════════════════════════════════════════════════════════

def pretrain_model_bce(model, X_train, y_train, epochs=500, lr=0.01,
                       batch_size=32, device="cpu", val_fraction=0.0,
                       patience=500, verbose=False, **_ignored):
    """Pretrain h0 with cosine-annealing LR and optional early stopping."""
    model = model.to(device)

    if isinstance(X_train, torch.Tensor):
        X_train = X_train.cpu().numpy()
    if isinstance(y_train, torch.Tensor):
        y_train = y_train.cpu().numpy()

    n = len(X_train)
    n_val = max(1, int(n * val_fraction))
    perm = np.random.permutation(n)
    val_idx, trn_idx = perm[:n_val], perm[n_val:]

    X_t = torch.FloatTensor(X_train[trn_idx]).to(device)
    y_t = torch.FloatTensor(y_train[trn_idx]).to(device)
    X_v = torch.FloatTensor(X_train[val_idx]).to(device)
    y_v = torch.FloatTensor(y_train[val_idx]).to(device)

    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.95)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss, best_state, wait = float("inf"), None, 0

    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx).squeeze(-1), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v).squeeze(-1), y_v).item()
        model.train()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch+1} "
                      f"(best val_loss={best_val_loss:.4f})")
            break

        if verbose and (epoch + 1) % 100 == 0:
            print(f"    Pretrain epoch {epoch+1}/{epochs}  "
                  f"val_loss={val_loss:.4f}  lr={scheduler.get_last_lr()[0]:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(f"    Pretraining finished: best val_loss={best_val_loss:.4f}")

    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  TRADES adversarial training
# ═══════════════════════════════════════════════════════════════════════════════

def _binary_logits_to_2class(z):
    """Convert single logit z → [z, 0] so standard KL/softmax work."""
    return torch.stack([z, torch.zeros_like(z)], dim=-1)


def _trades_inner_pgd(model, x_clean, epsilon, step_size, num_steps, device):
    """TRADES inner PGD: maximise KL(clean ‖ adv)."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    logits_clean = model(x_clean.detach()).squeeze(-1)
    p_clean = F.softmax(_binary_logits_to_2class(logits_clean), dim=-1).detach()

    x_adv = x_clean.detach() + 0.001 * torch.randn_like(x_clean)

    for _ in range(num_steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits_adv = model(x_adv).squeeze(-1)
        log_q = F.log_softmax(_binary_logits_to_2class(logits_adv), dim=-1)
        kl = F.kl_div(log_q, p_clean, reduction="sum")
        grad = torch.autograd.grad(kl, x_adv)[0]
        x_adv = x_adv.detach() + step_size * grad.detach().sign()
        x_adv = torch.min(torch.max(x_adv, x_clean.detach() - epsilon),
                          x_clean.detach() + epsilon)

    for p in model.parameters():
        p.requires_grad_(True)
    return x_adv.detach()


def train_model_trades(model, X_train, y_train, epochs=60, lr=1e-4,
                       batch_size=128, device="cuda", verbose=False,
                       epsilon=0.01, trades_beta=6.0, trades_pgd_steps=5,
                       trades_step_size=None, **_ignored):
    """TRADES adversarial training adapted for binary BCE."""
    if trades_step_size is None:
        trades_step_size = epsilon / 3.0

    model = model.to(device).train()

    X_t = _to_float_tensor(X_train)
    y_t = _to_float_tensor(y_train)
    loader = _make_loader(X_t, y_t, batch_size, device)

    bce = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    for epoch in range(epochs):
        total_loss = 0.0
        for bx, by in loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)

            x_adv = _trades_inner_pgd(model, bx, epsilon,
                                      trades_step_size, trades_pgd_steps, device)
            model.train()
            optimizer.zero_grad()

            logits_clean = model(bx).squeeze(-1)
            loss_clean = bce(logits_clean, by)

            p_clean = F.softmax(
                _binary_logits_to_2class(logits_clean.detach()), dim=-1)
            logits_adv = model(x_adv).squeeze(-1)
            log_q = F.log_softmax(_binary_logits_to_2class(logits_adv), dim=-1)
            loss_robust = F.kl_div(log_q, p_clean, reduction="batchmean")

            loss = loss_clean + trades_beta * loss_robust
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Adaptive epoch scaling
# ═══════════════════════════════════════════════════════════════════════════════

def compute_adaptive_epochs(T_size, N_effective, base_epochs=200, min_epochs=30):
    """Scale training epochs with |T| to avoid overfitting on tiny sets."""
    if T_size <= 0 or N_effective <= 0:
        return base_epochs
    ratio = T_size / N_effective
    scaled = int(base_epochs * np.sqrt(ratio))
    return max(min_epochs, min(base_epochs, scaled))


# ═══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _to_float_tensor(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().clone().float()
    return torch.FloatTensor(arr)


def _make_loader(X_t, y_t, batch_size, device):
    if len(X_t) < 10_000:
        X_t, y_t = X_t.to(device), y_t.to(device)
    return DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(len(X_t) >= 10_000 and device != "cpu"),
        num_workers=0,
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  Adversarial cross-entropy training
# ═══════════════════════════════════════════════════════════════════════════════

def train_model_adv_ce(model, X_train, y_train, epochs=100, lr=5e-3,
                       batch_size=128, device="cuda", verbose=False,
                       epsilon=0.01, adv_fraction=0.2, adv_steps=7,
                       adv_step_size=None, **_ignored):
    """
    Adversarial CE training — perturb a fraction of each batch with PGD
    (on cross-entropy) before computing the loss.
    """
    if adv_step_size is None:
        adv_step_size = 2.5 * epsilon / max(adv_steps, 1)

    torch.set_grad_enabled(True)
    model = model.to(device).train()
    for p in model.parameters():
        p.requires_grad_(True)

    X_t = _to_float_tensor(X_train)
    y_t = _to_long_tensor(y_train)

    loader = _make_loader(X_t, y_t, batch_size, device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    for epoch in range(epochs):
        total = 0.0
        for bx, by in loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)

            n = bx.shape[0]
            n_adv = max(1, int(n * adv_fraction))
            perm = torch.randperm(n, device=device)
            adv_idx = perm[:n_adv]

            x_adv = bx[adv_idx].clone().detach()
            y_adv = by[adv_idx]

            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)

            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
            for _ in range(adv_steps):
                x_adv = x_adv.detach().requires_grad_(True)
                loss_adv = F.cross_entropy(model(x_adv), y_adv)
                grad = torch.autograd.grad(loss_adv, x_adv)[0]
                x_adv = x_adv.detach() + adv_step_size * grad.sign()
                x_adv = torch.min(torch.max(x_adv, bx[adv_idx] - epsilon),
                                  bx[adv_idx] + epsilon)

            for p in model.parameters():
                p.requires_grad_(True)
            model.train()

            bx_mixed = bx.clone()
            bx_mixed[adv_idx] = x_adv.detach()

            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(bx_mixed), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        scheduler.step()
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, AdvCE: {total/len(loader):.4f}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Clean cross-entropy training
# ═══════════════════════════════════════════════════════════════════════════════

def train_model_ce(model, X_train, y_train, epochs=100, lr=5e-3,
                   batch_size=128, device="cuda", verbose=False, **_ignored):
    """Clean CE training with Adam + cosine LR."""
    torch.set_grad_enabled(True)
    model = model.to(device).train()
    for p in model.parameters():
        p.requires_grad_(True)

    X_t = _to_float_tensor(X_train)
    y_t = _to_long_tensor(y_train)

    loader = _make_loader(X_t, y_t, batch_size, device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    for epoch in range(epochs):
        total = 0.0
        for bx, by in loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        scheduler.step()
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, CE: {total/len(loader):.4f}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Pretraining with cosine LR + early stopping
# ═══════════════════════════════════════════════════════════════════════════════

def pretrain_model_ce(model, X_train, y_train, epochs=500, lr=0.01,
                      batch_size=32, device="cpu", val_fraction=0.0,
                      patience=500, verbose=False, **_ignored):
    """Pretrain h0 with cosine-annealing LR and optional early stopping."""
    torch.set_grad_enabled(True)
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(True)

    if isinstance(X_train, torch.Tensor):
        X_train = X_train.cpu().numpy()
    if isinstance(y_train, torch.Tensor):
        y_train = y_train.cpu().numpy()

    n = len(X_train)
    n_val = max(1, int(n * val_fraction))
    perm = np.random.permutation(n)
    val_idx, trn_idx = perm[:n_val], perm[n_val:]

    X_t = torch.FloatTensor(X_train[trn_idx]).to(device)
    y_t = torch.LongTensor(y_train[trn_idx]).to(device)
    X_v = torch.FloatTensor(X_train[val_idx]).to(device)
    y_v = torch.LongTensor(y_train[val_idx]).to(device)

    loader = DataLoader(TensorDataset(X_t, y_t),
                        batch_size=batch_size, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.95)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss, best_state, wait = float("inf"), None, 0

    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(model(X_v), y_v).item()
        model.train()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch+1} "
                      f"(best val_loss={best_val_loss:.4f})")
            break

        if verbose and (epoch + 1) % 100 == 0:
            print(f"    Pretrain epoch {epoch+1}/{epochs}  "
                  f"val_loss={val_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(f"    Pretraining finished: best val_loss={best_val_loss:.4f}")

    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  TRADES adversarial training (native multi-class)
# ═══════════════════════════════════════════════════════════════════════════════

def _trades_inner_pgd(model, x_clean, epsilon, step_size, num_steps, device):
    """TRADES inner PGD: maximise KL(softmax(clean) ‖ softmax(adv))."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    logits_clean = model(x_clean.detach())
    p_clean = F.softmax(logits_clean, dim=-1).detach()

    x_adv = x_clean.detach() + 0.001 * torch.randn_like(x_clean)

    for _ in range(num_steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits_adv = model(x_adv)
        log_q = F.log_softmax(logits_adv, dim=-1)
        kl = F.kl_div(log_q, p_clean, reduction="sum")
        grad = torch.autograd.grad(kl, x_adv)[0]
        x_adv = x_adv.detach() + step_size * grad.detach().sign()
        x_adv = torch.min(torch.max(x_adv, x_clean.detach() - epsilon),
                          x_clean.detach() + epsilon)

    for p in model.parameters():
        p.requires_grad_(True)
    return x_adv.detach()


def train_model_trades(model, X_train, y_train, epochs=60, lr=1e-4,
                       batch_size=128, device="cuda", verbose=False,
                       epsilon=0.01, trades_beta=6.0, trades_pgd_steps=5,
                       trades_step_size=None, **_ignored):
    """TRADES adversarial training for multi-class CE."""
    if trades_step_size is None:
        trades_step_size = epsilon / 3.0

    model = model.to(device).train()

    X_t = _to_float_tensor(X_train)
    y_t = _to_long_tensor(y_train)
    loader = _make_loader(X_t, y_t, batch_size, device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    for epoch in range(epochs):
        total_loss = 0.0
        for bx, by in loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)

            x_adv = _trades_inner_pgd(model, bx, epsilon,
                                      trades_step_size,
                                      trades_pgd_steps, device)
            model.train()
            optimizer.zero_grad()

            logits_clean = model(bx)
            loss_clean = F.cross_entropy(logits_clean, by)

            p_clean = F.softmax(logits_clean.detach(), dim=-1)
            logits_adv = model(x_adv)
            log_q = F.log_softmax(logits_adv, dim=-1)
            loss_robust = F.kl_div(log_q, p_clean, reduction="batchmean")

            loss = loss_clean + trades_beta * loss_robust
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  "
                  f"loss={total_loss/len(loader):.4f}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Adaptive epoch scaling
# ═══════════════════════════════════════════════════════════════════════════════

def compute_adaptive_epochs(T_size, N_effective, base_epochs=200, min_epochs=30):
    if T_size <= 0 or N_effective <= 0:
        return base_epochs
    ratio = T_size / N_effective
    scaled = int(base_epochs * np.sqrt(ratio))
    return max(min_epochs, min(base_epochs, scaled))


# ═══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _to_float_tensor(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().clone().float()
    return torch.as_tensor(arr, dtype=torch.float32)


def _to_long_tensor(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().clone().long()
    return torch.as_tensor(arr, dtype=torch.long)


def _make_loader(X_t, y_t, batch_size, device):
    if len(X_t) < 10_000:
        X_t, y_t = X_t.to(device), y_t.to(device)
    return DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(len(X_t) >= 10_000 and device != "cpu"),
        num_workers=0,
    )