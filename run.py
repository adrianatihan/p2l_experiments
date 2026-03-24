"""
Run P2L — change DATASET to switch between tasks.

    DATASET = "mnist"     →  BinaryMLP (784→1), ε=0.1
    DATASET = "cifar10"   →  CifarResNet (3×32×32→1), ε=2/255

Verification backends:
    VERIFIER = "autolirpa"  →  PGD inner loop + auto_LiRPA backward (incomplete)
    VERIFIER = "abcrown"    →  α,β-CROWN complete verifier (PGD + BaB built-in)
"""

import os
import time
import warnings
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from data import load_dataset
from models import get_model_fn
from training import train_model_bce, train_model_trades, pretrain_model_bce
from bounds import compute_generalization_bound
from utils import evaluate_model_binary, fmt_time
from p2l import pick_to_learn

warnings.filterwarnings("ignore")
np.random.seed(11)
torch.manual_seed(11)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION — edit this section to change dataset / model / params   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── Dataset (just change this one line to switch tasks) ──────────────────
DATASET = "mnist"          # "mnist" | "cifar10"

# Default model per dataset — override MODEL below if you want a custom one
DATASET_DEFAULTS = {
    "mnist":   {"model": "mnist_mlp",    "epsilon": 0.1},
    "cifar10": {"model": "cifar_resnet", "epsilon": 2.0 / 255.0},
}

MODEL   = DATASET_DEFAULTS[DATASET]["model"]
EPSILON = DATASET_DEFAULTS[DATASET]["epsilon"]

# ── Verifier backend ─────────────────────────────────────────────────────
VERIFIER = "abcrown"       # "autolirpa" | "abcrown"

# α,β-CROWN settings (only used when VERIFIER = "abcrown")
def _find_abcrown_path():
    """Walk alpha-beta-CROWN/ to find the directory containing abcrown.py."""
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "alpha-beta-CROWN")
    for dirpath, _, filenames in os.walk(root):
        if "abcrown.py" in filenames:
            return dirpath
    return root  # fallback — will error later with a clear message

ABCROWN_PATH    = _find_abcrown_path()
ABCROWN_TIMEOUT = 120        # per-instance timeout in seconds

# ── Data ──────────────────────────────────────────────────────────────────
N_SAMPLES  = 2000
TEST_SIZE  = 0.2

# ── P2L ───────────────────────────────────────────────────────────────────
PRETRAIN_PORTION = 0.5
DELTA            = 0.05      # confidence parameter for the bound

# ── Training ──────────────────────────────────────────────────────────────
PRETRAIN_EPOCHS = 100
RETRAIN_EPOCHS  = 60
PRETRAIN_LR     = 1e-3       # Adam LR for pretraining
RETRAIN_LR      = 1e-4       # Adam LR for retraining inside P2L loop

# ── Perturbation / attack ────────────────────────────────────────────────
# EPSILON is set above from DATASET_DEFAULTS (override here if needed)
PGD_STEPS    = 20
PGD_RESTARTS = 5

# ── TRADES (used as the retrain strategy) ─────────────────────────────────
USE_TRADES        = True      # False → plain BCE retraining
TRADES_BETA       = 6.0
TRADES_PGD_STEPS  = 5
TRADES_STEP_SIZE  = None      # None → epsilon / 3

# ── BCE threshold for PGD phase ──────────────────────────────────────────
BCE_THRESHOLD = 3.0

# ── Device ────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.perf_counter()

    # ── Load data ─────────────────────────────────────────────────────────
    X, y = load_dataset(DATASET, n_samples=N_SAMPLES)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y,
    )

    # ── Resolve model factory ─────────────────────────────────────────────
    model_cls, input_shape = get_model_fn(MODEL)
    model_fn = model_cls

    # ── Choose retrain strategy ───────────────────────────────────────────
    if USE_TRADES:
        retrain_fn = train_model_trades
        retrain_kwargs = dict(
            epsilon=EPSILON,
            trades_beta=TRADES_BETA,
            trades_pgd_steps=TRADES_PGD_STEPS,
            trades_step_size=TRADES_STEP_SIZE,
        )
    else:
        retrain_fn = train_model_bce
        retrain_kwargs = {}

    # ── Print config ──────────────────────────────────────────────────────
    print(f"P2L — {DATASET.upper()} — {MODEL}")
    print(f"Device: {DEVICE}"
          + (f"  GPU: {torch.cuda.get_device_name()}" if DEVICE == "cuda" else ""))
    print(f"Verifier: {VERIFIER}"
          + (f"  (timeout={ABCROWN_TIMEOUT}s)" if VERIFIER == "abcrown" else ""))
    print(f"Data: {len(X_train)} train, {len(X_test)} test  "
          f"shape={X_train.shape[1:]}")
    print(f"ε={EPSILON:.6f}")
    if VERIFIER == "autolirpa":
        print(f"PGD {PGD_STEPS}×{PGD_RESTARTS}  threshold={BCE_THRESHOLD}")
    if USE_TRADES:
        print(f"Retrain: TRADES  β={TRADES_BETA}  "
              f"inner_steps={TRADES_PGD_STEPS}")
    else:
        print(f"Retrain: clean BCE")
    print()

    # ── Run P2L ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    h, T_indices, stats = pick_to_learn(
        X_train, y_train,
        model_fn=model_fn,
        input_shape=input_shape,
        train_fn=retrain_fn,
        pretrain_fn=pretrain_model_bce,
        pretrain_portion=PRETRAIN_PORTION,
        epochs=RETRAIN_EPOCHS,
        lr=PRETRAIN_LR,
        retrain_lr=RETRAIN_LR,
        device=DEVICE,
        verbose=True,
        epsilon=EPSILON,
        pgd_steps=PGD_STEPS,
        pgd_restarts=PGD_RESTARTS,
        bce_threshold=BCE_THRESHOLD,
        pretrain_epochs=PRETRAIN_EPOCHS,
        retrain_kwargs=retrain_kwargs,
        verifier=VERIFIER,
        abcrown_path=ABCROWN_PATH,
        abcrown_timeout=ABCROWN_TIMEOUT,
    )
    print(f"\n[Timing] P2L: {fmt_time(time.perf_counter() - t0)}")

    # ── Generalization bound ──────────────────────────────────────────────
    N_eff = len(X_train) - int(len(X_train) * PRETRAIN_PORTION)
    bound = compute_generalization_bound(len(T_indices), N_eff, DELTA)

    # ── Test evaluation ───────────────────────────────────────────────────
    test_err, test_acc = evaluate_model_binary(h, X_test, y_test, DEVICE)

    # ── Robust verification on test set (same backend as training) ────────
    test_indices = list(range(len(X_test)))

    if VERIFIER == "abcrown":
        from verification_abcrown import verify_batch_abcrown
        print(f"\nRunning α,β-CROWN verification on test set...")
        test_statuses = verify_batch_abcrown(
            h, test_indices, X_test, y_test, EPSILON,
            input_shape=input_shape, device=DEVICE,
            abcrown_path=ABCROWN_PATH, timeout=ABCROWN_TIMEOUT,
            verbose=True,
        )
    else:
        from verification import verify_batch
        print(f"\nRunning auto_LiRPA verification on test set...")
        test_statuses = verify_batch(
            h, test_indices, X_test, y_test, EPSILON,
            input_shape=input_shape, device=DEVICE, verbose=True,
        )

    n_verified = sum(1 for s, _ in test_statuses.values() if s == "verified")
    n_unsafe   = sum(1 for s, _ in test_statuses.values() if s == "unsafe")
    n_unknown  = sum(1 for s, _ in test_statuses.values() if s == "unknown")
    robust_err = (n_unsafe + n_unknown) / len(X_test)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"|T| = {len(T_indices)} / {N_eff}  "
          f"({100 * len(T_indices) / N_eff:.1f}%)")
    print(f"Bound ε(|T|,δ) = {bound:.4f}  (δ={DELTA})")
    print(f"Test error: {test_err:.4f}  Accuracy: {test_acc:.4f}")
    print(f"Robust error ({VERIFIER}): {robust_err:.4f}  "
          f"(unsafe+unknown = not verified)")
    print(f"  verified={n_verified}  unsafe={n_unsafe}  unknown={n_unknown}")
    print(f"Bound holds: clean={'✓' if test_err <= bound else '✗'}  "
          f"robust={'✓' if robust_err <= bound else '✗'}")
    print(f"\nBreakdown: {stats}")
    print(f"{'=' * 60}")
    print(f"\nTotal: {fmt_time(time.perf_counter() - t_start)}")

    # ── Save ──────────────────────────────────────────────────────────────
    torch.save(h.state_dict(), "p2l_model.pt")
    np.save("T_indices.npy", np.array(T_indices))
    print("Saved p2l_model.pt and T_indices.npy")


if __name__ == "__main__":
    main()