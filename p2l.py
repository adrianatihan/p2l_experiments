"""
Pick-to-Learn core algorithm — dataset and model agnostic.

Two verifier backends:
  verifier="autolirpa":  PGD inner loop → auto_LiRPA backward (incomplete)
  verifier="abcrown":    α,β-CROWN complete verifier (PGD + BaB built-in)

Stopping condition:
    Stop when ALL remaining examples have worst_case_ce ≤ ce_threshold.
    This ensures every example is both correctly classified AND robust.

Selection ordering:
    worst_case_logits(h, z)[c] = lb[c] if c == y_true, ub[c] if c != y_true
    Total order: CE(worst_case_logits, y_true) descending.
    The example with highest worst-case CE is selected into T first.

Per-iteration reporting:
    Each iteration prints how many examples remain inappropriate
    (wc_ce > ce_threshold or verified-unsafe).
"""

import numpy as np
import torch

from attacks import pgd_attack_bce

LOG2 = float(np.log(2.0))


def pick_to_learn(
    X_data, y_data,
    *,
    model_fn,
    input_shape,
    train_fn,
    pretrain_fn,
    pretrain_portion=0.5,
    epochs=60,
    lr=1e-3,
    retrain_lr=1e-4,
    device="cuda",
    verbose=True,
    epsilon=0.01,
    pgd_steps=20,
    pgd_restarts=5,
    bce_threshold=3.0,
    ce_threshold=None,
    pretrain_epochs=100,
    pretrain_kwargs=None,
    retrain_kwargs=None,
    verifier="autolirpa",
    abcrown_path=None,
    abcrown_timeout=120,
    chunk_size = 16,
):
    """
    Pick-to-Learn with pluggable verification backend.

    Parameters
    ----------
    ce_threshold   : float — worst-case CE must be ≤ this for ALL remaining
                     examples to stop. Default log(2) (decision boundary).
    bce_threshold  : float — PGD phase: empirical worst-case BCE above this
                     triggers adding to T (autolirpa mode only)
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    if ce_threshold is None:
        ce_threshold = LOG2

    pretrain_kwargs = pretrain_kwargs or {}
    retrain_kwargs = retrain_kwargs or {}

    X_data = torch.tensor(X_data, dtype=torch.float32, device=device)
    y_data = torch.tensor(y_data, dtype=torch.long, device=device)

    n_samples = len(X_data)
    n_pretrain = int(n_samples * pretrain_portion)
    pretrain_indices = list(range(n_pretrain))
    available_indices = list(range(n_pretrain, n_samples))
    N_effective = n_samples - n_pretrain

    if verbose:
        tag = "α,β-CROWN (complete)" if verifier == "abcrown" \
              else "PGD → auto_LiRPA (incomplete)"
        print(f"P2L — verifier: {tag}")
        print(f"  Device : {device}"
              + (f"  GPU: {torch.cuda.get_device_name()}"
                 if device == "cuda" else ""))
        print(f"  Samples: {n_samples}  Pretrain: {n_pretrain}  "
              f"P2L pool: {N_effective}")
        print(f"  ε={epsilon:.6f}  Input shape: {input_shape}")
        print(f"  CE threshold: {ce_threshold:.4f}  (log2={LOG2:.4f})")
        if verifier == "autolirpa":
            print(f"  PGD {pgd_steps}×{pgd_restarts}  "
                  f"PGD threshold={bce_threshold}")
        else:
            print(f"  abcrown timeout: {abcrown_timeout}s/instance")

    # ── Step 1: Pretrain ──────────────────────────────────────────────────
    h = model_fn().to(device)
    if n_pretrain > 0:
        if verbose:
            print("\nStep 1: Pretraining...")
        h = pretrain_fn(
            h, X_data[pretrain_indices], y_data[pretrain_indices],
            epochs=pretrain_epochs, lr=lr, device=device,
            verbose=verbose, **pretrain_kwargs,
        )

    # ── Step 2: P2L loop ─────────────────────────────────────────────────
    T_indices = []
    iteration = 0
    stats = {"pgd_resolved": 0, "appropriate": 0, "inappropriate": 0}

    if verbose:
        print(f"\nStep 2: P2L loop\n")

    if verifier == "abcrown":
        _p2l_loop_abcrown(
            h, X_data, y_data, available_indices, pretrain_indices,
            T_indices, stats, iteration,
            model_fn=model_fn, input_shape=input_shape,
            train_fn=train_fn, retrain_lr=retrain_lr, epochs=epochs,
            device=device, verbose=verbose, epsilon=epsilon,
            retrain_kwargs=retrain_kwargs,
            abcrown_path=abcrown_path, abcrown_timeout=abcrown_timeout,
            ce_threshold=ce_threshold, chunk_size=chunk_size,
        )
    else:
        _p2l_loop_autolirpa(
            h, X_data, y_data, available_indices, pretrain_indices,
            T_indices, stats, iteration,
            input_shape=input_shape,
            train_fn=train_fn, retrain_lr=retrain_lr, epochs=epochs,
            device=device, verbose=verbose, epsilon=epsilon,
            pgd_steps=pgd_steps, pgd_restarts=pgd_restarts,
            bce_threshold=bce_threshold, retrain_kwargs=retrain_kwargs,
            ce_threshold=ce_threshold,
        )

    if verbose:
        print(f"\n  Final: |T| = {len(T_indices)} / {N_effective} "
              f"({100 * len(T_indices) / max(1, N_effective):.1f}%)")
        print(f"  Stats: {stats}")

    return h, T_indices, stats


# ═══════════════════════════════════════════════════════════════════════════════
#  auto_LiRPA flow: PGD inner loop → LiRPA gate
# ═══════════════════════════════════════════════════════════════════════════════

def _p2l_loop_autolirpa(
    h, X_data, y_data, available_indices, pretrain_indices,
    T_indices, stats, iteration, *,
    input_shape, train_fn, retrain_lr, epochs,
    device, verbose, epsilon, pgd_steps, pgd_restarts,
    bce_threshold, retrain_kwargs, ce_threshold,
):
    from verification import verify_batch

    threshold = bce_threshold
    prev_deltas = None

    while len(available_indices) > 0:

        # ── Phase A: PGD inner loop ───────────────────────────────────
        pgd_added = 0
        while len(available_indices) > 0:
            iteration += 1
            X_avail = X_data[available_indices]
            y_avail = y_data[available_indices]

            h.eval()
            if prev_deltas is not None and \
               prev_deltas.shape[0] != len(available_indices):
                prev_deltas = None

            pgd_bce, _, new_deltas = pgd_attack_bce(
                h, X_avail, y_avail, epsilon,
                pgd_steps=pgd_steps, pgd_restarts=pgd_restarts,
                device=device, prev_deltas=prev_deltas,
            )
            prev_deltas = new_deltas

            if not (pgd_bce > threshold).any():
                if verbose:
                    print(f"  iter {iteration:4d} | PGD converged "
                          f"(worst={pgd_bce.max().item():.4f} ≤ {threshold}), "
                          f"added {pgd_added} this round")
                break

            worst_local = int(pgd_bce.argmax())
            worst_bce = pgd_bce[worst_local].item()
            worst_global = available_indices[worst_local]
            stats["pgd_resolved"] += 1
            pgd_added += 1
            T_indices.append(worst_global)
            available_indices.remove(worst_global)
            prev_deltas = None

            if verbose:
                print(f"  iter {iteration:4d} | PGD idx {worst_global:5d} "
                      f"| BCE={worst_bce:.4f} | |T|={len(T_indices)}")

            h = _retrain(h, X_data, y_data, T_indices, pretrain_indices,
                         train_fn, retrain_lr, epochs, device, retrain_kwargs)

        if len(available_indices) == 0:
            break

        # ── Phase B: auto_LiRPA ───────────────────────────────────────
        iteration += 1
        if verbose:
            print(f"  iter {iteration:4d} | auto_LiRPA on "
                  f"{len(available_indices)} remaining...")

        statuses = verify_batch(
            h, available_indices, X_data, y_data, epsilon,
            input_shape=input_shape, device=device, verbose=verbose,
        )

        inappropriate, h, iteration = _process_results(
            statuses, stats, h, X_data, y_data,
            available_indices, pretrain_indices, T_indices,
            train_fn, retrain_lr, epochs, device, retrain_kwargs,
            iteration, verbose, verifier_name="LiRPA",
            ce_threshold=ce_threshold,
        )

        if inappropriate is None:
            break


# ═══════════════════════════════════════════════════════════════════════════════
#  α,β-CROWN flow
# ═══════════════════════════════════════════════════════════════════════════════

def _p2l_loop_abcrown(
    h, X_data, y_data, available_indices, pretrain_indices,
    T_indices, stats, iteration, *,
    model_fn, input_shape, train_fn, retrain_lr, epochs,
    device, verbose, epsilon, retrain_kwargs,
    abcrown_path, abcrown_timeout, ce_threshold, chunk_size,
):
    from verification_abcrown import verify_batch_abcrown

    while len(available_indices) > 0:
        iteration += 1

        if verbose:
            print(f"  iter {iteration:4d} | α,β-CROWN on "
                  f"{len(available_indices)} remaining...")

        statuses = verify_batch_abcrown(
            h, available_indices, X_data, y_data, epsilon,
            input_shape=input_shape, device=device,
            abcrown_path=abcrown_path, timeout=abcrown_timeout,
            verbose=verbose,
        )

        inappropriate, h, iteration = _process_results(
            statuses, stats, h, X_data, y_data,
            available_indices, pretrain_indices, T_indices,
            train_fn, retrain_lr, epochs, device, retrain_kwargs,
            iteration, verbose, verifier_name="abcrown",
            ce_threshold=ce_threshold,
        )

        if inappropriate is None:
            break


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared: process results, report counts, pick worst, retrain
# ═══════════════════════════════════════════════════════════════════════════════

def _process_results(
    statuses, stats, h, X_data, y_data,
    available_indices, pretrain_indices, T_indices,
    train_fn, retrain_lr, epochs, device, retrain_kwargs,
    iteration, verbose, verifier_name, ce_threshold,
):
    """
    Process verification results using worst-case CE threshold.

    An example is "appropriate" if:
      - status != "unsafe"  (no real counterexample found)
      - wc_ce ≤ ce_threshold  (correctly classified under worst-case perturbation)

    An example is "inappropriate" if:
      - status == "unsafe"  (counterexample found — always inappropriate)
      - wc_ce > ce_threshold  (misclassified or insufficiently robust at worst case)

    Returns
    -------
    (inappropriate_list, model, iteration) — or (None, model, iteration) if
    all examples are appropriate.
    """
    appropriate = []     # (gidx, wc_ce)
    inappropriate = []   # (gidx, status, wc_ce)

    for gidx, (status, wc_ce) in statuses.items():
        if status == "unsafe":
            # Real counterexample — always inappropriate
            wc_ce = max(wc_ce, ce_threshold + 1.0)
            inappropriate.append((gidx, status, wc_ce))
        elif wc_ce <= ce_threshold:
            appropriate.append((gidx, wc_ce))
        else:
            inappropriate.append((gidx, status, wc_ce))

    n_app = len(appropriate)
    n_inapp = len(inappropriate)
    n_remaining = len(available_indices)

    # Update running stats
    stats["appropriate"] = stats.get("appropriate", 0) + n_app
    stats["inappropriate"] = stats.get("inappropriate", 0) + n_inapp

    if verbose:
        n_unsafe = sum(1 for _, s, _ in inappropriate if s == "unsafe")
        n_ce_fail = n_inapp - n_unsafe
        worst_app = max((ce for _, ce in appropriate), default=0.0)
        worst_inapp = max((ce for _, _, ce in inappropriate), default=0.0)

        print(f"    ── Iteration {iteration} summary ──")
        print(f"    Remaining: {n_remaining}  |  "
              f"Appropriate: {n_app}  |  "
              f"Inappropriate: {n_inapp} "
              f"(unsafe={n_unsafe}, wc_ce>{ce_threshold:.4f}={n_ce_fail})")
        if n_app > 0:
            print(f"    Worst appropriate wc_ce: {worst_app:.4f}")
        if n_inapp > 0:
            print(f"    Worst inappropriate wc_ce: {worst_inapp:.4f}")

    if not inappropriate:
        if verbose:
            print(f"\n  ✓ All {n_remaining} examples appropriate "
                  f"(wc_ce ≤ {ce_threshold:.4f}). Stopping.")
        return None, h, iteration

    # Select worst inappropriate by worst-case CE
    inappropriate.sort(key=lambda t: t[2], reverse=True)
    pick_global, pick_status, pick_wc_ce = inappropriate[0]

    T_indices.append(pick_global)
    available_indices.remove(pick_global)

    if verbose:
        print(f"  iter {iteration:4d} | {verifier_name} added "
              f"idx {pick_global:5d} ({pick_status}, "
              f"wc_ce={pick_wc_ce:.4f}) | |T|={len(T_indices)}")

    h = _retrain(h, X_data, y_data, T_indices, pretrain_indices,
                 train_fn, retrain_lr, epochs, device, retrain_kwargs)

    return inappropriate, h, iteration


# ═══════════════════════════════════════════════════════════════════════════════

def _retrain(h, X_data, y_data, T_indices, pretrain_indices,
             train_fn, lr, epochs, device, extra_kwargs):
    combined = T_indices + pretrain_indices
    return train_fn(
        h, X_data[combined], y_data[combined],
        epochs=epochs, lr=lr, device=device,
        verbose=False, **extra_kwargs,
    )