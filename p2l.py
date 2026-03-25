"""
Pick-to-Learn core algorithm — dataset and model agnostic.

Two verifier backends:

  verifier="autolirpa" (default):
      PGD inner loop exhausts attackable examples, then auto_LiRPA backward
      (incomplete) verifies the rest. Fast but may leave many unknowns.

  verifier="abcrown":
      No separate PGD — α,β-CROWN runs its own PGD + alpha-CROWN + BaB
      (complete verification) on every remaining example each round.
      Slower per round but gives definitive verified/unsafe verdicts.

Both: stop only when ALL remaining examples are verified.
      Unknown = not verified (conservative).

Selection ordering:
    worst_case_logit(h, z) = lb if y=1, ub if y=0
    Pick the non-verified example with highest BCE(worst_case_logit, y).
"""

import torch
import numpy as np

from attacks import pgd_attack_bce


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
    pretrain_epochs=100,
    pretrain_kwargs=None,
    retrain_kwargs=None,
    verifier="autolirpa",
    abcrown_path=None,
    abcrown_timeout=120,
):
    """
    Pick-to-Learn with pluggable verification backend.

    Parameters
    ----------
    verifier       : "autolirpa" | "abcrown"
    abcrown_path   : str — path to alpha-beta-CROWN/complete_verifier/
                     (required when verifier="abcrown")
    abcrown_timeout : int — per-instance timeout for α,β-CROWN
    bce_threshold  : float — PGD phase only (autolirpa mode)
    pgd_steps, pgd_restarts : PGD config (autolirpa mode only)

    Returns
    -------
    model, T_indices, stats
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

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
              f"P2L pool: {len(available_indices)}")
        print(f"  ε={epsilon:.6f}  Input shape: {input_shape}")
        if verifier == "autolirpa":
            print(f"  PGD {pgd_steps}×{pgd_restarts}  "
                  f"threshold={bce_threshold}")
        else:
            print(f"  abcrown timeout: {abcrown_timeout}s/instance")

    # ── Step 1: Pretrain ──────────────────────────────────────────────────
    h = model_fn().to(device)

    if n_pretrain > 0:
        if verbose:
            print("\nStep 1: Pretraining...")
        h = pretrain_fn(
            h,
            X_data[pretrain_indices],
            y_data[pretrain_indices],
            epochs=pretrain_epochs,
            lr=lr,
            device=device,
            verbose=verbose,
            **pretrain_kwargs,
        )

    # ── Step 2: P2L loop ─────────────────────────────────────────────────
    T_indices = []
    iteration = 0
    stats = {
        "pgd_resolved": 0,
        "verified": 0,
        "falsified": 0,
        "unknown": 0,
    }

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
    bce_threshold, retrain_kwargs,
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

        non_verified, h, iteration = _process_verification_results_missclassification(
            statuses, stats, h, X_data, y_data,
            available_indices, pretrain_indices, T_indices,
            train_fn, retrain_lr, epochs, device, retrain_kwargs,
            iteration, verbose, verifier_name="LiRPA", ce_threshold=float(np.log(2.0))
        )

        if non_verified is None:
            break  # all verified


# ═══════════════════════════════════════════════════════════════════════════════
#  α,β-CROWN flow: no PGD — abcrown handles attack + BaB
# ═══════════════════════════════════════════════════════════════════════════════

def _p2l_loop_abcrown(
    h, X_data, y_data, available_indices, pretrain_indices,
    T_indices, stats, iteration, *,
    model_fn, input_shape, train_fn, retrain_lr, epochs,
    device, verbose, epsilon, retrain_kwargs,
    abcrown_path, abcrown_timeout,
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

        non_verified, h, iteration = _process_verification_results_missclassification(
            statuses, stats, h, X_data, y_data,
            available_indices, pretrain_indices, T_indices,
            train_fn, retrain_lr, epochs, device, retrain_kwargs,
            iteration, verbose, verifier_name="abcrown", ce_threshold=float(np.log(2.0))
        )

        if non_verified is None:
            break  # all verified


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared: process verification results, pick worst, retrain
# ═══════════════════════════════════════════════════════════════════════════════

def _process_verification_results(
    statuses, stats, h, X_data, y_data,
    available_indices, pretrain_indices, T_indices,
    train_fn, retrain_lr, epochs, device, retrain_kwargs,
    iteration, verbose, verifier_name,
):
    """
    Shared logic for both backends after verification returns.

    Returns
    -------
    (non_verified_list, model, iteration) or (None, model, iteration) if all
    verified.
    """
    non_verified = []
    n_ver = 0
    for gidx, (status, wc_bce) in statuses.items():
        if status == "verified":
            stats["verified"] += 1
            n_ver += 1
        else:
            if status == "unsafe":
                stats["falsified"] += 1
            else:
                stats["unknown"] += 1
            non_verified.append((gidx, status, wc_bce))

    n_not = len(non_verified)
    if verbose:
        nu = sum(1 for _, s, _ in non_verified if s == "unsafe")
        nk = n_not - nu
        print(f"    → {n_ver} verified, {nu} unsafe, {nk} unknown "
              f"({n_not} not verified)")

    if not non_verified:
        if verbose:
            print(f"\n  All {len(available_indices)} examples certified. "
                  f"Stopping.")
        return None, h, iteration

    # Select worst non-verified by worst-case CE from bounds
    non_verified.sort(key=lambda t: t[2], reverse=True)
    pick_global, pick_status, pick_wc_bce = non_verified[0]

    T_indices.append(pick_global)
    available_indices.remove(pick_global)

    if verbose:
        print(f"  iter {iteration:4d} | {verifier_name} added "
              f"idx {pick_global:5d} ({pick_status}, "
              f"wc_bce={pick_wc_bce:.4f}) | |T|={len(T_indices)}")

    h = _retrain(h, X_data, y_data, T_indices, pretrain_indices,
                 train_fn, retrain_lr, epochs, device, retrain_kwargs)

    return non_verified, h, iteration

def _process_verification_results_missclassification(
    statuses, stats, h, X_data, y_data,
    available_indices, pretrain_indices, T_indices,
    train_fn, retrain_lr, epochs, device, retrain_kwargs,
    iteration, verbose, verifier_name, ce_threshold,
):
    """
    Shared logic for both backends after verification returns.
 
    An example is "appropriate" if wc_bce ≤ ce_threshold — meaning the
    model correctly classifies it even under worst-case perturbation.
 
    If the verifier found an actual counterexample (status="unsafe"),
    the example is inappropriate regardless of wc_bce (the bounds may
    be loose, but the counterexample is real).
 
    Returns
    -------
    (inappropriate_list, model, iteration) or (None, model, iteration)
    if all examples are appropriate.
    """
    appropriate = []     # (gidx, wc_bce) — below threshold
    inappropriate = []   # (gidx, status, wc_bce) — above threshold or unsafe
    stats['appropriate'] = 0
    stats['inappropriate'] = 0
    for gidx, (status, wc_bce) in statuses.items():
        # If verifier found a real counterexample, the example is
        # inappropriate no matter what the bounds say
        if status == "unsafe":
            # Ensure wc_bce reflects this (bounds can be loose)
            wc_bce = max(wc_bce, ce_threshold + 1.0)
            inappropriate.append((gidx, status, wc_bce))
            stats["inappropriate"] += 1
        elif wc_bce <= ce_threshold:
            appropriate.append((gidx, wc_bce))
            stats["appropriate"] += 1
        else:
            # wc_bce > threshold: not yet appropriate (could be unknown
            # or verified-with-loose-bounds)
            inappropriate.append((gidx, status, wc_bce))
            stats["inappropriate"] += 1
 
    n_app = len(appropriate)
    n_inapp = len(inappropriate)
 
    if verbose:
        worst_app = max((b for _, b in appropriate), default=0.0)
        worst_inapp = max((b for _, _, b in inappropriate), default=0.0)
        print(f"    → {n_app} appropriate (wc_bce ≤ {ce_threshold:.4f}), "
              f"{n_inapp} inappropriate")
        if n_app > 0:
            print(f"      worst appropriate wc_bce: {worst_app:.4f}")
        if n_inapp > 0:
            print(f"      worst inappropriate wc_bce: {worst_inapp:.4f}")
 
    if not inappropriate:
        if verbose:
            print(f"\n  All {len(available_indices)} examples appropriate "
                  f"(wc_bce ≤ {ce_threshold:.4f}). Stopping.")
        return None, h, iteration
 
    # Select worst inappropriate by worst-case CE from bounds
    inappropriate.sort(key=lambda t: t[2], reverse=True)
    pick_global, pick_status, pick_wc_bce = inappropriate[0]
 
    T_indices.append(pick_global)
    available_indices.remove(pick_global)
 
    if verbose:
        print(f"  iter {iteration:4d} | {verifier_name} added "
              f"idx {pick_global:5d} ({pick_status}, "
              f"wc_bce={pick_wc_bce:.4f}) | |T|={len(T_indices)}")
 
    h = _retrain(h, X_data, y_data, T_indices, pretrain_indices,
                 train_fn, retrain_lr, epochs, device, retrain_kwargs)
 
    return inappropriate, h, iteration
# ═══════════════════════════════════════════════════════════════════════════════

def _retrain(h, X_data, y_data, T_indices, pretrain_indices,
             train_fn, lr, epochs, device, extra_kwargs):
    """Retrain the model on T ∪ pretrain data."""
    combined = T_indices + pretrain_indices
    return train_fn(
        h,
        X_data[combined],
        y_data[combined],
        epochs=epochs,
        lr=lr,
        device=device,
        verbose=False,
        **extra_kwargs,
    )