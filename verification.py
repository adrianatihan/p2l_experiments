"""
Verification via auto_LiRPA (direct Python API).

Handles skip connections natively — no ONNX conversion needed.
Chunked with OOM fallback to single-sample.

Returns both verification status AND worst-case BCE from bounds,
so the P2L loop can rank non-verified examples by certified hardness.
"""

import copy
import os
import sys
import torch
import torch.nn.functional as F

# ── Lazy import so the rest of the project works without auto_LiRPA ──────────
_BoundedModule = None
_BoundedTensor = None
_PerturbationLpNorm = None


def _ensure_autolirpa():
    global _BoundedModule, _BoundedTensor, _PerturbationLpNorm
    if _BoundedModule is not None:
        return

    # auto_LiRPA ships inside the α,β-CROWN repo — find it by walking
    project_dir = os.path.dirname(os.path.abspath(__file__))
    abcrown_root = os.path.join(project_dir, "alpha-beta-CROWN")
    if os.path.isdir(abcrown_root):
        for dirpath, dirnames, filenames in os.walk(abcrown_root):
            if "auto_LiRPA" in dirnames:
                candidate = dirpath  # parent of auto_LiRPA/
                if candidate not in sys.path:
                    sys.path.insert(0, candidate)
                break

    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    _BoundedModule = BoundedModule
    _BoundedTensor = BoundedTensor
    _PerturbationLpNorm = PerturbationLpNorm


# ═══════════════════════════════════════════════════════════════════════════════
#  Worst-case logit → CE ordering
# ═══════════════════════════════════════════════════════════════════════════════

def _worst_case_bce_from_bounds(lb, ub, y_batch):
    """
    Compute worst-case BCE from CROWN bounds on a single-logit model.

    Given bounds [lb, ub] on the output logit:
      worst_case_logit = lb  if y=1  (true class wants logit > 0, worst is min)
                         ub  if y=0  (true class wants logit < 0, worst is max)

    Returns BCE(worst_case_logit, y) per example — the cross-entropy
    evaluated at the worst-case logit under the perturbation set.
    """
    y_f = y_batch.float()
    worst_logit = torch.where(y_batch == 1, lb, ub)
    return F.binary_cross_entropy_with_logits(
        worst_logit, y_f, reduction="none"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════

def verify_batch(model, indices, X_data, y_data, epsilon,
                 input_shape, device="cuda", verbose=True, chunk_size=4):
    """
    Verify examples using auto_LiRPA backward (alpha-CROWN).

    Parameters
    ----------
    model       : nn.Module — single-logit binary classifier
    indices     : list[int] — which rows of X_data to verify
    X_data      : tensor or ndarray, full dataset
    y_data      : tensor or ndarray, full labels
    epsilon     : float — L∞ perturbation radius
    input_shape : tuple — e.g. (1, 3, 32, 32) or (1, 784)
    device      : str
    verbose     : bool
    chunk_size  : int — samples per auto_LiRPA call

    Returns
    -------
    dict[int → (str, float)]
        global_index → (status, worst_case_bce)
        status is 'verified' | 'unsafe' | 'unknown'
        worst_case_bce is the BCE at the worst-case logit from bounds
        (inf for OOM / unknown with no bounds)
    """
    _ensure_autolirpa()

    if len(indices) == 0:
        return {}

    results = {}
    chunks = [indices[i:i + chunk_size]
              for i in range(0, len(indices), chunk_size)]

    if verbose:
        print(f"    Verifying {len(indices)} samples in "
              f"{len(chunks)} chunks (≤{chunk_size}) via auto_LiRPA")

    total_ver, total_unf, total_unk = 0, 0, 0

    for ci, chunk_idx in enumerate(chunks):
        torch.cuda.empty_cache()

        res = _verify_chunk(model, chunk_idx, X_data, y_data,
                            epsilon, input_shape, device)

        if res is None:
            # OOM → single-sample fallback
            if verbose and len(chunk_idx) > 1:
                print(f"        Chunk {ci+1}: OOM, retrying 1-by-1")
            for gidx in chunk_idx:
                torch.cuda.empty_cache()
                single = _verify_chunk(model, [gidx], X_data, y_data,
                                       epsilon, input_shape, device)
                if single is None:
                    # Total OOM — unknown with infinite worst-case BCE
                    results[gidx] = ("unknown", float("inf"))
                    total_unk += 1
                else:
                    results.update(single)
                    for status, _ in single.values():
                        total_ver += status == "verified"
                        total_unf += status == "unsafe"
                        total_unk += status == "unknown"
        else:
            results.update(res)
            for status, _ in res.values():
                total_ver += status == "verified"
                total_unf += status == "unsafe"
                total_unk += status == "unknown"

        if verbose and (ci + 1) % max(1, len(chunks) // 5) == 0:
            print(f"        Progress: {ci+1}/{len(chunks)} chunks  "
                  f"({total_ver} verified, {total_unf} unsafe, {total_unk} unknown)")

    if verbose:
        print(f"        Done: {total_ver} verified, "
              f"{total_unf} unsafe, {total_unk} unknown")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Internal
# ═══════════════════════════════════════════════════════════════════════════════

def _verify_chunk(model, chunk_indices, X_data, y_data,
                  epsilon, input_shape, device):
    """
    Verify a small chunk.
    Returns dict[int → (str, float)] or None on OOM.
    """
    results = {}

    try:
        model_copy = copy.deepcopy(model).to(device).eval()
        dummy = torch.zeros(*input_shape, device=device)
        bounded_model = _BoundedModule(model_copy, dummy,
                                       device=device, verbose=0)

        if isinstance(X_data, torch.Tensor):
            x_batch = X_data[chunk_indices].float().to(device)
            y_batch = y_data[chunk_indices].to(device)
        else:
            x_batch = torch.FloatTensor(X_data[chunk_indices]).to(device)
            y_batch = torch.LongTensor(
                [int(y_data[i]) for i in chunk_indices]).to(device)

        # Reshape flat vectors to expected input shape if needed
        if x_batch.dim() == 2 and len(input_shape) == 4:
            C, H, W = input_shape[1], input_shape[2], input_shape[3]
            x_batch = x_batch.view(-1, C, H, W)

        ptb = _PerturbationLpNorm(norm=float("inf"), eps=epsilon)
        x_bounded = _BoundedTensor(x_batch, ptb)

        lb, ub = bounded_model.compute_bounds(
            x=(x_bounded,), method="backward",
            bound_upper=True, bound_lower=True)

        lb = lb.detach().squeeze(-1)
        ub = ub.detach().squeeze(-1)

        # Compute worst-case BCE from bounds for all examples in chunk
        wc_bce = _worst_case_bce_from_bounds(lb, ub, y_batch)

        for i, gidx in enumerate(chunk_indices):
            y_val = int(y_batch[i])
            bce_val = wc_bce[i].item()

            if y_val == 1:
                if lb[i].item() > 0:
                    status = "verified"
                elif ub[i].item() < 0:
                    status = "unsafe"
                else:
                    status = "unknown"
            else:
                if ub[i].item() < 0:
                    status = "verified"
                elif lb[i].item() > 0:
                    status = "unsafe"
                else:
                    status = "unknown"

            results[gidx] = (status, bce_val)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return None
        raise
    finally:
        try:
            del bounded_model, model_copy, x_batch, x_bounded
        except NameError:
            pass
        torch.cuda.empty_cache()

    return results