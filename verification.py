"""
Verification via auto_LiRPA (direct Python API).

Handles skip connections natively — no ONNX conversion needed.
Chunked with OOM fallback to single-sample.

Returns both verification status AND worst-case CE from bounds.
Supports both binary (single logit) and multi-class (C logits) models.
"""

import copy
import os
import sys
import torch
import torch.nn.functional as F

_BoundedModule = None
_BoundedTensor = None
_PerturbationLpNorm = None


def _ensure_autolirpa():
    global _BoundedModule, _BoundedTensor, _PerturbationLpNorm
    if _BoundedModule is not None:
        return
    try:
        from auto_LiRPA import BoundedModule, BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm
        _BoundedModule = BoundedModule
        _BoundedTensor = BoundedTensor
        _PerturbationLpNorm = PerturbationLpNorm
        return
    except ImportError:
        pass
    search_roots = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "alpha-beta-CROWN"),
        "/content/alpha-beta-CROWN",
    ]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, _ in os.walk(root):
            if "auto_LiRPA" in dirnames:
                if dirpath not in sys.path:
                    sys.path.insert(0, dirpath)
                break
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    _BoundedModule = BoundedModule
    _BoundedTensor = BoundedTensor
    _PerturbationLpNorm = PerturbationLpNorm


# ═══════════════════════════════════════════════════════════════════════════════
#  Worst-case CE from bounds (multi-class & binary)
# ═══════════════════════════════════════════════════════════════════════════════

def worst_case_ce_from_bounds(lb, ub, y_batch):
    """
    Compute worst-case cross-entropy from bounds on output logits.

    worst_case_logits[c] = lb[c]  if c == y_true   (minimise true class)
                           ub[c]  if c != y_true   (maximise other classes)

    CE = CrossEntropy(worst_case_logits, y_true)

    Handles both:
      - Single logit binary: lb, ub shape (N,), y ∈ {0, 1}
        → BCE(worst_logit, y) where worst_logit = lb if y=1, ub if y=0
      - Multi-class: lb, ub shape (N, C), y ∈ {0, ..., C-1}
        → NLL(log_softmax(worst_logits), y)
    """
    if lb.dim() == 1:
        # Binary single-logit
        y_f = y_batch.float()
        worst_logit = torch.where(y_batch == 1, lb, ub)
        return F.binary_cross_entropy_with_logits(
            worst_logit, y_f, reduction="none"
        )
    else:
        # Multi-class: (N, C)
        N, C = lb.shape
        idx = torch.arange(N, device=lb.device)
        worst_logits = ub.clone()                      # all classes get upper bound
        worst_logits[idx, y_batch] = lb[idx, y_batch]  # true class gets lower bound
        return F.cross_entropy(worst_logits, y_batch, reduction="none")


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════

def verify_batch(model, indices, X_data, y_data, epsilon,
                 input_shape, device="cuda", verbose=True, chunk_size=4):
    """
    Verify examples using auto_LiRPA backward.

    Returns
    -------
    dict[int → (str, float)]
        global_index → (status, worst_case_ce)
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
            if verbose and len(chunk_idx) > 1:
                print(f"        Chunk {ci+1}: OOM, retrying 1-by-1")
            for gidx in chunk_idx:
                torch.cuda.empty_cache()
                single = _verify_chunk(model, [gidx], X_data, y_data,
                                       epsilon, input_shape, device)
                if single is None:
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
    Verify a small chunk. Returns dict[int → (str, float)] or None on OOM.
    Supports both single-logit (binary) and multi-class outputs.
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

        if x_batch.dim() == 2 and len(input_shape) == 4:
            C, H, W = input_shape[1], input_shape[2], input_shape[3]
            x_batch = x_batch.view(-1, C, H, W)

        ptb = _PerturbationLpNorm(norm=float("inf"), eps=epsilon)
        x_bounded = _BoundedTensor(x_batch, ptb)

        lb, ub = bounded_model.compute_bounds(
            x=(x_bounded,), method="backward",
            bound_upper=True, bound_lower=True)

        lb = lb.detach()
        ub = ub.detach()

        # Determine if binary (single logit) or multi-class
        is_binary = (lb.dim() == 2 and lb.shape[1] == 1)
        if is_binary:
            lb = lb.squeeze(-1)
            ub = ub.squeeze(-1)

        wc_ce = worst_case_ce_from_bounds(lb, ub, y_batch)

        for i, gidx in enumerate(chunk_indices):
            ce_val = wc_ce[i].item()

            if is_binary:
                y_val = int(y_batch[i])
                if y_val == 1:
                    verified = lb[i].item() > 0
                    unsafe = ub[i].item() < 0
                else:
                    verified = ub[i].item() < 0
                    unsafe = lb[i].item() > 0
            else:
                # Multi-class: verified if lb[y] > max_{c≠y} ub[c]
                y_val = int(y_batch[i])
                lb_true = lb[i, y_val].item()
                ub_other = ub[i].clone()
                ub_other[y_val] = -float("inf")
                max_ub_other = ub_other.max().item()
                verified = lb_true > max_ub_other
                # unsafe if ub[y] < max_{c≠y} lb[c]
                ub_true = ub[i, y_val].item()
                lb_other = lb[i].clone()
                lb_other[y_val] = -float("inf")
                max_lb_other = lb_other.max().item()
                unsafe = ub_true < max_lb_other

            if verified:
                status = "verified"
            elif unsafe:
                status = "unsafe"
            else:
                status = "unknown"

            results[gidx] = (status, ce_val)

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