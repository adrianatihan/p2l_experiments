"""
Semantic perturbation support for P2L.


Supported perturbation types:
  "linf"                
  "brightness"          
  "contrast"            
"""

import copy
import os
import sys
import tempfile
import shutil
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
#  Wrapper models — prepend semantic perturbation layers
# ═══════════════════════════════════════════════════════════════════════════════

class BrightnessWrapper(nn.Module):
    """Prepend a brightness perturbation layer to any base model.

    """

    def __init__(self, base_model, clamp_output=True):
        super().__init__()
        self.base_model = base_model
        self.clamp_output = clamp_output

    def forward(self, x, beta):
        if x.dim() == 4:                       # (N, C, H, W)
            beta = beta.view(-1, 1, 1, 1)
        elif x.dim() == 2:                     # (N, D)
            pass                               # beta is (N, 1), broadcasts
        x_perturbed = x + beta
        if self.clamp_output:
            x_perturbed = x_perturbed.clamp(0.0, 1.0)
        return self.base_model(x_perturbed)


class ContrastWrapper(nn.Module):
    """Prepend a contrast perturbation layer.

    """

    def __init__(self, base_model, clamp_output=True):
        super().__init__()
        self.base_model = base_model
        self.clamp_output = clamp_output

    def forward(self, x, alpha):
        if x.dim() == 4:
            alpha = alpha.view(-1, 1, 1, 1)
        factor = 1.0 + alpha
        x_perturbed = factor * x
        if self.clamp_output:
            x_perturbed = x_perturbed.clamp(0.0, 1.0)
        return self.base_model(x_perturbed)


class BrightnessContrastWrapper(nn.Module):


    def __init__(self, base_model, clamp_output=True):
        super().__init__()
        self.base_model = base_model
        self.clamp_output = clamp_output

    def forward(self, x, params):
        beta = params[:, 0:1]
        alpha = params[:, 1:2]
        if x.dim() == 4:
            beta = beta.unsqueeze(-1).unsqueeze(-1)    # (N,1,1,1)
            alpha = alpha.unsqueeze(-1).unsqueeze(-1)
        x_perturbed = (1.0 + alpha) * x + beta
        if self.clamp_output:
            x_perturbed = x_perturbed.clamp(0.0, 1.0)
        return self.base_model(x_perturbed)


def get_wrapper(base_model, perturbation_type, clamp_output=True):

    if perturbation_type == "brightness":
        wrapper = BrightnessWrapper(base_model, clamp_output=clamp_output)
        n_params = 1
        def nominal(N, device):
            return torch.zeros(N, 1, device=device)
        return wrapper, n_params, nominal

    elif perturbation_type == "contrast":
        wrapper = ContrastWrapper(base_model, clamp_output=clamp_output)
        n_params = 1
        def nominal(N, device):
            return torch.zeros(N, 1, device=device)
        return wrapper, n_params, nominal

    elif perturbation_type == "brightness_contrast":
        wrapper = BrightnessContrastWrapper(base_model, clamp_output=clamp_output)
        n_params = 2
        def nominal(N, device):
            return torch.zeros(N, 2, device=device)
        return wrapper, n_params, nominal

    else:
        raise ValueError(f"Unknown perturbation_type: {perturbation_type}")



def pgd_attack_semantic(model, X, y, epsilon, perturbation_type="brightness",
                        pgd_steps=20, pgd_restarts=5, step_size=None,
                        device="cuda", clamp_pixels=True):

    if step_size is None:
        step_size = 2.5 * epsilon / pgd_steps

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    X = X.to(device).detach()
    y_f = y.float().to(device).detach()
    N = X.shape[0]

    # Determine n_params
    if perturbation_type == "brightness_contrast":
        n_params = 2
    else:
        n_params = 1

    best_bce = torch.full((N,), -float("inf"), device=device)
    best_params = torch.zeros(N, n_params, device=device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    for r in range(pgd_restarts):
        # Random init in [-ε, ε]
        params = torch.empty(N, n_params, device=device).uniform_(-epsilon, epsilon)

        for _ in range(pgd_steps):
            params = params.detach().requires_grad_(True)
            x_pert = _apply_semantic(X, params, perturbation_type,
                                     clamp=clamp_pixels)
            loss = criterion(model(x_pert).squeeze(-1), y_f)
            grad = torch.autograd.grad(loss.sum(), params)[0]
            params = params.detach() + step_size * grad.detach().sign()
            params = params.clamp(-epsilon, epsilon)

        with torch.no_grad():
            x_pert = _apply_semantic(X, params, perturbation_type,
                                     clamp=clamp_pixels)
            bce = criterion(model(x_pert.detach()).squeeze(-1), y_f)
            improved = bce > best_bce
            best_bce[improved] = bce[improved]
            best_params[improved] = params[improved]

    for p in model.parameters():
        p.requires_grad_(True)

    # Compute best adversarial inputs in pixel space
    with torch.no_grad():
        best_inputs = _apply_semantic(X, best_params, perturbation_type,
                                      clamp=clamp_pixels)

    return best_bce.detach(), best_inputs.detach(), best_params.detach()


def _apply_semantic(x, params, perturbation_type, clamp=True):
    """Apply semantic perturbation in pixel space (differentiable)."""
    if perturbation_type == "brightness":
        beta = params[:, 0:1]
        if x.dim() == 4:
            beta = beta.unsqueeze(-1).unsqueeze(-1)
        out = x + beta

    elif perturbation_type == "contrast":
        alpha = params[:, 0:1]
        if x.dim() == 4:
            alpha = alpha.unsqueeze(-1).unsqueeze(-1)
        out = (1.0 + alpha) * x

    elif perturbation_type == "brightness_contrast":
        beta = params[:, 0:1]
        alpha = params[:, 1:2]
        if x.dim() == 4:
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            alpha = alpha.unsqueeze(-1).unsqueeze(-1)
        out = (1.0 + alpha) * x + beta

    else:
        raise ValueError(f"Unknown perturbation_type: {perturbation_type}")

    if clamp:
        out = out.clamp(0.0, 1.0)
    return out



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
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "alpha-beta-CROWN"),
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



def verify_batch_semantic(model, indices, X_data, y_data, epsilon,
                          input_shape, perturbation_type="brightness",
                          device="cuda", verbose=True, chunk_size=1,
                          clamp_pixels=True):
    
    _ensure_autolirpa()

    if len(indices) == 0:
        return {}

    results = {}
    chunks = [indices[i:i + chunk_size]
              for i in range(0, len(indices), chunk_size)]

    if verbose:
        print(f"    Verifying {len(indices)} samples "
              f"({perturbation_type}, ε={epsilon:.6f}) via auto_LiRPA "
              f"[CROWN A-matrix method]")

    total_ver, total_unf, total_unk = 0, 0, 0

    for ci, chunk_idx in enumerate(chunks):
        torch.cuda.empty_cache()

        res = _verify_chunk_semantic(
            model, chunk_idx, X_data, y_data, epsilon,
            input_shape, perturbation_type, device, clamp_pixels)

        if res is None:
            if verbose and len(chunk_idx) > 1:
                print(f"        Chunk {ci+1}: OOM, retrying 1-by-1")
            for gidx in chunk_idx:
                torch.cuda.empty_cache()
                single = _verify_chunk_semantic(
                    model, [gidx], X_data, y_data, epsilon,
                    input_shape, perturbation_type, device, clamp_pixels)
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
                  f"({total_ver} verified, {total_unf} unsafe, "
                  f"{total_unk} unknown)")

    if verbose:
        print(f"        Done: {total_ver} verified, "
              f"{total_unf} unsafe, {total_unk} unknown")

    return results


def _semantic_bounds_from_A(lA, uA, lbias, ubias, x_batch, epsilon,
                            perturbation_type):

    N = lA.shape[0]
    out_dim = lA.shape[1]
    lA_flat = lA.reshape(N, out_dim, -1)     # (N, out, D)
    uA_flat = uA.reshape(N, out_dim, -1)
    x_flat = x_batch.reshape(N, -1)          # (N, D)

    if perturbation_type == "brightness":
        # x' = x₀ + β·1,  β ∈ [-ε, ε]
        # f(x') ≥ A_L·x₀ + b_L + β · sum(A_L)
        # min over β:  − ε · |sum(A_L)|
        sum_lA = lA_flat.sum(dim=2)                          # (N, out)
        sum_uA = uA_flat.sum(dim=2)
        lb = (lA_flat * x_flat.unsqueeze(1)).sum(2) + lbias - epsilon * sum_lA.abs()
        ub = (uA_flat * x_flat.unsqueeze(1)).sum(2) + ubias + epsilon * sum_uA.abs()

    elif perturbation_type == "contrast":
        # x' = (1 + α)·x₀,  α ∈ [-ε, ε]
        # f(x') ≥ A_L·((1+α)·x₀) + b_L = (A_L·x₀ + b_L) + α·(A_L·x₀)
        # min over α:  − ε · |A_L · x₀|  (element-wise, then sum)
        # Note: A_L · x₀ here means the dot product per output dim
        lAx = (lA_flat * x_flat.unsqueeze(1)).sum(2)        # (N, out)
        uAx = (uA_flat * x_flat.unsqueeze(1)).sum(2)
        lb = lAx + lbias - epsilon * lAx.abs()
        ub = uAx + ubias + epsilon * uAx.abs()

    elif perturbation_type == "brightness_contrast":
        # x' = (1+α)·x₀ + β,  (α, β) ∈ [-ε, ε]²
        # f(x') ≥ A_L·x₀ + b_L + α·(A_L·x₀) + β·sum(A_L)
        # min: −ε·|A_L·x₀| − ε·|sum(A_L)|
        lAx = (lA_flat * x_flat.unsqueeze(1)).sum(2)
        uAx = (uA_flat * x_flat.unsqueeze(1)).sum(2)
        sum_lA = lA_flat.sum(dim=2)
        sum_uA = uA_flat.sum(dim=2)
        lb = lAx + lbias - epsilon * lAx.abs() - epsilon * sum_lA.abs()
        ub = uAx + ubias + epsilon * uAx.abs() + epsilon * sum_uA.abs()

    else:
        raise ValueError(f"Unknown perturbation_type: {perturbation_type}")

    return lb, ub


def _verify_chunk_semantic(model, chunk_indices, X_data, y_data,
                           epsilon, input_shape, perturbation_type,
                           device, clamp_pixels):

    _ensure_autolirpa()
    results = {}

    try:
        model_copy = copy.deepcopy(model).to(device).eval()

        # ── Prepare data ──────────────────────────────────────────────
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

        N = x_batch.shape[0]

        # ── Build BoundedModule on the base model (single input) ──────
        dummy = torch.zeros(*input_shape, device=device)
        bounded_model = _BoundedModule(model_copy, dummy,
                                       device=device, verbose=0)

        # ── Run CROWN with L∞ eps to get intermediate bounds, ─────────
        #    then extract the A matrices w.r.t. the input.
        #    The L∞ intermediate bounds are a sound over-approximation
        #    of the brightness set, so the A matrices are valid.
        ptb = _PerturbationLpNorm(norm=float("inf"), eps=epsilon)
        x_bounded = _BoundedTensor(x_batch, ptb)

        required_A = {
            bounded_model.output_name[0]: {
                bounded_model.input_name[0]: ['lA', 'uA', 'lbias', 'ubias']
            }
        }

        _lb, _ub, A_dict = bounded_model.compute_bounds(
            x=(x_bounded,), method="backward",
            return_A=True, needed_A_dict=required_A,
            bound_upper=True, bound_lower=True)

        A_info = A_dict[bounded_model.output_name[0]][
            bounded_model.input_name[0]]
        lA = A_info['lA'].detach()        # (N, out_dim, *input_dims)
        uA = A_info['uA'].detach()
        lbias = A_info['lbias'].detach()  # (N, out_dim)
        ubias = A_info['ubias'].detach()

        # ── Compute semantic-specific bounds ──────────────────────────
        lb, ub = _semantic_bounds_from_A(
            lA, uA, lbias, ubias, x_batch, epsilon, perturbation_type)

        lb = lb.detach()
        ub = ub.detach()

        # ── Classify results ──────────────────────────────────────────
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
                y_val = int(y_batch[i])
                lb_true = lb[i, y_val].item()
                ub_other = ub[i].clone()
                ub_other[y_val] = -float("inf")
                max_ub_other = ub_other.max().item()
                verified = lb_true > max_ub_other
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


class FlatSemanticWrapper(nn.Module):


    def __init__(self, base_model, n_pixels, image_shape, perturbation_type,
                 clamp_output=True):
        super().__init__()
        self.base_model = base_model
        self.n_pixels = n_pixels
        self.image_shape = image_shape      # (C, H, W) or None for MLP
        self.perturbation_type = perturbation_type
        self.clamp_output = clamp_output

    def forward(self, x_flat):
        pixels = x_flat[:, :self.n_pixels]          # (N, D)
        params = x_flat[:, self.n_pixels:]           # (N, 1) or (N, 2)

        if self.perturbation_type == "brightness":
            beta = params[:, 0:1]                    # (N, 1)
            perturbed = pixels + beta                # (N, D) — 2D broadcast

        elif self.perturbation_type == "contrast":
            alpha = params[:, 0:1]
            perturbed = (1.0 + alpha) * pixels

        elif self.perturbation_type == "brightness_contrast":
            beta = params[:, 0:1]
            alpha = params[:, 1:2]
            perturbed = (1.0 + alpha) * pixels + beta

        else:
            raise ValueError(self.perturbation_type)

        if self.clamp_output:
            # ReLU decomposition of clamp(0, 1):
            #   relu(x) − relu(x − 1)
            # Avoids HardTanh/Clip op that abcrown's _fast_backward_propagation
            # doesn't support.  auto_LiRPA handles BoundRelu natively.
            perturbed = torch.relu(perturbed) - torch.relu(perturbed - 1.0)

        # Reshape to image dims for base model (skip for MLPs)
        if self.image_shape is not None:
            perturbed = perturbed.view(-1, *self.image_shape)

        return self.base_model(perturbed)



class BakedPixelWrapper(nn.Module):


    def __init__(self, base_model, x_fixed, image_shape, perturbation_type):
        super().__init__()
        self.base_model = base_model
        self.register_buffer('x_fixed', x_fixed)   # (1, D) — baked pixels
        self.image_shape = image_shape
        self.perturbation_type = perturbation_type

    def forward(self, params):
        if self.perturbation_type == "contrast":
            alpha = params[:, 0:1]                        # (N, 1)
            perturbed = (1.0 + alpha) * self.x_fixed      # const * perturbed

        elif self.perturbation_type == "brightness_contrast":
            beta = params[:, 0:1]
            alpha = params[:, 1:2]
            perturbed = (1.0 + alpha) * self.x_fixed + beta

        elif self.perturbation_type == "brightness":
            beta = params[:, 0:1]
            perturbed = self.x_fixed + beta

        else:
            raise ValueError(self.perturbation_type)

        if self.image_shape is not None:
            perturbed = perturbed.view(-1, *self.image_shape)

        return self.base_model(perturbed)


def verify_batch_semantic_abcrown(model, indices, X_data, y_data, epsilon,
                                  input_shape, perturbation_type="brightness",
                                  device="cuda", abcrown_path=None,
                                  timeout=120, verbose=True, batch_size=32,
                                  **_ignored):
    
    if not abcrown_path:
        raise ValueError(
            "abcrown_path must point to alpha-beta-CROWN/complete_verifier/")

    if len(indices) == 0:
        return {}

    ordered = list(indices)
    n_params = 2 if perturbation_type == "brightness_contrast" else 1
    n_pixels = 1
    for d in input_shape[1:]:
        n_pixels *= d
    image_shape = tuple(input_shape[1:]) if len(input_shape) == 4 else None

    if batch_size is None or batch_size >= len(ordered):
        outer_batches = [ordered]
    else:
        outer_batches = [ordered[i:i + batch_size]
                         for i in range(0, len(ordered), batch_size)]

    if verbose:
        print(f"    Verifying {len(ordered)} examples "
              f"({perturbation_type}, ε={epsilon:.6f}) via α,β-CROWN "
              f"[BakedPixelWrapper, {n_params}D input, per-sample ONNX]")

    work_dir = tempfile.mkdtemp(prefix="abcrown_semantic_")
    all_statuses = {}

    try:
        yaml_path = _write_baked_yaml(work_dir, device, timeout)

        spec_dir = os.path.join(work_dir, "specs")
        onnx_dir = os.path.join(work_dir, "onnx")
        os.makedirs(spec_dir, exist_ok=True)
        os.makedirs(onnx_dir, exist_ok=True)

        for bi, batch in enumerate(outer_batches):
            if verbose and len(outer_batches) > 1:
                print(f"    ── Outer batch {bi+1}/{len(outer_batches)} ──")

            csv_lines = []
            for gidx in batch:
                if isinstance(X_data, torch.Tensor):
                    x_sample = X_data[gidx].cpu().float().reshape(1, -1)
                    y_val = int(y_data[gidx].item())
                else:
                    x_sample = torch.FloatTensor(
                        X_data[gidx]).reshape(1, -1)
                    y_val = int(y_data[gidx])

                onnx_path = _export_baked_onnx(
                    model, x_sample, image_shape, perturbation_type,
                    n_params, onnx_dir, gidx)

                spec_path = _write_params_only_vnnlib(
                    gidx, y_val, epsilon, n_params, spec_dir)

                csv_lines.append(f"{onnx_path},{spec_path},{timeout}")

            csv_path = os.path.join(work_dir, f"instances_{bi}.csv")
            with open(csv_path, "w") as f:
                f.write("\n".join(csv_lines) + "\n")

            results_path = os.path.join(work_dir, f"results_{bi}.txt")
            _run_abcrown_subprocess(
                abcrown_path, yaml_path, csv_path, results_path, verbose)

            batch_statuses = _parse_abcrown_results(
                results_path, batch, verbose)
            all_statuses.update(batch_statuses)

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        torch.cuda.empty_cache()

    results = _add_ranking_ce(
        all_statuses, model, X_data, y_data, input_shape, device)

    if verbose:
        nv = sum(1 for s, _ in results.values() if s == "verified")
        nu = sum(1 for s, _ in results.values() if s == "unsafe")
        nk = sum(1 for s, _ in results.values() if s == "unknown")
        print(f"        Done: {nv} verified, {nu} unsafe, {nk} unknown")

    return results



def _export_baked_onnx(model, x_sample, image_shape, perturbation_type,
                       n_params, onnx_dir, gidx):
    """Export BakedPixelWrapper for one sample (pixels as constants)."""
    onnx_path = os.path.join(onnx_dir, f"model_{gidx}.onnx")
    base_copy = copy.deepcopy(model).cpu().eval()

    wrapper = BakedPixelWrapper(
        base_copy, x_sample.cpu(), image_shape, perturbation_type)
    wrapper = wrapper.cpu().eval()

    dummy_params = torch.zeros(1, n_params)
    torch.onnx.export(
        wrapper, dummy_params, onnx_path,
        input_names=["input"], output_names=["output"],
        opset_version=13,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
    )
    return onnx_path


def _write_params_only_vnnlib(gidx, y_val, epsilon, n_params, spec_dir):

    lines = []
    for i in range(n_params):
        lines.append(f"(declare-const X_{i} Real)")
    lines.append("(declare-const Y_0 Real)")
    lines.append("")

    for i in range(n_params):
        lines.append(f"(assert (>= X_{i} {-epsilon:.12f}))")
        lines.append(f"(assert (<= X_{i} {epsilon:.12f}))")

    if y_val == 1:
        lines.append("\n(assert (<= Y_0 0.0))")
    else:
        lines.append("\n(assert (>= Y_0 0.0))")

    spec_path = os.path.join(spec_dir, f"prop_{gidx}.vnnlib")
    with open(spec_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return spec_path


def _write_baked_yaml(work_dir, device, timeout):
    """YAML for BakedPixelWrapper: full speed, KFSB fine (no exotic nodes)."""
    yaml_path = os.path.join(work_dir, "config.yaml")
    device_str = "cuda" if device.startswith("cuda") else "cpu"
    config = f"""\
general:
  device: {device_str}
  seed: 0
  complete_verifier: bab
  enable_incomplete_verification: true
  csv_name: null
  results_file: null

model:
  onnx_path: null

solver:
  batch_size: 64
  alpha-crown:
    iteration: 100
    lr_alpha: 0.1
  beta-crown:
    iteration: 50
    lr_alpha: 0.01
    lr_beta: 0.05

bab:
  timeout: {timeout}
  branching:
    method: kfsb

attack:
  pgd_order: skip
"""
    with open(yaml_path, "w") as f:
        f.write(config)
    return yaml_path


def _run_abcrown_subprocess(abcrown_path, yaml_path, csv_path,
                            results_path, verbose):
    abcrown_script = os.path.join(abcrown_path, "abcrown.py")
    if not os.path.isfile(abcrown_script):
        raise FileNotFoundError(f"Cannot find {abcrown_script}")

    cmd = [
        sys.executable, abcrown_script,
        "--config", yaml_path,
        "--csv_name", csv_path,
        "--results_file", results_path,
    ]

    env = os.environ.copy()
    extra_paths = [abcrown_path]
    auto_lirpa_dir = os.path.join(abcrown_path, "auto_LiRPA")
    if os.path.isdir(auto_lirpa_dir):
        extra_paths.append(auto_lirpa_dir)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        extra_paths + ([existing] if existing else []))

    if verbose:
        print(f"    Running α,β-CROWN: {' '.join(cmd[-6:])}")

    try:
        proc = subprocess.run(
            cmd, cwd=abcrown_path, env=env,
            capture_output=True, text=True, timeout=None)
        if verbose and proc.returncode != 0:
            for line in proc.stderr.strip().split("\n")[-10:]:
                print(f"      {line}")
        if verbose:
            for line in proc.stdout.strip().split("\n")[-5:]:
                print(f"      {line}")
    except Exception as e:
        if verbose:
            print(f"    α,β-CROWN failed: {e}")


def _parse_abcrown_results(results_path, ordered_indices, verbose):
    statuses = {}
    if not os.path.isfile(results_path):
        for gidx in ordered_indices:
            statuses[gidx] = "unknown"
        return statuses

    try:
        import pickle
        with open(results_path, "rb") as f:
            data = pickle.load(f)
    except Exception:
        for gidx in ordered_indices:
            statuses[gidx] = "unknown"
        return statuses

    per_instance = None
    if isinstance(data, dict) and "results" in data:
        per_instance = data["results"]
    elif isinstance(data, (list, tuple)):
        per_instance = data

    if per_instance is not None:
        for i, gidx in enumerate(ordered_indices):
            if i < len(per_instance):
                entry = per_instance[i]
                raw = str(entry[0]) if isinstance(entry, (list, tuple)) \
                    else str(entry)
                token = raw.strip().lower()
                if any(s in token for s in
                       ("safe", "holds", "unsat", "verified")):
                    statuses[gidx] = "verified"
                elif any(s in token for s in
                         ("unsafe", "violated", "sat", "counterexample")):
                    statuses[gidx] = "unsafe"
                else:
                    statuses[gidx] = "unknown"
            else:
                statuses[gidx] = "unknown"
    else:
        for gidx in ordered_indices:
            statuses[gidx] = "unknown"

    return statuses


def _add_ranking_ce(statuses, model, X_data, y_data, input_shape, device):

    results = {}
    unsat = [g for g, s in statuses.items() if s != "verified"]
    sat = [g for g, s in statuses.items() if s == "verified"]

    for gidx in unsat:
        results[gidx] = (statuses[gidx], float("inf"))

    if sat:
        model.eval()
        with torch.no_grad():
            for i in range(0, len(sat), 64):
                chunk = sat[i:i + 64]
                if isinstance(X_data, torch.Tensor):
                    x = X_data[chunk].float().to(device)
                    y = y_data[chunk].to(device)
                else:
                    x = torch.FloatTensor(X_data[chunk]).to(device)
                    y = torch.LongTensor(
                        [int(y_data[g]) for g in chunk]).to(device)
                if x.dim() == 2 and len(input_shape) == 4:
                    x = x.view(-1, *input_shape[1:])
                logits = model(x)
                if logits.shape[-1] == 1:
                    ce = F.binary_cross_entropy_with_logits(
                        logits.squeeze(-1), y.float(), reduction="none")
                else:
                    ce = F.cross_entropy(logits, y, reduction="none")
                for j, gidx in enumerate(chunk):
                    results[gidx] = ("verified", ce[j].item())

    return results