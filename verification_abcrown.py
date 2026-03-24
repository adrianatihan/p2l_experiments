"""
Verification via α,β-CROWN complete verifier.

Flow:
  1. Export PyTorch model → ONNX (temp dir)
  2. Write VNN-LIB specs per sample (L∞ input bounds + unsafe output cond.)
  3. Write instances CSV + YAML config
  4. Run α,β-CROWN via subprocess (includes PGD attack + BaB)
  5. Parse results → verified / unsafe / unknown
  6. Compute ranking bounds via quick auto_LiRPA backward for non-verified

Returns the same dict[int → (status, worst_case_bce)] as verification.py
so the P2L loop can rank non-verified examples identically.
"""

import os
import sys
import copy
import shutil
import tempfile
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

# ── Re-use auto_LiRPA ranking from the other backend ─────────────────────
from verification import _worst_case_bce_from_bounds


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════

def verify_batch_abcrown(model, indices, X_data, y_data, epsilon,
                         input_shape, device="cuda", abcrown_path=None,
                         timeout=120, verbose=True):
    """
    Verify examples using α,β-CROWN (complete: PGD + alpha-CROWN + BaB).

    Parameters
    ----------
    model        : nn.Module — single-logit binary classifier
    indices      : list[int] — which rows of X_data to verify
    X_data       : tensor or ndarray, full dataset
    y_data       : tensor or ndarray, full labels
    epsilon      : float — L∞ perturbation radius
    input_shape  : tuple — e.g. (1, 3, 32, 32) or (1, 784)
    device       : str
    abcrown_path : str — path to alpha-beta-CROWN/complete_verifier/
    timeout      : int — per-instance timeout in seconds
    verbose      : bool

    Returns
    -------
    dict[int → (str, float)]
        global_index → (status, worst_case_bce)
        status is 'verified' | 'unsafe' | 'unknown'
    """
    if not abcrown_path:
        raise ValueError(
            "abcrown_path must point to alpha-beta-CROWN/complete_verifier/"
        )

    if len(indices) == 0:
        return {}

    work_dir = tempfile.mkdtemp(prefix="abcrown_p2l_")

    try:
        # ── 1. Export model to ONNX ───────────────────────────────────
        onnx_path = _export_onnx(model, input_shape, work_dir, device)

        # ── 2. Write VNN-LIB specs ───────────────────────────────────
        spec_dir = os.path.join(work_dir, "specs")
        os.makedirs(spec_dir, exist_ok=True)

        idx_to_spec = {}
        for gidx in indices:
            spec_path = _write_vnnlib(gidx, X_data, y_data, epsilon,
                                      input_shape, spec_dir)
            idx_to_spec[gidx] = spec_path

        # ── 3. Write instances CSV ────────────────────────────────────
        csv_path = os.path.join(work_dir, "instances.csv")
        # Keep insertion order so we can map results back
        ordered_indices = list(indices)
        with open(csv_path, "w") as f:
            for gidx in ordered_indices:
                f.write(f"{onnx_path},{idx_to_spec[gidx]},{timeout}\n")

        # ── 4. Write YAML config ─────────────────────────────────────
        yaml_path = _write_yaml_config(work_dir, device, timeout)

        # ── 5. Run α,β-CROWN ─────────────────────────────────────────
        results_path = os.path.join(work_dir, "results.txt")
        success = _run_abcrown(abcrown_path, yaml_path, csv_path,
                               results_path, verbose)

        # ── 6. Parse results ──────────────────────────────────────────
        abcrown_statuses = _parse_results(results_path, ordered_indices,
                                          verbose)

        # ── 7. Add ranking bounds for non-verified ────────────────────
        results = _add_ranking_bounds(abcrown_statuses, model, X_data,
                                      y_data, epsilon, input_shape, device)

        if verbose:
            nv = sum(1 for s, _ in results.values() if s == "verified")
            nu = sum(1 for s, _ in results.values() if s == "unsafe")
            nk = sum(1 for s, _ in results.values() if s == "unknown")
            print(f"        Done: {nv} verified, "
                  f"{nu} unsafe, {nk} unknown")

        return results

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  ONNX export
# ═══════════════════════════════════════════════════════════════════════════════

def _export_onnx(model, input_shape, work_dir, device):
    """Export PyTorch model to ONNX in work_dir. Returns path."""
    onnx_path = os.path.join(work_dir, "model.onnx")
    model_cpu = copy.deepcopy(model).cpu().eval()
    dummy = torch.randn(*input_shape)

    torch.onnx.export(
        model_cpu, dummy, onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
    )
    return onnx_path


# ═══════════════════════════════════════════════════════════════════════════════
#  VNN-LIB spec generation
# ═══════════════════════════════════════════════════════════════════════════════

def _write_vnnlib(gidx, X_data, y_data, epsilon, input_shape, spec_dir):
    """
    Write a VNN-LIB spec for one sample.

    Input constraints: x_i ∈ [x_i − ε, x_i + ε]  (L∞ ball)
    Output property (unsafe condition):
        y=1 → unsafe if logit ≤ 0   →  (assert (<= Y_0 0.0))
        y=0 → unsafe if logit ≥ 0   →  (assert (>= Y_0 0.0))
    """
    if isinstance(X_data, torch.Tensor):
        x = X_data[gidx].cpu().numpy().flatten().astype(np.float64)
        y = int(y_data[gidx].item())
    else:
        x = np.asarray(X_data[gidx], dtype=np.float64).flatten()
        y = int(y_data[gidx])

    n_inputs = len(x)
    spec_path = os.path.join(spec_dir, f"prop_{gidx}.vnnlib")

    with open(spec_path, "w") as f:
        # Declare input variables
        for i in range(n_inputs):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("(declare-const Y_0 Real)\n\n")

        # Input bounds
        for i in range(n_inputs):
            lb = x[i] - epsilon
            ub = x[i] + epsilon
            f.write(f"(assert (>= X_{i} {lb:.12f}))\n")
            f.write(f"(assert (<= X_{i} {ub:.12f}))\n")

        f.write("\n")

        # Output property: unsafe condition
        if y == 1:
            f.write("; y=1: unsafe if logit <= 0\n")
            f.write("(assert (<= Y_0 0.0))\n")
        else:
            f.write("; y=0: unsafe if logit >= 0\n")
            f.write("(assert (>= Y_0 0.0))\n")

    return spec_path


# ═══════════════════════════════════════════════════════════════════════════════
#  YAML config
# ═══════════════════════════════════════════════════════════════════════════════

def _write_yaml_config(work_dir, device, timeout):
    """Write a minimal α,β-CROWN YAML config. Returns path."""
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
  batch_size: 2048
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
  pgd_order: before
  pgd_steps: 100
  pgd_restarts: 50
"""
    with open(yaml_path, "w") as f:
        f.write(config)
    return yaml_path


# ═══════════════════════════════════════════════════════════════════════════════
#  Run α,β-CROWN subprocess
# ═══════════════════════════════════════════════════════════════════════════════

def _run_abcrown(abcrown_path, yaml_path, csv_path, results_path, verbose):
    """
    Run α,β-CROWN via subprocess.
    The complete_verifier reads the CSV of (model, spec, timeout) triples,
    runs PGD + incomplete + BaB on each, and writes results.
    """
    abcrown_script = os.path.join(abcrown_path, "abcrown.py")
    if not os.path.isfile(abcrown_script):
        raise FileNotFoundError(
            f"Cannot find {abcrown_script}. "
            f"Set ABCROWN_PATH to the complete_verifier/ directory."
        )

    cmd = [
        sys.executable, abcrown_script,
        "--config", yaml_path,
        "--csv_name", csv_path,
        "--results_file", results_path,
    ]

    # auto_LiRPA lives inside the complete_verifier dir — the subprocess
    # needs PYTHONPATH set so `from auto_LiRPA import ...` works
    env = os.environ.copy()
    extra_paths = [abcrown_path]
    auto_lirpa_dir = os.path.join(abcrown_path, "auto_LiRPA")
    if os.path.isdir(auto_lirpa_dir):
        extra_paths.append(auto_lirpa_dir)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(extra_paths + ([existing] if existing else []))

    if verbose:
        print(f"    Running α,β-CROWN: {' '.join(cmd[-6:])}")

    try:
        proc = subprocess.run(
            cmd,
            cwd=abcrown_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=None,  # overall timeout handled per-instance by abcrown
        )
        if verbose and proc.returncode != 0:
            # Print last 20 lines of stderr for debugging
            err_lines = proc.stderr.strip().split("\n")[-20:]
            print("    α,β-CROWN stderr (last 20 lines):")
            for line in err_lines:
                print(f"      {line}")
        if verbose:
            # Print last 10 lines of stdout to see what abcrown did
            out_lines = proc.stdout.strip().split("\n")[-10:]
            print("    α,β-CROWN stdout (last 10 lines):")
            for line in out_lines:
                print(f"      {line}")
        return proc.returncode == 0

    except Exception as e:
        if verbose:
            print(f"    α,β-CROWN failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  Parse results
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_results(results_path, ordered_indices, verbose):
    """
    Parse α,β-CROWN results file → dict[int → str].

    The pickle format is:
        {'results': [['safe-incomplete', time], ['unsafe-pgd', time], ...],
         'summary': defaultdict(list, {'safe-incomplete': [...], ...}),
         'bab_ret': [...]}

    'results' has one [status, time] pair per instance in CSV order.
    """
    statuses = {}

    if not os.path.isfile(results_path):
        if verbose:
            print("    Warning: results file not found, marking all unknown")
        for gidx in ordered_indices:
            statuses[gidx] = "unknown"
        return statuses

    try:
        import pickle
        with open(results_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        if verbose:
            print(f"    Failed to load results: {e}, marking all unknown")
        for gidx in ordered_indices:
            statuses[gidx] = "unknown"
        return statuses

    # ── Extract per-instance results ──────────────────────────────────
    per_instance = None

    if isinstance(data, dict) and "results" in data:
        # Standard abcrown format: data['results'] = [[status, time], ...]
        per_instance = data["results"]
    elif isinstance(data, dict) and "summary" in data:
        # Fallback: reconstruct from summary dict
        summary = data["summary"]
        n = len(ordered_indices)
        per_instance = [["unknown", 0.0]] * n
        for status_str, idx_list in summary.items():
            for idx in idx_list:
                if idx < n:
                    per_instance[idx] = [status_str, 0.0]
    elif isinstance(data, (list, tuple)):
        per_instance = data

    if per_instance is not None:
        for i, gidx in enumerate(ordered_indices):
            if i < len(per_instance):
                entry = per_instance[i]
                if isinstance(entry, (list, tuple)) and len(entry) >= 1:
                    statuses[gidx] = _normalise_status(str(entry[0]))
                else:
                    statuses[gidx] = _normalise_status(str(entry))
            else:
                statuses[gidx] = "unknown"
    else:
        if verbose:
            print(f"    Unexpected pickle structure: keys={list(data.keys()) if isinstance(data, dict) else type(data)}")
        for gidx in ordered_indices:
            statuses[gidx] = "unknown"

    if verbose:
        nv = sum(1 for s in statuses.values() if s == "verified")
        nu = sum(1 for s in statuses.values() if s == "unsafe")
        nk = sum(1 for s in statuses.values() if s == "unknown")
        print(f"    α,β-CROWN results: {nv} verified, "
              f"{nu} unsafe, {nk} unknown")

    return statuses


def _normalise_status(raw):
    """Map α,β-CROWN status strings to our convention."""
    token = raw.strip().lower()
    if any(s in token for s in ("safe", "holds", "unsat", "verified")):
        return "verified"
    elif any(s in token for s in ("unsafe", "violated", "sat", "counterexample")):
        return "unsafe"
    else:
        return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
#  Ranking bounds via quick auto_LiRPA backward
# ═══════════════════════════════════════════════════════════════════════════════

def _add_ranking_bounds(abcrown_statuses, model, X_data, y_data,
                        epsilon, input_shape, device):
    """
    For non-verified examples, compute worst-case BCE from auto_LiRPA
    backward bounds (fast, incomplete — just for ranking, not for status).

    Verified examples get wc_bce = 0.0 (doesn't matter, they won't be
    selected). Non-verified examples that fail the backward pass get inf.
    """
    from verification import _ensure_autolirpa, _verify_chunk

    results = {}

    non_verified_indices = [
        gidx for gidx, status in abcrown_statuses.items()
        if status != "verified"
    ]

    # For verified: status from abcrown, wc_bce = 0 (won't be selected)
    for gidx, status in abcrown_statuses.items():
        if status == "verified":
            results[gidx] = ("verified", 0.0)

    if not non_verified_indices:
        return results

    # Quick auto_LiRPA backward on non-verified for ranking bounds
    try:
        _ensure_autolirpa()
        chunk_results = _verify_chunk(
            model, non_verified_indices, X_data, y_data,
            epsilon, input_shape, device,
        )
        if chunk_results is not None:
            for gidx in non_verified_indices:
                abcrown_status = abcrown_statuses[gidx]
                # Use abcrown's status (authoritative), LiRPA's wc_bce (ranking)
                _, wc_bce = chunk_results.get(gidx, (None, float("inf")))
                results[gidx] = (abcrown_status, wc_bce)
        else:
            # OOM — fall back to inf
            for gidx in non_verified_indices:
                results[gidx] = (abcrown_statuses[gidx], float("inf"))

    except Exception:
        # auto_LiRPA not available or failed — use inf for all non-verified
        for gidx in non_verified_indices:
            results[gidx] = (abcrown_statuses[gidx], float("inf"))

    return results