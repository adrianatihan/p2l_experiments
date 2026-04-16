"""
Verification via α,β-CROWN complete verifier.

Flow:
  1. Export PyTorch model → ONNX
  2. Write VNN-LIB specs per sample
  3. Run α,β-CROWN via subprocess (PGD + BaB)
  4. Parse results → verified / unsafe / unknown
  5. Compute ranking via worst-case CE from auto_LiRPA bounds

Smart ranking:
  - If any UNSAT (non-verified) examples exist → compute CE bounds only
    for those (an unsat will be picked, sat bounds don't matter)
  - If ALL are SAT (verified) → compute CE bounds for ALL remaining
    (needed to check ce_threshold stopping condition)
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════

def verify_batch_abcrown(model, indices, X_data, y_data, epsilon,
                         input_shape, device="cuda", abcrown_path=None,
                         timeout=120, verbose=True, chunk_size=None):
    """Verify examples using α,β-CROWN for multi-class classification."""
    if not abcrown_path:
        raise ValueError(
            "abcrown_path must point to alpha-beta-CROWN/complete_verifier/"
        )

    if len(indices) == 0:
        return {}

    # Infer num_classes via a forward pass
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, *input_shape[1:]).to(device)
        num_classes = int(model(dummy).shape[-1])

    if verbose:
        print(f"    Verifying {len(indices)} examples via α,β-CROWN "
              f"(multi-class, C={num_classes})")

    work_dir = tempfile.mkdtemp(prefix="abcrown_p2l_")
    all_statuses = {}

    try:
        # ── 1. Export ONNX once ───────────────────────────────────────
        onnx_path = _export_onnx(model, input_shape, work_dir, device)

        # ── 2. Bulk-write all VNN-LIB specs ───────────────────────────
        spec_dir = os.path.join(work_dir, "specs")
        os.makedirs(spec_dir, exist_ok=True)

        ordered = list(indices)
        spec_paths = _write_vnnlib_bulk(
            ordered, X_data, y_data, epsilon, spec_dir, num_classes)

        # ── 3. Single instances CSV ───────────────────────────────────
        csv_path = os.path.join(work_dir, "instances.csv")
        csv_lines = [f"{onnx_path},{spec_paths[gidx]},{timeout}"
                     for gidx in ordered]
        with open(csv_path, "w") as f:
            f.write("\n".join(csv_lines) + "\n")

        # ── 4. YAML config + run ──────────────────────────────────────
        yaml_path = _write_yaml_config(work_dir, device, timeout)
        results_path = os.path.join(work_dir, "results.txt")
        _run_abcrown(abcrown_path, yaml_path, csv_path,
                     results_path, verbose=verbose)

        # ── 5. Parse ──────────────────────────────────────────────────
        all_statuses = _parse_results(results_path, ordered,
                                      verbose=verbose)

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        torch.cuda.empty_cache()

    # ── 6. Compute ranking CE ─────────────────────────────────────────
    results = _add_ranking_bounds(all_statuses, model, X_data, y_data,
                                  epsilon, input_shape, device)

    if verbose:
        nv = sum(1 for s, _ in results.values() if s == "verified")
        nu = sum(1 for s, _ in results.values() if s == "unsafe")
        nk = sum(1 for s, _ in results.values() if s == "unknown")
        print(f"        Done: {nv} verified, {nu} unsafe, {nk} unknown")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  ONNX export
# ═══════════════════════════════════════════════════════════════════════════════

def _export_onnx(model, input_shape, work_dir, device):
    onnx_path = os.path.join(work_dir, "model.onnx")
    model_cpu = copy.deepcopy(model).cpu().eval()
    dummy = torch.randn(*input_shape)
    torch.onnx.export(
        model_cpu, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        opset_version=13,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
    )
    return onnx_path


# ═══════════════════════════════════════════════════════════════════════════════
#  VNN-LIB spec generation (multi-class)
# ═══════════════════════════════════════════════════════════════════════════════

def _write_vnnlib_bulk(indices, X_data, y_data, epsilon, spec_dir, num_classes):
    """
    Write VNN-LIB specs for multi-class robustness.

    For each example (x, y_true), the property asserts the UNSAFE condition:
        ∃ c ≠ y_true.  Y_c ≥ Y_{y_true}

    encoded as:
        (assert (or
          (and (>= Y_c0 Y_{y_true}))
          (and (>= Y_c1 Y_{y_true}))
          ...
        ))

    α,β-CROWN returns UNSAT (safe/verified) if the model is robust,
    SAT (unsafe) if there's a counterexample.
    """
    spec_paths = {}

    if isinstance(X_data, torch.Tensor):
        X_np = X_data[indices].cpu().numpy().reshape(len(indices), -1)
        y_np = y_data[indices].cpu().numpy()
    else:
        X_np = np.asarray(X_data[indices], dtype=np.float64).reshape(
            len(indices), -1)
        y_np = np.asarray([int(y_data[g]) for g in indices])

    n_inputs = X_np.shape[1]

    # Pre-build shared declarations (same for all instances)
    declares = "".join(f"(declare-const X_{i} Real)\n" for i in range(n_inputs))
    declares += "".join(f"(declare-const Y_{c} Real)\n" for c in range(num_classes))
    declares += "\n"

    for j, gidx in enumerate(indices):
        x = X_np[j]
        y = int(y_np[j])

        # Input bounds
        lbs = x - epsilon
        ubs = x + epsilon
        bounds_lines = []
        for i in range(n_inputs):
            bounds_lines.append(f"(assert (>= X_{i} {lbs[i]:.12f}))")
            bounds_lines.append(f"(assert (<= X_{i} {ubs[i]:.12f}))")
        bounds = "\n".join(bounds_lines)

        # Multi-class unsafe disjunction
        other_classes = [c for c in range(num_classes) if c != y]
        or_clauses = "\n".join(
            f"  (and (>= Y_{c} Y_{y}))" for c in other_classes
        )
        prop = (
            f"\n; y={y}: unsafe if any other class dominates the true class\n"
            f"(assert (or\n{or_clauses}\n))\n"
        )

        spec_path = os.path.join(spec_dir, f"prop_{gidx}.vnnlib")
        with open(spec_path, "w") as f:
            f.write(declares)
            f.write(bounds)
            f.write(prop)

        spec_paths[gidx] = spec_path

    return spec_paths


# ═══════════════════════════════════════════════════════════════════════════════
#  YAML config
# ═══════════════════════════════════════════════════════════════════════════════

def _write_yaml_config(work_dir, device, timeout):
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Run α,β-CROWN subprocess
# ═══════════════════════════════════════════════════════════════════════════════

def _run_abcrown(abcrown_path, yaml_path, csv_path, results_path, verbose):
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
            capture_output=True, text=True, timeout=None,
        )
        if verbose and proc.returncode != 0:
            for line in proc.stderr.strip().split("\n")[-20:]:
                print(f"      {line}")
        if verbose:
            for line in proc.stdout.strip().split("\n")[-10:]:
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

    per_instance = None
    if isinstance(data, dict) and "results" in data:
        per_instance = data["results"]
    elif isinstance(data, dict) and "summary" in data:
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
                raw = str(entry[0]) if isinstance(entry, (list, tuple)) else str(entry)
                statuses[gidx] = _normalise_status(raw)
            else:
                statuses[gidx] = "unknown"
    else:
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
    token = raw.strip().lower()
    if any(s in token for s in ("safe", "holds", "unsat", "verified")):
        return "verified"
    elif any(s in token for s in ("unsafe", "violated", "sat", "counterexample")):
        return "unsafe"
    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
#  Ranking bounds (multi-class)
# ═══════════════════════════════════════════════════════════════════════════════

def _add_ranking_bounds(abcrown_statuses, model, X_data, y_data,
                        epsilon, input_shape, device, ranking_chunk=64):
    """
    - Unsafe/unknown: wc_ce = inf (always picked first)
    - Verified: clean forward-pass CE (for ce_threshold check)
    """
    results = {}

    unsat_indices = [g for g, s in abcrown_statuses.items() if s != "verified"]
    sat_indices = [g for g, s in abcrown_statuses.items() if s == "verified"]

    for gidx in unsat_indices:
        results[gidx] = (abcrown_statuses[gidx], float("inf"))

    if not sat_indices:
        return results

    model.eval()
    with torch.no_grad():
        for i in range(0, len(sat_indices), ranking_chunk):
            chunk = sat_indices[i:i + ranking_chunk]

            if isinstance(X_data, torch.Tensor):
                x = X_data[chunk].float().to(device)
                y = y_data[chunk].long().to(device)
            else:
                x = torch.FloatTensor(X_data[chunk]).to(device)
                y = torch.LongTensor(
                    [int(y_data[g]) for g in chunk]).to(device)

            if x.dim() == 2 and len(input_shape) == 4:
                x = x.view(-1, *input_shape[1:])

            logits = model(x)
            ce = F.cross_entropy(logits, y, reduction="none")

            for j, gidx in enumerate(chunk):
                results[gidx] = ("verified", ce[j].item())

    return results