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
                         timeout=120, verbose=True, chunk_size=16):
    """
    Verify examples using α,β-CROWN (complete: PGD + alpha-CROWN + BaB).
    Processes in chunks. Returns dict[int → (status, worst_case_ce)].
    """
    if not abcrown_path:
        raise ValueError(
            "abcrown_path must point to alpha-beta-CROWN/complete_verifier/"
        )

    if len(indices) == 0:
        return {}

    chunks = [indices[i:i + chunk_size]
              for i in range(0, len(indices), chunk_size)]

    if verbose:
        print(f"    Verifying {len(indices)} examples in "
              f"{len(chunks)} chunks (≤{chunk_size}) via α,β-CROWN")

    all_statuses = {}

    for ci, chunk_indices in enumerate(chunks):
        work_dir = tempfile.mkdtemp(prefix="abcrown_p2l_")

        try:
            onnx_path = _export_onnx(model, input_shape, work_dir, device)

            spec_dir = os.path.join(work_dir, "specs")
            os.makedirs(spec_dir, exist_ok=True)

            idx_to_spec = {}
            for gidx in chunk_indices:
                idx_to_spec[gidx] = _write_vnnlib(
                    gidx, X_data, y_data, epsilon, input_shape, spec_dir)

            csv_path = os.path.join(work_dir, "instances.csv")
            ordered = list(chunk_indices)
            with open(csv_path, "w") as f:
                for gidx in ordered:
                    f.write(f"{onnx_path},{idx_to_spec[gidx]},{timeout}\n")

            yaml_path = _write_yaml_config(work_dir, device, timeout)

            results_path = os.path.join(work_dir, "results.txt")
            _run_abcrown(abcrown_path, yaml_path, csv_path,
                         results_path, verbose=(verbose and ci == 0))

            chunk_statuses = _parse_results(results_path, ordered,
                                            verbose=(verbose and ci == 0))
            all_statuses.update(chunk_statuses)

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            torch.cuda.empty_cache()

        if verbose:
            nv = sum(1 for s in all_statuses.values() if s == "verified")
            nu = sum(1 for s in all_statuses.values() if s == "unsafe")
            nk = sum(1 for s in all_statuses.values() if s == "unknown")
            print(f"        Chunk {ci+1}/{len(chunks)}: "
                  f"{nv} verified, {nu} unsafe, {nk} unknown so far")

    # ── Compute ranking CE bounds ─────────────────────────────────────
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
#  VNN-LIB spec generation
# ═══════════════════════════════════════════════════════════════════════════════

def _write_vnnlib(gidx, X_data, y_data, epsilon, input_shape, spec_dir):
    if isinstance(X_data, torch.Tensor):
        x = X_data[gidx].cpu().numpy().flatten().astype(np.float64)
        y = int(y_data[gidx].item())
    else:
        x = np.asarray(X_data[gidx], dtype=np.float64).flatten()
        y = int(y_data[gidx])

    n_inputs = len(x)
    spec_path = os.path.join(spec_dir, f"prop_{gidx}.vnnlib")

    with open(spec_path, "w") as f:
        for i in range(n_inputs):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("(declare-const Y_0 Real)\n\n")

        for i in range(n_inputs):
            lb = x[i] - epsilon
            ub = x[i] + epsilon
            f.write(f"(assert (>= X_{i} {lb:.12f}))\n")
            f.write(f"(assert (<= X_{i} {ub:.12f}))\n")
        f.write("\n")

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
#  Ranking bounds via auto_LiRPA backward
# ═══════════════════════════════════════════════════════════════════════════════

def _add_ranking_bounds(abcrown_statuses, model, X_data, y_data,
                        epsilon, input_shape, device, ranking_chunk=4):
    """
    Compute worst-case CE from auto_LiRPA bounds for ranking.

    Smart strategy:
      - If any UNSAT (unsafe/unknown) examples exist → compute CE only
        for those. An unsat example will always be picked over a sat one,
        so sat CE doesn't matter for selection.
      - If ALL are SAT (verified) → compute CE for ALL remaining.
        The P2L loop needs wc_ce for every example to check whether
        ce_threshold is satisfied (the stopping condition).

    abcrown's status (verified/unsafe/unknown) is preserved.
    Only the wc_ce ranking comes from auto_LiRPA bounds.
    """
    from verification import _ensure_autolirpa, _verify_chunk

    results = {}
    all_indices = list(abcrown_statuses.keys())

    unsat_indices = [g for g, s in abcrown_statuses.items()
                     if s != "verified"]
    sat_indices = [g for g, s in abcrown_statuses.items()
                   if s == "verified"]

    if unsat_indices:
        # Unsat exist → only compute bounds for unsat (they'll be picked)
        # Sat examples get wc_ce = 0 (won't be selected)
        for gidx in sat_indices:
            results[gidx] = ("verified", 0.0)
        indices_to_bound = unsat_indices
    else:
        # All verified → compute bounds for ALL (check ce_threshold)
        indices_to_bound = all_indices

    if not indices_to_bound:
        return results

    try:
        _ensure_autolirpa()
        chunks = [indices_to_bound[i:i + ranking_chunk]
                  for i in range(0, len(indices_to_bound), ranking_chunk)]

        for chunk in chunks:
            torch.cuda.empty_cache()
            chunk_results = _verify_chunk(
                model, chunk, X_data, y_data,
                epsilon, input_shape, device,
            )
            if chunk_results is not None:
                for gidx in chunk:
                    _, wc_ce = chunk_results.get(gidx, (None, float("inf")))
                    results[gidx] = (abcrown_statuses[gidx], wc_ce)
            else:
                # OOM — single-sample fallback
                for gidx in chunk:
                    torch.cuda.empty_cache()
                    single = _verify_chunk(
                        model, [gidx], X_data, y_data,
                        epsilon, input_shape, device,
                    )
                    if single is not None:
                        _, wc_ce = single.get(gidx, (None, float("inf")))
                        results[gidx] = (abcrown_statuses[gidx], wc_ce)
                    else:
                        results[gidx] = (abcrown_statuses[gidx], float("inf"))

    except Exception:
        for gidx in indices_to_bound:
            if gidx not in results:
                results[gidx] = (abcrown_statuses[gidx], float("inf"))

    return results