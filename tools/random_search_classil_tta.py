#!/usr/bin/env python3
"""Two-phase random search for MergeSlide-TTA Class-IL TCP.

Phase 1 optimizes task routing accuracy. Phase 2 fixes the best routing
hyperparameters and optimizes class-level bACC. Every trial runs the full
configured folds from the YAML config.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


SEARCH_SPACE_ROUTING: dict[str, list[Any]] = {
    "alpha": [0.25, 0.5, 0.75, 1.0],
    "gamma": [0.1, 0.3, 0.5, 0.8, 1.0],
    "delta_margin": [0.05, 0.10, 0.15, 0.20],
    "tp_anchor_beta": [0.1, 0.3, 0.5, 0.7],
    "ema_alpha_prompt": [0.99, 0.995, 0.999],
    "select_mode": ["intersection"],
}

SEARCH_SPACE_CLASS: dict[str, list[Any]] = {
    "lr": [2e-5, 5e-5, 1e-4, 2e-4],
    "beta": [0.3, 0.5, 1.0, 2.0],
    "top_ratio": [0.3, 0.5, 0.7],
    "entropy_threshold": [0.3, 0.4, 0.5],
    "gamma_margin": [0.0, 0.05, 0.1],
}

BASE_PARAMS: dict[str, Any] = {
    "M": 8,
    "K_sub": 300,
    "top_ratio": 0.5,
    "alpha": 0.5,
    "beta": 1.0,
    "lr": 1e-4,
    "n_steps": 5,
    "tta_param_scope": "ln_only",
    "entropy_threshold": 0.4,
    "gamma": 0.5,
    "select_mode": "intersection",
    "ema_alpha": 0.999,
    "ema_alpha_prompt": 0.999,
    "delta_margin": 0.10,
    "tp_anchor_beta": 0.3,
    "gamma_margin": 0.0,
}

BOOL_FLAGS = {
    "episodic": "--episodic",
    "use_task_diversity": "--use_task_diversity",
    "no_task_agreement": "--no_task_agreement",
    "no_teacher": "--no_teacher",
    "no_adapt_prompts": "--no_adapt_prompts",
    "no_reset_prompt_per_task": "--no_reset_prompt_per_task",
    "verbose_loss": "--verbose_loss",
}


def search_space_for(phase: str) -> dict[str, list[Any]]:
    return SEARCH_SPACE_ROUTING if phase == "routing" else SEARCH_SPACE_CLASS


def sample_config(rng: random.Random, phase: str) -> dict[str, Any]:
    return {k: rng.choice(v) for k, v in search_space_for(phase).items()}


def build_manifest(n_trials: int, seed: int, phase: str) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    return [
        {"trial_id": trial_id, "params": sample_config(rng, phase)}
        for trial_id in range(n_trials)
    ]


def save_manifest(manifest: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    keys = list(search_space_for(manifest_phase_from_path(path)).keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial_id", *keys])
        writer.writeheader()
        for item in manifest:
            writer.writerow({"trial_id": item["trial_id"], **item["params"]})


def manifest_phase_from_path(path: Path) -> str:
    name = path.name.lower()
    return "class" if "class" in name else "routing"


def load_manifest(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError(f"Invalid manifest format: {path}")
    return manifest


def load_fixed_params(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if "params" in raw and isinstance(raw["params"], dict):
        return dict(raw["params"])
    return {
        key: value
        for key, value in raw.items()
        if key in BASE_PARAMS or key in SEARCH_SPACE_ROUTING or key in SEARCH_SPACE_CLASS
    }


def parse_percent_line(stdout: str, marker: str) -> float | None:
    for line in stdout.splitlines():
        if marker not in line:
            continue
        try:
            return float(line.split(marker, 1)[1].strip().split("%", 1)[0])
        except (IndexError, ValueError):
            continue
    return None


def parse_bacc(stdout: str) -> float | None:
    return parse_percent_line(stdout, "Balanced Acc:")


def parse_acc(stdout: str) -> float | None:
    return parse_percent_line(stdout, "Accuracy:")


def parse_routing_acc(stdout: str) -> float | None:
    return parse_percent_line(stdout, "Routing Accuracy (mean):")


def parse_task_values(stdout: str, prefix: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for line in stdout.splitlines():
        s = line.strip()
        if not s.startswith(prefix):
            continue
        try:
            left, right = s.split(":", 1)
            task_id = int(left.replace(prefix, "").strip())
            values[task_id] = float(right.strip().split("%", 1)[0])
        except (IndexError, ValueError):
            continue
    return values


def run_command_streaming(
    cmd: list[str],
    cwd: str,
    stdout_path: Path,
    stderr_path: Path,
    header_lines: list[str],
    timeout_sec: int | None,
) -> tuple[str, str, int]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    with stdout_path.open("w", buffering=1) as stdout_file, stderr_path.open("w", buffering=1) as stderr_file:
        for line in header_lines:
            stdout_file.write(line.rstrip("\n") + "\n")
            stderr_file.write(line.rstrip("\n") + "\n")

        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        def pump(stream, file_handle, sink: list[str], is_err: bool) -> None:
            for line in iter(stream.readline, ""):
                sink.append(line)
                file_handle.write(line)
                file_handle.flush()
                if is_err:
                    sys.stderr.write(line)
                    sys.stderr.flush()
                else:
                    sys.stdout.write(line)
                    sys.stdout.flush()
            stream.close()

        stdout_thread = threading.Thread(target=pump, args=(proc.stdout, stdout_file, stdout_chunks, False), daemon=True)
        stderr_thread = threading.Thread(target=pump, args=(proc.stderr, stderr_file, stderr_chunks, True), daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        try:
            returncode = proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
            returncode = proc.wait()
        stdout_thread.join()
        stderr_thread.join()

    return "".join(stdout_chunks), "".join(stderr_chunks), returncode


def coerce_scalar(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return coerce_scalar(obj)


def has_cached_result(result_csv: Path) -> bool:
    if not result_csv.exists() or result_csv.stat().st_size == 0:
        return False
    try:
        with result_csv.open("r", newline="") as f:
            return any(True for _ in csv.DictReader(f))
    except Exception:
        return False


def parse_stdout_summary(stdout: str) -> tuple[float | None, float | None, float | None, dict[int, float], dict[int, float]]:
    bacc = parse_bacc(stdout)
    acc = parse_acc(stdout)
    routing_acc = parse_routing_acc(stdout)
    task_baccs = parse_task_values(stdout, "Task ")
    task_routing = parse_task_values(stdout, "Routing Task ")
    return bacc, acc, routing_acc, task_baccs, task_routing


def build_trial_row(
    trial_id: int,
    setting: str,
    phase: str,
    params: dict[str, Any],
    stdout: str,
    elapsed: float | None,
    returncode: int,
    trial_dir: Path,
    result_csv: Path,
    stderr_tail: str = "",
) -> dict[str, Any]:
    bacc, acc, routing_acc, task_baccs, task_routing = parse_stdout_summary(stdout)
    metric = routing_acc if phase == "routing" else bacc
    status = "ok" if returncode == 0 and metric is not None else "failed"
    row = {
        "trial_id": trial_id,
        "setting": setting,
        "phase": phase,
        "status": status,
        "objective": "routing_acc" if phase == "routing" else "bacc_mean",
        "objective_value": metric if metric is not None else float("nan"),
        "bacc_mean": bacc if bacc is not None else float("nan"),
        "acc_mean": acc if acc is not None else float("nan"),
        "routing_acc": routing_acc if routing_acc is not None else float("nan"),
        "elapsed_s": elapsed if elapsed is not None else float("nan"),
        "returncode": returncode,
        "stdout_log": str(trial_dir / "stdout.log"),
        "stderr_log": str(trial_dir / "stderr.log"),
        "result_csv": str(result_csv),
        "stderr_tail": stderr_tail,
    }
    for task_id in range(6):
        row[f"task_{task_id}_bacc"] = task_baccs.get(task_id, float("nan"))
        row[f"task_{task_id}_routing_acc"] = task_routing.get(task_id, float("nan"))
    row.update(params)
    return row


def append_csv_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(json_safe(row))


def params_to_cli(params: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in params.items():
        if key in BOOL_FLAGS:
            if bool(value):
                args.append(BOOL_FLAGS[key])
            continue
        args.extend([f"--{key}", str(value)])
    return args


def run_one_trial(args: argparse.Namespace, trial_id: int, trial_params: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    trial_dir = output_dir / f"trial_{trial_id:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    result_csv = trial_dir / "results.csv"
    efficiency_json = trial_dir / "efficiency.json"
    params = dict(BASE_PARAMS)
    params.update(args.fixed_params)
    params.update(trial_params)
    params["verbose_loss"] = bool(args.verbose_loss)
    params["use_task_diversity"] = bool(args.use_task_diversity)
    params["no_task_agreement"] = bool(args.no_task_agreement)
    params["no_teacher"] = bool(args.no_teacher)
    params["no_adapt_prompts"] = bool(args.no_adapt_prompts)
    params["no_reset_prompt_per_task"] = bool(args.no_reset_prompt_per_task)

    with (trial_dir / "params.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe({"trial_id": trial_id, "setting": args.setting, "phase": args.phase, "params": params}), f, indent=2)

    if has_cached_result(result_csv):
        stdout_path = trial_dir / "stdout.log"
        stdout = stdout_path.read_text(errors="replace") if stdout_path.exists() else ""
        print(f"[Trial {trial_id:04d}] SKIP cached result -> {result_csv}")
        return build_trial_row(
            trial_id=trial_id,
            setting=args.setting,
            phase=args.phase,
            params=params,
            stdout=stdout,
            elapsed=None,
            returncode=0,
            trial_dir=trial_dir,
            result_csv=result_csv,
        )

    cmd = [
        args.python_bin,
        "-u",
        args.entrypoint_wrapper,
        "--entrypoint",
        "test_classIL_tta.py",
        "--config",
        args.base_config,
        "--save_dir",
        args.finetuned_dir,
        "--merge_model_path",
        args.merge_dir,
        "--mode",
        "tcp",
        "--result_csv",
        str(result_csv),
        "--efficiency_json",
        str(efficiency_json),
        *params_to_cli(params),
    ]

    print(f"\n{'=' * 72}")
    print(f"[Trial {trial_id:04d}] setting={args.setting} phase={args.phase}")
    for key in sorted(params):
        print(f"    {key:24s} = {params[key]}")
    print(f"    result_csv              = {result_csv}")
    print(f"{'=' * 72}")

    started = time.time()
    try:
        stdout, stderr, returncode = run_command_streaming(
            cmd=cmd,
            cwd=args.project_root,
            stdout_path=trial_dir / "stdout.log",
            stderr_path=trial_dir / "stderr.log",
            header_lines=[
                f"[INFO] start at {datetime.now()}",
                f"[INFO] command={' '.join(cmd)}",
                f"[INFO] trial_id={trial_id} setting={args.setting} phase={args.phase}",
                f"[INFO] params={json.dumps(json_safe(params), sort_keys=True)}",
            ],
            timeout_sec=args.timeout_sec,
        )
    except Exception as exc:
        stdout = ""
        stderr = f"STREAMING_ERROR: {exc}"
        returncode = -1
    elapsed = time.time() - started

    row = build_trial_row(
        trial_id=trial_id,
        setting=args.setting,
        phase=args.phase,
        params=params,
        stdout=stdout,
        elapsed=elapsed,
        returncode=returncode,
        trial_dir=trial_dir,
        result_csv=result_csv,
        stderr_tail=stderr[-600:],
    )
    if row["status"] == "ok":
        print(
            f"[Trial {trial_id:04d}] OK objective={row['objective_value']:.4f}% "
            f"routing={row['routing_acc']:.4f}% bACC={row['bacc_mean']:.4f}% "
            f"elapsed={elapsed / 60:.1f} min"
        )
    else:
        print(f"[Trial {trial_id:04d}] FAILED rc={returncode} stderr_tail={stderr[-300:]}")
    return row


def metric_value(row: dict[str, Any], phase: str) -> float:
    key = "routing_acc" if phase == "routing" else "bacc_mean"
    try:
        value = float(row.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")
    return value


def sorted_valid(rows: list[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
    valid = [row for row in rows if not math.isnan(metric_value(row, phase))]
    return sorted(valid, key=lambda row: metric_value(row, phase), reverse=True)


def print_top(rows: list[dict[str, Any]], phase: str, n: int) -> None:
    ranked = sorted_valid(rows, phase)[:n]
    metric = "routing_acc" if phase == "routing" else "bACC"
    print(f"\n{'-' * 72}")
    print(f"TOP {len(ranked)} TRIALS (phase={phase}, objective={metric})")
    if not ranked:
        print("  No successful trial yet.")
    for idx, row in enumerate(ranked, start=1):
        print(
            f"  #{idx} {metric}={metric_value(row, phase):.4f}% "
            f"routing={float(row.get('routing_acc', float('nan'))):.4f}% "
            f"bACC={float(row.get('bacc_mean', float('nan'))):.4f}% "
            f"trial={int(float(row['trial_id'])):04d}"
        )
        keys = sorted(k for k in row if k in BASE_PARAMS or k in SEARCH_SPACE_ROUTING or k in SEARCH_SPACE_CLASS)
        for key in keys:
            print(f"       {key:24s} = {row.get(key)}")
    print(f"{'-' * 72}\n")


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(json_safe(row))


def write_top_artifacts(rows: list[dict[str, Any]], output_dir: Path, phase: str, top_k: int) -> None:
    ranked = sorted_valid(rows, phase)
    if not ranked:
        raise ValueError("No successful trials to summarize.")
    best = dict(ranked[0])
    param_keys = sorted(k for k in best if k in BASE_PARAMS or k in SEARCH_SPACE_ROUTING or k in SEARCH_SPACE_CLASS)
    best["params"] = {key: best[key] for key in param_keys}
    (output_dir / "best_trial.json").write_text(json.dumps(json_safe(best), indent=2), encoding="utf-8")
    (output_dir / "best_config.json").write_text(json.dumps(json_safe(best["params"]), indent=2), encoding="utf-8")
    top_rows = ranked[:top_k]
    write_summary(top_rows, output_dir / "top_k.csv")
    (output_dir / "top_configs.json").write_text(json.dumps(json_safe(top_rows), indent=2), encoding="utf-8")
    print(f"[TTA Random Search] best trial -> {output_dir / 'best_trial.json'}")
    print(f"[TTA Random Search] best config -> {output_dir / 'best_config.json'}")
    print(f"[TTA Random Search] top-{len(top_rows)} csv -> {output_dir / 'top_k.csv'}")


def collect_worker_rows(output_dir: Path) -> list[dict[str, Any]]:
    summary_files = sorted(output_dir.parent.glob(f"*/{output_dir.name}/summary_*.csv"))
    summary_files.extend(sorted(output_dir.glob("summary_*.csv")))
    rows_by_trial: dict[int, dict[str, Any]] = {}
    for path in summary_files:
        if path.name == "summary_all.csv":
            continue
        with path.open("r", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    trial_id = int(float(row["trial_id"]))
                except (KeyError, TypeError, ValueError):
                    continue
                rows_by_trial[trial_id] = row
    return [rows_by_trial[key] for key in sorted(rows_by_trial)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random search Class-IL MergeSlide-TTA hyperparameters.")
    parser.add_argument("--phase", choices=["routing", "class"], default="routing")
    parser.add_argument("--setting", choices=["ind", "ood"], default="ind")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--merge_dir", required=True)
    parser.add_argument("--finetuned_dir", required=True)
    parser.add_argument("--output_dir", default="./logs/random_search_classil_tta")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--entrypoint_wrapper", default="tools/run_classil_with_pt_features.py")
    parser.add_argument("--fixed_params_json", default="")
    parser.add_argument("--manifest_path", default="")
    parser.add_argument("--prepare_manifest", action="store_true")
    parser.add_argument("--summarize_only", action="store_true")
    parser.add_argument("--trial_start", type=int, default=0)
    parser.add_argument("--trial_end", type=int, default=None)
    parser.add_argument("--timeout_sec", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--verbose_loss", action="store_true", default=True)
    parser.add_argument("--quiet_loss", action="store_false", dest="verbose_loss")
    parser.add_argument("--use_task_diversity", action="store_true")
    parser.add_argument("--no_task_agreement", action="store_true")
    parser.add_argument("--no_teacher", action="store_true")
    parser.add_argument("--no_adapt_prompts", action="store_true")
    parser.add_argument("--no_reset_prompt_per_task", action="store_true")
    args = parser.parse_args()
    args.timeout_sec = args.timeout_sec or None
    args.fixed_params = load_fixed_params(args.fixed_params_json)
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.setting
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_path) if args.manifest_path else output_dir / f"manifest_{args.setting}_{args.phase}_{args.n_trials}_{args.seed}.json"

    print(f"[TTA Random Search] phase={args.phase} setting={args.setting} n_trials={args.n_trials}")
    print(f"[TTA Random Search] base_config={args.base_config}")
    print(f"[TTA Random Search] output_dir={output_dir}")
    print(f"[TTA Random Search] manifest={manifest_path}")
    print(f"[TTA Random Search] fixed_params={json.dumps(json_safe(args.fixed_params), sort_keys=True)}")
    print("[TTA Random Search] Search space:")
    for key, values in search_space_for(args.phase).items():
        print(f"  {key:24s}: {values}")

    if args.prepare_manifest:
        manifest = build_manifest(args.n_trials, args.seed, args.phase)
        save_manifest(manifest, manifest_path)
        print(f"[TTA Random Search] manifest written -> {manifest_path}")
        print(f"[TTA Random Search] manifest csv -> {manifest_path.with_suffix('.csv')}")
        return

    if args.summarize_only:
        rows = collect_worker_rows(output_dir)
        if not rows:
            raise FileNotFoundError(f"No worker summaries found for {output_dir}")
        write_summary(rows, output_dir / "summary_all.csv")
        write_top_artifacts(rows, output_dir, args.phase, args.top_k)
        print_top(rows, args.phase, args.top_k)
        return

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}; run with --prepare_manifest first.")

    manifest = load_manifest(manifest_path)
    trial_end = args.trial_end if args.trial_end is not None else args.n_trials
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = output_dir / f"summary_{timestamp}.csv"
    rows: list[dict[str, Any]] = []
    fieldnames: list[str] | None = None

    for item in manifest[args.trial_start:trial_end]:
        trial_id = int(item["trial_id"])
        row = run_one_trial(args, trial_id, dict(item["params"]), output_dir)
        rows.append(row)
        if fieldnames is None:
            fieldnames = list(row.keys())
        append_csv_row(summary_csv, row, fieldnames)
        print_top(rows, args.phase, min(args.top_k, 5))

    print(f"[TTA Random Search] worker summary -> {summary_csv}")
    print_top(rows, args.phase, args.top_k)


if __name__ == "__main__":
    main()
