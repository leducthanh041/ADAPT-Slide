#!/usr/bin/env python3
"""Compute ACC-based mACC/BWT/FGT from prefix model-merging runs.

Expected prefix layout:

  <prefix_root>/task_0/<method manifest>
  <prefix_root>/task_1/<method manifest>
  ...

Each prefix manifest must contain fold metrics with task-level keys:
`task0_acc`, `task1_acc`, ...
If `task0_bacc`, `task1_bacc`, ... are available, the exported matrix also
contains balanced accuracy for performance-drop plots.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prefix manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_manifest_paths(prefix_dir: Path, method: str, setting: str, eval_setting: str) -> list[Path]:
    method = method.lower()
    if method == "adamerging":
        return [prefix_dir / f"adamerging_{eval_setting}_summary.json"]
    if method == "adarank":
        return [
            prefix_dir / f"manifest_{setting}_{eval_setting}.json",
            prefix_dir / f"manifest_{setting}_all.json",
        ]
    if method in {"hivec", "hi-vec", "hi_vec"}:
        return [
            prefix_dir / f"manifest_{setting}_{eval_setting}.json",
            prefix_dir / f"manifest_{setting}_all.json",
        ]
    raise ValueError(f"Unsupported method: {method}")


def load_prefix_payload(prefix_root: Path, task_end: int, method: str, setting: str, eval_setting: str) -> dict[str, Any]:
    prefix_dir = prefix_root / f"task_{task_end}"
    candidates = candidate_manifest_paths(prefix_dir, method, setting, eval_setting)
    for path in candidates:
        if path.exists():
            return load_json(path)
    raise FileNotFoundError("Missing prefix manifest. Tried: " + ", ".join(str(p) for p in candidates))


def fold_rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("folds", [])
    out = []
    for row in rows:
        if "metrics" in row:
            metrics = dict(row["metrics"])
            metrics["fold"] = row.get("fold")
            out.append(metrics)
        else:
            out.append(dict(row))
    return out


def build_matrix(prefix_root: Path, method: str, setting: str, eval_setting: str, num_tasks: int):
    acc_matrix_by_fold: dict[int, list[list[float | None]]] = {}
    bacc_matrix_by_fold: dict[int, list[list[float | None]]] = {}
    seq_acc_by_fold: dict[int, list[float]] = {}
    seq_bacc_by_fold: dict[int, list[float]] = {}

    for task_end in range(num_tasks):
        payload = load_prefix_payload(prefix_root, task_end, method, setting, eval_setting)
        rows = fold_rows_from_payload(payload)
        for row in rows:
            fold = int(row["fold"])
            acc_matrix_by_fold.setdefault(fold, [[None for _ in range(num_tasks)] for _ in range(num_tasks)])
            bacc_matrix_by_fold.setdefault(fold, [[None for _ in range(num_tasks)] for _ in range(num_tasks)])
            seq_acc_by_fold.setdefault(fold, [0.0 for _ in range(num_tasks)])
            seq_bacc_by_fold.setdefault(fold, [0.0 for _ in range(num_tasks)])
            if "acc" in row:
                seq_acc_by_fold[fold][task_end] = float(row["acc"])
            if "bacc" in row:
                seq_bacc_by_fold[fold][task_end] = float(row["bacc"])
            for eval_task in range(task_end + 1):
                key = f"task{eval_task}_acc"
                if key not in row:
                    raise KeyError(
                        f"Missing {key} for method={method} setting={setting} "
                        f"prefix task_{task_end} fold={fold}. Re-run after task-level ACC patch."
                    )
                acc_matrix_by_fold[fold][task_end][eval_task] = float(row[key])
                bacc_key = f"task{eval_task}_bacc"
                if bacc_key not in row:
                    raise KeyError(
                        f"Missing {bacc_key} for method={method} setting={setting} "
                        f"prefix task_{task_end} fold={fold}. Re-run prefix inference after task-level bACC patch."
                    )
                bacc_matrix_by_fold[fold][task_end][eval_task] = float(row[bacc_key])
    return acc_matrix_by_fold, bacc_matrix_by_fold, seq_acc_by_fold, seq_bacc_by_fold


def compute_transfer_metrics(matrix: list[list[float | None]], seq_acc: list[float]) -> dict[str, float]:
    num_tasks = len(matrix)
    final_row = matrix[num_tasks - 1]
    bwt_vals = []
    fgt_vals = []
    for task_id in range(num_tasks - 1):
        learned = matrix[task_id][task_id]
        final = final_row[task_id]
        if learned is None or final is None:
            continue
        future_values = [matrix[t][task_id] for t in range(task_id, num_tasks) if matrix[t][task_id] is not None]
        bwt_vals.append(float(final) - float(learned))
        fgt_vals.append(max(float(v) for v in future_values) - float(final))
    return {
        "final_acc": float(seq_acc[num_tasks - 1]),
        "mean_acc": float(mean(seq_acc)),
        "macc": float(mean(seq_acc)),
        "bwt": float(mean(bwt_vals)) if bwt_vals else 0.0,
        "fgt": float(mean(fgt_vals)) if fgt_vals else 0.0,
    }


def pct(value: float) -> str:
    return f"{value * 100:.4f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute prefix continual metrics from task-level ACC manifests.")
    parser.add_argument("--method", required=True, choices=["adamerging", "adarank", "hivec"])
    parser.add_argument("--setting", required=True, choices=["ind", "ood", "ind_reverse"])
    parser.add_argument("--eval_setting", default="class_il")
    parser.add_argument("--prefix_root", required=True, type=Path)
    parser.add_argument("--num_tasks", type=int, default=6)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    acc_matrix_by_fold, bacc_matrix_by_fold, seq_acc_by_fold, seq_bacc_by_fold = build_matrix(
        prefix_root=args.prefix_root,
        method=args.method,
        setting=args.setting,
        eval_setting=args.eval_setting,
        num_tasks=args.num_tasks,
    )

    matrix_rows = []
    fold_metric_rows = []
    for fold in sorted(acc_matrix_by_fold):
        matrix = acc_matrix_by_fold[fold]
        bacc_matrix = bacc_matrix_by_fold[fold]
        seq_acc = seq_acc_by_fold[fold]
        metrics = compute_transfer_metrics(matrix, seq_acc)
        metrics["fold"] = fold
        fold_metric_rows.append(metrics)
        for seq_task in range(args.num_tasks):
            for eval_task in range(seq_task + 1):
                matrix_rows.append({
                    "fold": fold,
                    "seq_task": seq_task,
                    "eval_task": eval_task,
                    "acc": matrix[seq_task][eval_task],
                    "bacc": bacc_matrix[seq_task][eval_task],
                    "seq_acc": seq_acc[seq_task],
                    "seq_bacc": seq_bacc_by_fold[fold][seq_task],
                })

    matrix_csv = args.out_dir / f"{args.method}_{args.setting}_{args.eval_setting}_continual_matrix.csv"
    with matrix_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "seq_task", "eval_task", "acc", "bacc", "seq_acc", "seq_bacc"])
        writer.writeheader()
        writer.writerows(matrix_rows)

    fold_csv = args.out_dir / f"{args.method}_{args.setting}_{args.eval_setting}_continual_fold_metrics.csv"
    with fold_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "final_acc", "mean_acc", "macc", "bwt", "fgt"])
        writer.writeheader()
        writer.writerows(fold_metric_rows)

    keys = ["final_acc", "mean_acc", "macc", "bwt", "fgt"]
    summary = {
        "method": args.method,
        "setting": args.setting,
        "eval_setting": args.eval_setting,
        "prefix_root": str(args.prefix_root),
        "num_tasks": args.num_tasks,
        "folds": fold_metric_rows,
        "matrix_csv": str(matrix_csv),
        "fold_metrics_csv": str(fold_csv),
    }
    for key in keys:
        vals = [float(row[key]) for row in fold_metric_rows]
        summary[f"{key}_mean"] = float(mean(vals)) if vals else 0.0
        summary[f"{key}_std"] = float(pstdev(vals)) if len(vals) > 1 else 0.0

    summary_json = args.out_dir / f"{args.method}_{args.setting}_{args.eval_setting}_continual_metrics.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== Continual Prefix Metrics =====")
    print(f"Method: {args.method}")
    print(f"Setting: {args.setting}")
    print(f"Eval: {args.eval_setting}")
    for key in keys:
        print(f"{key}: {pct(summary[f'{key}_mean'])} ({pct(summary[f'{key}_std'])})")
    print(f"[INFO] matrix_csv={matrix_csv}")
    print(f"[INFO] fold_metrics_csv={fold_csv}")
    print(f"[INFO] summary_json={summary_json}")


if __name__ == "__main__":
    main()
