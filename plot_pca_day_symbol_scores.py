#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional


def _sub_tag(sub: str) -> str:
    sub = str(sub)
    return sub if sub.startswith("sub-") else f"sub-{sub}"


def _natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _load_condition_symbol(run_dir: Path) -> Optional[str]:
    assignment = _load_json(run_dir / "condition_assignment.json")
    symbol = assignment.get("symbol")
    if symbol:
        return str(symbol)

    metadata = _load_json(run_dir / "session_metadata.json")
    psychopy = metadata.get("psychopy")
    if isinstance(psychopy, dict):
        condition = psychopy.get("condition_assignment")
        if isinstance(condition, dict) and condition.get("symbol"):
            return str(condition["symbol"])
    return None


def _load_reg_ready_map(run_dir: Path) -> tuple[Optional[dict[int, bool]], Optional[int]]:
    status_path = run_dir / "regression_status_rt.csv"
    if not status_path.exists():
        return None, None

    ready: dict[int, bool] = {}
    with status_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "volume_idx" not in reader.fieldnames or "reg_ready" not in reader.fieldnames:
            return None, None
        for row in reader:
            try:
                ready[int(row["volume_idx"])] = bool(int(row["reg_ready"]))
            except (TypeError, ValueError):
                continue

    ready_vols = [vol for vol, is_ready in ready.items() if is_ready]
    first_ready = min(ready_vols) if ready_vols else None
    return ready, first_ready


def _iter_nf_run_dirs(day_dir: Path, include_runs: Optional[set[str]], exclude_runs: set[str]):
    func_dir = day_dir / "func"
    if not func_dir.exists():
        raise FileNotFoundError(f"Functional run folder not found: {func_dir}")

    for run_dir in sorted([p for p in func_dir.iterdir() if p.is_dir()], key=_natural_key):
        run_name = run_dir.name
        if include_runs is not None and run_name not in include_runs:
            continue
        if run_name in exclude_runs:
            continue
        if not (run_dir / "pca_realtime_scores.csv").exists():
            continue
        yield run_dir


def _collect_scores(
    day_dir: Path,
    *,
    score_column: str,
    include_runs: Optional[set[str]],
    exclude_runs: set[str],
    require_regression_status: bool,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    run_summaries: list[dict] = []

    for run_dir in _iter_nf_run_dirs(day_dir, include_runs, exclude_runs):
        scores_path = run_dir / "pca_realtime_scores.csv"
        fallback_symbol = _load_condition_symbol(run_dir)
        ready_map, first_ready = _load_reg_ready_map(run_dir)
        if ready_map is None and require_regression_status:
            run_summaries.append(
                {
                    "run": run_dir.name,
                    "included": 0,
                    "excluded_missing_regression_status": True,
                    "score_file": str(scores_path),
                }
            )
            continue

        included = 0
        excluded_not_ready = 0
        excluded_first_ready = 0
        excluded_bad_score = 0
        with scores_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or score_column not in reader.fieldnames:
                raise ValueError(f"{scores_path} does not contain score column {score_column!r}")
            for row in reader:
                try:
                    volume_idx = int(float(row.get("volume_idx", "")))
                except (TypeError, ValueError):
                    continue

                if ready_map is not None:
                    if not ready_map.get(volume_idx, False):
                        excluded_not_ready += 1
                        continue
                    if first_ready is not None and volume_idx <= first_ready:
                        excluded_first_ready += 1
                        continue

                try:
                    score = float(row.get(score_column, ""))
                except (TypeError, ValueError):
                    excluded_bad_score += 1
                    continue
                if not math.isfinite(score):
                    excluded_bad_score += 1
                    continue

                symbol = row.get("symbol") or fallback_symbol
                if not symbol:
                    symbol = run_dir.name

                rows.append(
                    {
                        "run": run_dir.name,
                        "volume_idx": volume_idx,
                        "symbol": str(symbol),
                        "condition_id": row.get("condition_id", ""),
                        "score_label": row.get("score_label", ""),
                        "score_column": score_column,
                        "score": score,
                    }
                )
                included += 1

        run_summaries.append(
            {
                "run": run_dir.name,
                "included": included,
                "excluded_not_ready": excluded_not_ready,
                "excluded_first_ready": excluded_first_ready,
                "excluded_bad_score": excluded_bad_score,
                "score_file": str(scores_path),
            }
        )

    return rows, run_summaries


def _write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run", "volume_idx", "symbol", "condition_id", "score_label", "score_column", "score"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _plot_boxplot(path: Path, rows: list[dict], *, score_column: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["symbol"])].append(float(row["score"]))

    symbols = sorted(grouped)
    data = [grouped[symbol] for symbol in symbols]
    labels = [f"{symbol}\nn={len(grouped[symbol])}" for symbol in symbols]

    fig_width = max(8, 1.4 * len(symbols))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    bp = ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True)
    for box in bp["boxes"]:
        box.set(facecolor="#d9d9d9", edgecolor="black")
    for key in ("whiskers", "caps", "medians", "means"):
        for artist in bp[key]:
            artist.set(color="black")
    ax.set_title(title)
    ax.set_xlabel("Condition symbol")
    ax.set_ylabel(score_column)
    ax.grid(axis="y", alpha=0.25)
    if score_column == "feedback_score":
        ax.set_ylim(0, 100)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot PCA neurofeedback scores from all NF runs in one subject/day, "
            "grouped by blinded condition symbol."
        )
    )
    parser.add_argument("--sub", "--subj", dest="sub", required=True, help="Subject ID, e.g. 99999")
    parser.add_argument("--day", required=True, help="Day/session folder, e.g. 3")
    parser.add_argument("--base-data", type=Path, default=Path("/SSD2/DecNef_py/data"))
    parser.add_argument(
        "--score-column",
        default="feedback_score",
        help="Column from pca_realtime_scores.csv to plot, e.g. feedback_score, score_z, raw_component_score.",
    )
    parser.add_argument(
        "--include-runs",
        default=None,
        help="Comma-separated run folders to include. Defaults to every run with pca_realtime_scores.csv.",
    )
    parser.add_argument(
        "--exclude-runs",
        default="",
        help="Comma-separated run folders to exclude.",
    )
    parser.add_argument(
        "--allow-missing-regression-status",
        action="store_true",
        help="Include runs even if regression_status_rt.csv is missing.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--csv-out", type=Path, default=None, help="Output CSV with included plotted scores.")
    args = parser.parse_args()

    subject_root = args.base_data / _sub_tag(args.sub)
    day_dir = subject_root / str(args.day)
    if not day_dir.exists():
        raise FileNotFoundError(f"Day folder not found: {day_dir}")

    include_runs = (
        {item.strip() for item in args.include_runs.split(",") if item.strip()}
        if args.include_runs
        else None
    )
    exclude_runs = {item.strip() for item in args.exclude_runs.split(",") if item.strip()}

    rows, summaries = _collect_scores(
        day_dir,
        score_column=args.score_column,
        include_runs=include_runs,
        exclude_runs=exclude_runs,
        require_regression_status=not args.allow_missing_regression_status,
    )
    if not rows:
        raise ValueError(
            "No PCA NF scores found after filtering. Check run folders, score column, "
            "and regression_status_rt.csv."
        )

    out_png = args.out or (day_dir / f"pca_day_symbol_{args.score_column}_boxplot.png")
    out_csv = args.csv_out or out_png.with_suffix(".csv")
    _write_rows_csv(out_csv, rows)
    _plot_boxplot(
        out_png,
        rows,
        score_column=args.score_column,
        title=f"{_sub_tag(args.sub)} day {args.day}: PCA NF scores by symbol",
    )

    print(f"Saved plot: {out_png}")
    print(f"Saved plotted values: {out_csv}")
    for summary in summaries:
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
