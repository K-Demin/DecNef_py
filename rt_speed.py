"""Opt-in wall-clock profiling; plot after acquisition, never in the TR loop."""
import argparse
import csv
import logging
import math
import threading
import time
from pathlib import Path

STAGES = {
    "DICOM": "DICOM — DICOM to NIfTI", "FMAP": "FMAP — unwarping (PyHySCO)",
    "MC": "MC — motion correction + FD", "QC": "DVARS — signal-change / censor checks",
    "REG": "REG — denoising", "TRANS": "TRANS — spatial resampling",
    "SMOOTH": "SMOOTH — masked smoothing", "MASK": "MASK — analysis mask",
    "SCORE": "SCORE — decoder + score output", "OUTPUT": "OUTPUT — status logs / preview",
    "OTHER": "OTHER — processing overhead",
}
FIELDS = [
    "volume", "source", "cached_input", "attempts", "reg_ready", "status",
    "arrival_interval_ms", "idle_wait_ms", "worker_queue_ms", "order_queue_ms", "file_write_ms",
    "processing_ms", "latency_ms", *[f"{s}_ms" for s in STAGES],
]


class SpeedRecorder:
    """Thread-safe, monotonic timings; completed rows are flushed immediately."""

    def __init__(self, path, clock=time.perf_counter):
        self.path, self.clock = Path(path), clock
        self.rows, self.lock = {}, threading.Lock()
        self.last_arrival = self.last_done = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.with_suffix(".png").unlink(missing_ok=True)  # Never leave a stale report for a new run.
        with self.path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, FIELDS).writeheader()

    def arrive(self, volume, backlog=False):
        with self.lock:
            if volume in self.rows:
                return
            now = self.clock()
            busy = any("done" not in r for r in self.rows.values())
            row = dict.fromkeys(FIELDS, "")
            row.update(volume=volume, source="backlog" if backlog else "live",
                       cached_input=False, attempts=0, detected=now, work=0.)
            row.update({f"{s}_ms": 0. for s in STAGES})
            if not backlog:
                if self.last_arrival is not None:
                    row["arrival_interval_ms"] = 1000 * (now - self.last_arrival)
                if self.last_done is not None:
                    row["idle_wait_ms"] = 0. if busy else 1000 * (now - self.last_done)
                self.last_arrival = now
            self.rows[volume] = row

    def file_event(self, volume, event):
        """Created→closed callback interval, when both precede row completion.

        Some filesystems do not emit close notifications. Missing events and
        backlog files stay unknown. Event-delivery delay is included; this is
        not scanner-to-disk transfer time.
        """
        if event not in {"created", "closed"}:
            raise ValueError("Expected a created or closed file event.")
        self.arrive(volume)
        with self.lock:
            row = self.rows[volume]
            if row["source"] != "live" or "done" in row:
                return
            key = f"file_{event}"
            if key in row:
                return
            row[key] = self.clock()
            if event == "closed" and "file_created" in row:
                row["file_write_ms"] = 1000 * (row[key] - row["file_created"])

    def conversion_start(self, volume, cached=False):
        with self.lock:
            row, now = self.rows[volume], self.clock()
            if not row["attempts"]:
                row["worker_queue_ms"] = 1000 * (now - row["detected"])
            row["attempts"] += 1
            row["cached_input"] = row["cached_input"] or cached
            return now

    def conversion_end(self, volume, start):
        with self.lock:
            row, now = self.rows[volume], self.clock()
            elapsed = now - start
            row["DICOM_ms"] += 1000 * elapsed
            row["work"] += elapsed
            row["converted"] = now

    def process_start(self, volume):
        with self.lock:
            row, now = self.rows[volume], self.clock()
            if "process_started" not in row:
                row["order_queue_ms"] = 1000 * (now - row.get("converted", now))
                row["process_started"] = now
            return now

    def process_end(self, volume, start):
        with self.lock:
            self.rows[volume]["work"] += self.clock() - start

    def stage(self, volume, label, seconds):
        with self.lock:
            self.rows[volume][f"{label}_ms"] += 1000 * seconds

    def readiness(self, volume, ready):
        with self.lock:
            self.rows[volume]["reg_ready"] = bool(ready)

    def finish(self, volume, success):
        with self.lock:
            row = self.rows[volume]
            if "done" in row:
                return
            row["done"] = self.last_done = self.clock()
            row["status"] = "ok" if success else "failed"
            row["latency_ms"] = 1000 * (row["done"] - row["detected"])
            row["processing_ms"] = 1000 * row["work"]
            accounted = sum(row[f"{s}_ms"] for s in STAGES if s != "OTHER")
            row["OTHER_ms"] = max(0., row["processing_ms"] - accounted)
            with self.path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, FIELDS, extrasaction="ignore").writerow(row)


def median_ci(values, *, sample_indices=None, n_resamples=2000, seed=0):
    """Median and approximate 95% moving-block-bootstrap CI, in input units.

    Adjacent samples are resampled together (ceil(n**(1/3)) per block),
    retaining short-range timing dependence. This assumes approximate local
    stationarity; use the per-volume CSV to inspect drift across the run.
    https://arch.readthedocs.io/en/stable/bootstrap/timeseries-bootstraps.html
    """
    import numpy as np
    data = np.asarray(values, dtype=float)
    positions = np.arange(len(data)) if sample_indices is None else np.asarray(sample_indices)
    valid = np.isfinite(data)
    data, positions = data[valid], positions[valid]
    n = len(data)
    median = float(np.median(data)) if n else float("nan")
    if n < 8 or np.any(np.diff(positions) != 1):
        return median, float("nan"), float("nan")
    block = int(np.ceil(n ** (1 / 3)))
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples)
    for begin in range(0, n_resamples, 64):
        count = min(64, n_resamples - begin)
        starts = rng.integers(0, n - block + 1, (count, math.ceil(n / block)))
        indices = (starts[..., None] + np.arange(block)).reshape(count, -1)[:, :n]
        estimates[begin:begin + count] = np.median(data[indices], axis=1)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return median, float(low), float(high)


def select_profile_rows(rows):
    """Compare one acquisition/cache cohort, preferentially ready, fresh live data."""
    good = [r for r in rows if r["status"] == "ok"]
    ready = [r for r in good if r["reg_ready"] == "True"]
    candidates = ready or good
    for source, cached in [("live", False), ("backlog", False), ("live", True), ("backlog", True)]:
        chosen = [r for r in candidates if r["source"] == source
                  and (r["cached_input"] == "True") == cached]
        if chosen:
            label = f"{source}, {'reused NIfTI' if cached else 'fresh conversion'}, "
            return chosen, label + ("ready volumes" if ready else "warm-up only")
    return [], "no successful volumes"


def plot_speed(csv_path, tr=1.0, output=None):
    """Render measured data only. Returns None when no finished volumes exist."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None
    rows, skipped = [], 0
    required = ["processing_ms", "latency_ms", *[f"{s}_ms" for s in STAGES]]
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                valid = (int(row["volume"]) >= 0 and row["status"] in {"ok", "failed"}
                         and all(math.isfinite(float(row[k])) and float(row[k]) >= 0
                                 for k in required))
            except (ValueError, TypeError, KeyError):
                valid = False
            if valid:
                rows.append(row)
            else:
                skipped += 1
    if skipped:
        logging.warning("Speed plot skipped %d incomplete/invalid CSV rows.", skipped)
    rows.sort(key=lambda r: int(r["volume"]))
    if not rows:
        return None
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    measured, cohort = select_profile_rows(rows)
    def values(key):
        return np.array([float(r[key]) if r.get(key) not in (None, "") else np.nan
                         for r in measured])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(16, 9), facecolor="#f5f8fc")
    ax = fig.add_axes([.32, .24, .63, .58])
    ax.set_facecolor("#f5f8fc")
    ax.grid(axis="x", color="#dae2ec", linewidth=.7)
    ax.set_axisbelow(True)
    fig.text(.045, .94, "Per-stage duration", fontsize=25,
             weight="bold", color="#102b46")
    failed = sum(r["status"] != "ok" for r in rows)
    fig.text(.045, .89, f"Median with 95% CI  ·  {cohort}  ·  n = {len(measured)}  ·  TR {tr:g} s"
             f"  ·  {failed} failed excluded", color="#52667c")
    inputs = [
        ("idle_wait_ms", "WAIT — idle until next volume"),
        ("file_write_ms", "FILE — creation → close notifications"),
        ("worker_queue_ms", "WATCHDOG — detection → conversion start"),
        ("order_queue_ms", "ORDER — waiting for prior volumes"),
    ]
    metrics = inputs + [(f"{s}_ms", label) for s, label in STAGES.items()
                        if np.any(values(f"{s}_ms") > 0)]
    positions = [int(r["volume"]) for r in measured]
    summaries = [median_ci(values(key), sample_indices=positions) for key, _ in metrics]
    ends = [max(m, hi) if np.isfinite(hi) else m for m, _, hi in summaries]
    limit = max([e for e in ends if np.isfinite(e)] or [1.]) * 1.45 or 1.
    for y, ((key, _), (med, low, high), end) in enumerate(zip(metrics, summaries, ends)):
        count = int(np.isfinite(values(key)).sum())
        if not count:
            ax.text(limit*.012, y, "not observed", va="center", color="#7a8796")
            continue
        ax.barh(y, med, color="#ce935c" if y < len(inputs) else "#167d9b", height=.55)
        if np.isfinite(low):
            ax.hlines(y, low, high, color="#102b46", linewidth=2, zorder=4)
            ax.vlines([low, high], y-.13, y+.13, color="#102b46", linewidth=1.6, zorder=4)
        fmt = lambda value: f"{value:.1f}" if value < 10 else f"{value:.0f}"
        interval = f"[{fmt(low)}, {fmt(high)}]" if np.isfinite(low) else "CI unavailable"
        ax.text(end + limit*.012, y, f"{fmt(med)}  {interval}   n={count}", va="center", fontsize=10)
    ax.axhline(len(inputs)-.5, color="#c4cfdc", linewidth=1)
    ax.set_yticks(range(len(metrics)), [label for _, label in metrics])
    ax.invert_yaxis()
    ax.set_xlim(0, limit)
    ax.set_xlabel("Duration (ms) · median [95% confidence interval]", labelpad=12)
    fig.text(.045, .055, "Whiskers: approximate 95% block-bootstrap CI of the median (2,000 resamples). CI unavailable for n < 8 or gaps; assumes approximate stationarity.\n"
             "Input intervals overlap. FILE includes event-delivery delay; transfer before detection is unmeasured. Do not sum these intervals.\n"
             "Processing rows include their file I/O. Full per-volume timings remain in speed_test.csv.",
             fontsize=9, color="#52667c", linespacing=1.5)
    output = Path(output) if output else csv_path.with_suffix(".png")
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Run folder's speed_test.csv")
    parser.add_argument("--tr", type=float, default=1., help="Configured TR in seconds")
    args = parser.parse_args()
    print(plot_speed(args.csv, args.tr) or "No finished volume timings to plot.")
