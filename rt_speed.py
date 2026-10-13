"""Opt-in wall-clock profiling; plot after acquisition, never in the TR loop."""
import argparse
import csv
import logging
import math
import threading
import time
from pathlib import Path

STAGES = {
    "DICOM": "DICOM conversion", "FMAP": "PyHySCO unwarping",
    "MC": "Motion correction + FD", "QC": "DVARS / censor flags",
    "REG": "Cumulative GLM / scaling", "TRANS": "Spatial resampling",
    "SMOOTH": "Masked smoothing", "MASK": "Analysis mask",
    "SCORE": "Decoder + score output", "OUTPUT": "QC / preview output",
    "OTHER": "Other processing overhead",
}
FIELDS = [
    "volume", "source", "cached_input", "attempts", "reg_ready", "status",
    "arrival_interval_ms", "idle_wait_ms", "worker_queue_ms", "order_queue_ms",
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

    def values(key, subset=rows):
        return np.array([float(r[key]) if r.get(key) != "" else np.nan for r in subset])

    good = [r for r in rows if r["status"] == "ok"]
    ready = [r for r in good if r["reg_ready"] == "True"]
    measured = ready or good
    title_suffix = "regression-ready volumes" if ready else "completed volumes (warm-up)"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(16, 9), facecolor="#f5f8fc")
    # Fixed 16:9 margins leave room for long stage labels and a slide-sized font.
    ax_bar = fig.add_axes([.215, .18, .265, .61])
    ax_run = fig.add_axes([.595, .555, .37, .235])
    ax_input = fig.add_axes([.595, .18, .37, .235])
    for ax in (ax_bar, ax_run, ax_input):
        ax.set_facecolor("#f5f8fc")
        ax.grid(axis="x" if ax is ax_bar else "y", color="#dae2ec", linewidth=.7)
        ax.set_axisbelow(True)
    fig.text(.045, .94, "Real-time processing · measured speed", fontsize=25,
             weight="bold", color="#102b46")
    backlog = sum(r["source"] == "backlog" for r in rows)
    cached = sum(r["cached_input"] == "True" for r in rows)
    failed = len(rows) - len(good)
    fig.text(.045, .892,
             f"{len(good)} completed  ·  {failed} failed  ·  {backlog} backlog  ·  "
             f"{cached} reused NIfTI  ·  configured TR {tr:g} s", color="#52667c")

    stages = [s for s in STAGES if measured and np.any(values(f"{s}_ms", measured) > 0)]
    if stages:
        med = [np.median(values(f"{s}_ms", measured)) for s in stages]
        p90 = [np.percentile(values(f"{s}_ms", measured), 90) for s in stages]
        y = np.arange(len(stages))
        ax_bar.barh(y, med, color="#167d9b", height=.56, label="Median")
        ax_bar.scatter(p90, y, s=34, color="#ed9856", zorder=3, label="90th percentile")
        limit = max(p90) * 1.34 or 1.
        for yi, m, p in zip(y, med, p90):
            fmt = lambda value: f"{value:.1f}" if value < 10 else f"{value:.0f}"
            ax_bar.text(p + limit*.025, yi, f"{fmt(m)} / {fmt(p)}", va="center", fontsize=10)
        ax_bar.set_yticks(y, [STAGES[s] for s in stages])
        ax_bar.invert_yaxis()
        ax_bar.set_xlim(0, limit)
        ax_bar.legend(loc="lower right", frameon=True, facecolor="#f5f8fc",
                      edgecolor="none", fontsize=10)
    else:
        ax_bar.text(.1, .5, "No completed stage timings", transform=ax_bar.transAxes)
    ax_bar.set_title(f"Per-stage duration\n{title_suffix}", loc="left", fontsize=12,
                     weight="bold", pad=16)
    ax_bar.set_xlabel("Milliseconds · median / P90")

    x = values("volume")
    processing = values("processing_ms") / 1000
    latency = values("latency_ms") / 1000
    ok = np.array([r["status"] == "ok" for r in rows])
    warmup = ok & np.array([r["reg_ready"] != "True" for r in rows])
    ax_run.plot(x, np.where(ok, latency, np.nan), color="#8d9aab", linewidth=1.1,
                label="Detection → completion")
    ax_run.plot(x, np.where(ok, processing, np.nan), color="#167d9b", linewidth=1.6,
                label="Processing (incl. file I/O)")
    ax_run.scatter(x[warmup], processing[warmup], s=20, color="#ed9856", label="Warm-up", zorder=4)
    if np.any(~ok):
        ax_run.scatter(x[~ok], latency[~ok], marker="x", color="#b94252", label="Failed")
    ax_run.axhline(tr, color="#b94252", linestyle="--", linewidth=1, label="TR budget")
    ax_run.set(title="Per-volume processing and latency", ylabel="Seconds", xlabel="Volume")
    ax_run.set_ylim(bottom=0)
    ax_run.set_ylim(0, max(float(np.nanmax(latency)), tr, .001) * 1.35)
    ax_run.legend(loc="upper left", fontsize=9, frameon=True,
                  facecolor="#f5f8fc", edgecolor="none", ncol=2)

    live = [r for r in rows if r["source"] == "live"]
    live_x = values("volume", live)
    if any(r["arrival_interval_ms"] != "" for r in live):
        ax_input.plot(live_x, values("arrival_interval_ms", live)/1000,
                      color="#7066af", linewidth=1.5, label="Incoming-file interval")
        ax_input.plot(live_x, values("idle_wait_ms", live)/1000,
                      color="#b2bcc9", linewidth=1.2, label="Idle waiting for next file")
        queue = values("worker_queue_ms", live) + values("order_queue_ms", live)
        ax_input.plot(live_x, queue/1000, color="#ed9856", linewidth=1.1, label="Worker + ordering queue")
        input_peak = np.concatenate([values("arrival_interval_ms", live),
                                     values("idle_wait_ms", live), queue]) / 1000
        ax_input.set_ylim(0, max(float(np.nanmax(input_peak)), .001) * 1.5)
        ax_input.legend(loc="upper left", fontsize=9, frameon=True,
                        facecolor="#f5f8fc", edgecolor="none")
    else:
        ax_input.text(.03, .48, "Waiting for at least two live arrivals.\n"
                      "Existing files do not measure scanner cadence.", transform=ax_input.transAxes,
                      color="#52667c")
    ax_input.set(title="Incoming volumes and waiting", ylabel="Seconds", xlabel="Volume")
    ax_input.set_ylim(bottom=0)
    fig.text(.045, .055, "Processing includes file I/O and attempted work. Detection-to-completion also includes queues and retry backoff.\n"
             "Incoming intervals start at file detection, not scanner triggers. Interrupted work may be absent. Cumulative GLM cost can grow during the run.",
             fontsize=10, color="#52667c", linespacing=1.6)
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
