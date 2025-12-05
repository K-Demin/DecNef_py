# plot_scores.py
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore


def plot_scores(scores_csv: str | Path, out_png: str | Path | None = None):
    scores_csv = Path(scores_csv)
    vols = []
    raw_scores = []
    z_scores = []

    with open(scores_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vols.append(int(row["volume_idx"]))
            raw_scores.append(float(row["score_raw"]))
            z_val = row.get("score_z", "")
            try:
                z_scores.append(float(z_val))
            except ValueError:
                z_scores.append(np.nan)

    vols = np.array(vols)
    raw_scores = np.array(raw_scores)
    z_scores = np.array(z_scores)
    z_scores[np.isnan(z_scores)] = 0


    plt.figure(figsize=(10, 5))
    plt.plot(vols, zscore(raw_scores), label="raw decoder value (X·w)")
    plt.plot(vols, zscore(z_scores), label="z-scored decoder value", alpha=0.7)
    plt.xlabel("Volume index")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()

    if out_png is None:
        out_png = scores_csv.with_suffix("_plot.png")
    out_png = Path(out_png)
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved plot to {out_png}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python plot_scores.py /path/to/scores.csv [out.png]")
        raise SystemExit
    scores_csv = sys.argv[1]
    out_png = sys.argv[2] if len(sys.argv) > 2 else None
    plot_scores(scores_csv, out_png)
