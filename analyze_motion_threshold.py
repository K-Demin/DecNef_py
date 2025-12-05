#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def zscore(x):
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if s == 0 or np.isnan(s):
        return np.zeros_like(x, dtype=float)
    return (x - m) / s


def load_scores(run_dir: Path):
    """
    Expect: scores.csv with header:
    volume_idx, timestamp, score_raw, score_z
    """
    scores_path = run_dir / "scores.csv"
    if not scores_path.exists():
        return None, None

    data = np.genfromtxt(scores_path, delimiter=",", names=True)
    if data.size == 0:
        return None, None

    # volume_idx is int, score_z is float
    vol_idx = data["volume_idx"].astype(int)
    z_dec = data["score_z"].astype(float)
    return vol_idx, z_dec


def load_fd_from_csv(run_dir: Path):
    """
    Load FD from fd_rt.csv:
    header: volume_idx,fd
    """
    fd_path = run_dir / "fd_rt.csv"
    if not fd_path.exists():
        return None, None

    data = np.genfromtxt(fd_path, delimiter=",", names=True)
    if data.size == 0:
        return None, None

    vol_idx = data["volume_idx"].astype(int)
    fd = data["fd"].astype(float)
    return vol_idx, fd


def align_by_volume_idx(vol_fd, fd, vol_dec, z_dec):
    """
    Align FD and decoder series by volume_idx intersection.
    Returns aligned arrays: fd_aligned, z_dec_aligned, vols.
    """
    fd_dict = {int(v): float(f) for v, f in zip(vol_fd, fd)}
    dec_dict = {int(v): float(z) for v, z in zip(vol_dec, z_dec)}

    common_vols = sorted(set(fd_dict.keys()) & set(dec_dict.keys()))
    if not common_vols:
        return None, None, None

    fd_aligned = np.array([fd_dict[v] for v in common_vols], dtype=float)
    z_dec_aligned = np.array([dec_dict[v] for v in common_vols], dtype=float)
    return fd_aligned, z_dec_aligned, np.array(common_vols, dtype=int)


def find_runs(base: Path, sub_filter: str | None = None):
    """
    Find all run dirs that look like:
      base / sub-*/day/func/run
    and contain scores.csv AND fd_rt.csv.
    """
    if sub_filter is None:
        sub_dirs = sorted([d for d in base.glob("sub-*") if d.is_dir()])
    else:
        sub_dirs = [base / f"sub-{sub_filter}"]
        if not sub_dirs[0].is_dir():
            raise FileNotFoundError(f"No such subject dir: {sub_dirs[0]}")

    run_dirs = []

    for sub in sub_dirs:
        for day_dir in sorted(sub.glob("*")):
            if not day_dir.is_dir():
                continue
            func_dir = day_dir / "func"
            if not func_dir.is_dir():
                continue
            for run_dir in sorted(func_dir.glob("*")):
                if not run_dir.is_dir():
                    continue
                if (run_dir / "scores.csv").exists() and (run_dir / "fd_rt.csv").exists():
                    run_dirs.append(run_dir)

    return run_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate FD and decoder scores to help choose a motion threshold."
    )
    parser.add_argument(
        "--base-data",
        default="/SSD2/DecNef_py/data",
        help="Base data folder (same as RT pipeline).",
    )
    parser.add_argument(
        "--sub",
        default=None,
        help="Optional subject filter, e.g. 00085. If omitted, uses all sub-*.",
    )
    parser.add_argument(
        "--out-prefix",
        default="motion_decoder_analysis",
        help="Prefix for output plots/files.",
    )
    args = parser.parse_args()

    base = Path(args.base_data)
    run_dirs = find_runs(base, sub_filter=args.sub)

    if not run_dirs:
        raise RuntimeError("No run dirs with both scores.csv and fd_rt.csv found.")

    print(f"Found {len(run_dirs)} runs with FD + decoder scores.")

    FD_all = []
    Z_all = []
    dZ_all = []

    for run_dir in run_dirs:
        vol_dec, z_dec = load_scores(run_dir)
        vol_fd, fd = load_fd_from_csv(run_dir)
        if vol_dec is None or vol_fd is None:
            continue

        fd_aligned, z_aligned, vols = align_by_volume_idx(vol_fd, fd, vol_dec, z_dec)
        if fd_aligned is None:
            continue

        # keep only finite values
        mask = np.isfinite(fd_aligned) & np.isfinite(z_aligned)
        fd_aligned = fd_aligned[mask]
        z_aligned = z_aligned[mask]

        if fd_aligned.size == 0:
            continue

        FD_all.append(fd_aligned)
        Z_all.append(z_aligned)
        # per-run Δdecoder (one-step diff), pad first with NaN
        dZ = np.empty_like(z_aligned)
        dZ[0] = np.nan
        dZ[1:] = np.diff(z_aligned)
        dZ_all.append(dZ)

    if not FD_all:
        raise RuntimeError("No valid FD/decoder pairs after alignment.")

    FD_all = np.concatenate(FD_all)
    Z_all = np.concatenate(Z_all)
    dZ_all = np.concatenate(dZ_all)

    # Remove NaN dZ for correlation
    mask_dZ = np.isfinite(FD_all) & np.isfinite(dZ_all)
    FD_for_dZ = FD_all[mask_dZ]
    dZ_for_corr = np.abs(dZ_all[mask_dZ])

    print(f"Total usable volumes: {FD_all.size}")

    # -------------------- basic distribution of FD -------------------- #
    q = np.percentile(FD_all, [50, 75, 90, 95, 99])
    print("FD quantiles [mm]:")
    print(f"  50%: {q[0]:.4f}")
    print(f"  75%: {q[1]:.4f}")
    print(f"  90%: {q[2]:.4f}")
    print(f"  95%: {q[3]:.4f}")
    print(f"  99%: {q[4]:.4f}")

    # Overall corr between FD and |Δdecoder|
    def safe_corr(a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b)
        a = a[m]
        b = b[m]
        if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    r_all = safe_corr(FD_for_dZ, dZ_for_corr)
    print(f"Corr(FD, |Δdecoder_z|) across all volumes: {r_all:.4f}")

    # -------------------- candidate thresholds -------------------- #
    # Some fixed + quantile-based candidates
    candidates = sorted(set([
        0.2, 0.3, 0.4, 0.5,
        float(q[2]),  # 90th
        float(q[3]),  # 95th
        float(q[4]),  # 99th
    ]))

    print("\nCandidate thresholds and proportion of volumes above them:")
    for thr in candidates:
        frac = np.mean(FD_all > thr)
        r_low = safe_corr(FD_for_dZ[FD_for_dZ <= thr],
                          dZ_for_corr[FD_for_dZ <= thr])
        print(f"  FD > {thr:.3f} → {frac*100:5.1f}% of volumes; "
              f"corr(FD, |ΔZ|) below thr = {r_low:.4f}")

    # -------------------- plots -------------------- #
    out_prefix = Path(args.out_prefix)

    # 1) Histogram of FD
    plt.figure(figsize=(8, 5))
    plt.hist(FD_all, bins=50)
    for thr in candidates:
        plt.axvline(thr, linestyle="--")
    plt.xlabel("FD [mm]")
    plt.ylabel("Count")
    plt.title("Distribution of FD across all runs")
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix("_FD_hist.png"), dpi=200)
    plt.close()

    # 2) FD vs |Δdecoder| scatter
    plt.figure(figsize=(8, 5))
    plt.scatter(FD_for_dZ, dZ_for_corr, s=4, alpha=0.4)
    for thr in candidates:
        plt.axvline(thr, linestyle="--")
    plt.xlabel("FD [mm]")
    plt.ylabel("|Δ decoder_z|")
    plt.title("FD vs |Δdecoder_z| (per TR)")
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix("_FD_vs_dZ.png"), dpi=200)
    plt.close()

    # 3) FD and decoder_z as z-scores vs volume index (sorted by FD)
    order = np.argsort(FD_all)
    FD_sorted = FD_all[order]
    Z_sorted = Z_all[order]
    z_FD = zscore(FD_sorted)
    z_Z = zscore(Z_sorted)

    plt.figure(figsize=(10, 5))
    plt.plot(z_FD, label="z(FD)")
    plt.plot(z_Z, label="z(decoder_z)")
    for thr in candidates:
        # draw vertical at approx position where FD crosses thr
        idx_thr = np.searchsorted(FD_sorted, thr)
        plt.axvline(idx_thr, linestyle="--", alpha=0.5)
    plt.xlabel("Volumes sorted by FD")
    plt.ylabel("z-score")
    plt.title("z(FD) and z(decoder_z) vs sorted volumes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix("_zFD_zDec_sorted.png"), dpi=200)
    plt.close()

    print(f"\nSaved plots with prefix {out_prefix}")


if __name__ == "__main__":
    main()
