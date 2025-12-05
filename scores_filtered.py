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
    scores_path = run_dir / "scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"scores.csv not found at {scores_path}")

    data = np.genfromtxt(scores_path, delimiter=",", names=True)
    vol_idx = data["volume_idx"].astype(int)
    z_dec = data["score_z"].astype(float)
    return vol_idx, z_dec


def load_fd_from_csv(run_dir: Path):
    fd_path = run_dir / "fd_rt.csv"
    if not fd_path.exists():
        return None, None

    data = np.genfromtxt(fd_path, delimiter=",", names=True)
    vol_idx = data["volume_idx"].astype(int)
    fd = data["fd"].astype(float)
    return vol_idx, fd


def load_fd_from_motion(run_dir: Path, radius_mm: float = 50.0):
    motion_path = run_dir / "motion_rt.1D"
    if not motion_path.exists():
        return None, None

    motion = np.loadtxt(motion_path)
    if motion.ndim == 1:
        motion = motion[None, :]

    delta = np.zeros_like(motion)
    delta[1:, :] = motion[1:, :] - motion[:-1, :]

    d_trans = delta[:, 0:3]
    d_rot_deg = delta[:, 3:6]
    d_rot_rad = d_rot_deg * np.pi / 180.0
    d_rot_mm = d_rot_rad * radius_mm

    fd = np.sum(np.abs(d_trans), axis=1) + np.sum(np.abs(d_rot_mm), axis=1)
    vol_idx = np.arange(1, len(fd) + 1, dtype=int)

    return vol_idx, fd


def align_by_volume_idx(vol_fd, fd, vol_dec, z_dec):
    fd_dict = {int(v): float(f) for v, f in zip(vol_fd, fd)}
    dec_dict = {int(v): float(z) for v, z in zip(vol_dec, z_dec)}

    common = sorted(set(fd_dict.keys()) & set(dec_dict.keys()))
    if not common:
        raise RuntimeError("No overlapping volume indices")

    fd_aligned = np.array([fd_dict[v] for v in common])
    dec_aligned = np.array([dec_dict[v] for v in common])
    return fd_aligned, dec_aligned, np.array(common, dtype=int)


def build_censor_mask(fd, fd_thresh, censor_next=False):
    censored = fd >= fd_thresh
    if censor_next:
        next_bad = np.zeros_like(censored)
        next_bad[1:] = censored[:-1]
        censored = np.logical_or(censored, next_bad)
    return censored


def interpolate_over_censored(series, censored_mask):
    series = np.asarray(series, dtype=float)
    idx = np.arange(series.size)
    good = (~censored_mask) & ~np.isnan(series)

    if good.sum() < 2:
        return series.copy()

    return np.interp(idx, idx[good], series[good])


def main():
    parser = argparse.ArgumentParser(
        description="Plot decoder values with FD censoring and save filtered smoothed results."
    )
    parser.add_argument("--sub", required=True)
    parser.add_argument("--day", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--base-data",
                        default="/SSD2/DecNef_py/data")
    parser.add_argument("--fd-thresh", type=float, default=0.5)
    parser.add_argument("--radius-mm", type=float, default=50.0)
    parser.add_argument("--censor-next", action="store_true")
    parser.add_argument("--no-zscore-decoder", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    base = Path(args.base_data)
    run_dir = base / f"sub-{args.sub}" / args.day / "func" / args.run
    print(f"Using run_dir: {run_dir}")

    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder missing: {run_dir}")

    # ---- load data ----
    vol_dec, dec = load_scores(run_dir)

    vol_fd, fd = load_fd_from_csv(run_dir)
    if vol_fd is None:
        vol_fd, fd = load_fd_from_motion(run_dir, radius_mm=args.radius_mm)
    if vol_fd is None:
        raise FileNotFoundError("No fd_rt.csv or motion_rt.1D")

    fd, dec, vols = align_by_volume_idx(vol_fd, fd, vol_dec, dec)

    # Z-score decoder across run (usual)
    if args.no_zscore_decoder:
        dec_orig = dec.copy()
    else:
        dec_orig = zscore(dec)

    fd_z = zscore(fd)

    # ---- censor ----
    censored = build_censor_mask(fd, args.fd_thresh, censor_next=args.censor_next)
    dec_interp = interpolate_over_censored(dec_orig, censored)

    # final series = original where OK, interpolated where censored
    dec_final = dec_orig.copy()
    dec_final[censored] = dec_interp[censored]

    # ---- save filtered results ----
    out_csv = run_dir / "scores_filtered.csv"
    print(f"Saving scores_filtered.csv → {out_csv}")

    header = "volume_idx,fd,censored,decoder_original,decoder_interpolated,decoder_final"
    out_arr = np.column_stack([
        vols,
        fd,
        censored.astype(int),
        dec_orig,
        dec_interp,
        dec_final,
    ])
    np.savetxt(out_csv, out_arr, delimiter=",", header=header, comments="", fmt="%.6f")

    # ---- plotting ----
    if args.out is None:
        out_png = run_dir / "decoder_fd_censor.png"
    else:
        out_png = Path(args.out)

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # FD
    ax[0].plot(vols, fd_z, label="FD (z)")
    ax[0].scatter(vols[censored], fd_z[censored], color="red",
                  label="censored", s=15, zorder=5)
    ax[0].set_ylabel("FD (z)")
    ax[0].legend()

    # Decoder
    ax[1].plot(vols, dec_orig, alpha=0.4, label="decoder original")
    ax[1].plot(vols, dec_interp, linewidth=2, label="decoder interpolated")
    ax[1].scatter(vols[censored], dec_orig[censored], color="red", s=15)
    ax[1].plot(vols, dec_final, linewidth=2, label="decoder final (mixed)")
    ax[1].set_xlabel("TR")
    ax[1].set_ylabel("Decoder (z)")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved plot → {out_png}")


if __name__ == "__main__":
    main()
