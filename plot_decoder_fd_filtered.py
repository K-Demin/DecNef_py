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
    (as written by the RT pipeline)
    """
    scores_path = run_dir / "scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"scores.csv not found at {scores_path}")

    data = np.genfromtxt(scores_path, delimiter=",", names=True)
    vol_idx = data["volume_idx"].astype(int)
    z_dec = data["score_z"].astype(float)
    return vol_idx, z_dec


def load_fd_from_csv(run_dir: Path):
    """
    Prefer fd_rt.csv (volume_idx,fd) if it exists.
    """
    fd_path = run_dir / "fd_rt.csv"
    if not fd_path.exists():
        return None, None

    data = np.genfromtxt(fd_path, delimiter=",", names=True)
    vol_idx = data["volume_idx"].astype(int)
    fd = data["fd"].astype(float)
    return vol_idx, fd


def load_fd_from_motion(run_dir: Path, radius_mm: float = 50.0):
    """
    Fallback: compute FD from motion_rt.1D (AFNI / RtpVolreg style):
      cols 0–2: translations (mm)
      cols 3–5: rotations (deg) -> radians -> mm via radius.

    FD(t) = sum_i |Δtrans_i(t)| + sum_j |radius * Δrot_rad_j(t)|
    """
    motion_path = run_dir / "motion_rt.1D"
    if not motion_path.exists():
        return None, None

    motion = np.loadtxt(motion_path)  # shape [T, 6]
    if motion.ndim == 1:
        motion = motion[None, :]

    # Δ params between successive TRs
    delta = np.zeros_like(motion)
    delta[1:, :] = motion[1:, :] - motion[:-1, :]

    # translations: mm
    d_trans = delta[:, 0:3]

    # rotations: degrees → radians → mm
    d_rot_deg = delta[:, 3:6]
    d_rot_rad = d_rot_deg * np.pi / 180.0
    d_rot_mm = d_rot_rad * radius_mm

    fd = np.sum(np.abs(d_trans), axis=1) + np.sum(np.abs(d_rot_mm), axis=1)
    vol_idx = np.arange(1, len(fd) + 1, dtype=int)

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
        raise RuntimeError("No overlapping volume indices between FD and scores.csv")

    fd_aligned = np.array([fd_dict[v] for v in common_vols], dtype=float)
    z_dec_aligned = np.array([dec_dict[v] for v in common_vols], dtype=float)
    return fd_aligned, z_dec_aligned, np.array(common_vols, dtype=int)


def build_censor_mask(fd, fd_thresh, censor_next=False):
    """
    Returns boolean mask of censored TRs:
      censored[k] = True if fd[k] >= fd_thresh
      plus, if censor_next=True, also censored[k+1] when fd[k] >= fd_thresh
    """
    fd = np.asarray(fd, dtype=float)
    censored = fd >= fd_thresh

    if censor_next:
        # mark k+1 if k is censored
        next_bad = np.zeros_like(censored)
        next_bad[1:] = censored[:-1]
        censored = np.logical_or(censored, next_bad)

    return censored


def interpolate_over_censored(series, censored_mask):
    """
    Linearly interpolate over censored points.
    series: 1D array
    censored_mask: True for "bad" points to replace by interpolation

    Returns a new array with interpolated values for censored points.
    """
    series = np.asarray(series, dtype=float)
    idx = np.arange(series.size)

    good = (~censored_mask) & ~np.isnan(series)
    if good.sum() < 2:
        # not enough points for interpolation → just return original
        return series.copy()

    interp = np.interp(idx, idx[good], series[good])
    return interp


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot decoder values with motion censoring and interpolated (smoothed) "
            "decoder over censored TRs."
        )
    )
    parser.add_argument("--sub", required=True, help="Subject ID, e.g. 00086")
    parser.add_argument("--day", required=True, help="Day/session, e.g. 3")
    parser.add_argument("--run", required=True, help="Run/block, e.g. 13")
    parser.add_argument(
        "--base-data",
        default="/SSD2/DecNef_py/data",
        help="Base data folder (same as RT pipeline).",
    )
    parser.add_argument(
        "--fd-thresh",
        type=float,
        default=0.5,
        help="FD threshold (mm) for censoring.",
    )
    parser.add_argument(
        "--radius-mm",
        type=float,
        default=50.0,
        help="Brain radius (mm) for converting rotations to FD (if using motion_rt.1D).",
    )
    parser.add_argument(
        "--censor-next",
        action="store_true",
        help="Also censor TR+1 after any TR with FD >= threshold.",
    )
    parser.add_argument(
        "--no-zscore-decoder",
        action="store_true",
        help="If set, do NOT z-score decoder across run (use score_z as is).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path (default: run_dir/decoder_fd_censor.png).",
    )
    args = parser.parse_args()

    base = Path(args.base_data)
    run_dir = base / f"sub-{args.sub}" / args.day / "func" / args.run
    print(f"Using run_dir: {run_dir}")

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # --- load decoder scores (score_z from RT) ---
    vol_dec, dec_z = load_scores(run_dir)

    # --- load FD (CSV preferred, otherwise motion_rt.1D) ---
    vol_fd, fd = load_fd_from_csv(run_dir)
    if vol_fd is None:
        print("fd_rt.csv not found; trying motion_rt.1D")
        vol_fd, fd = load_fd_from_motion(run_dir, radius_mm=args.radius_mm)
    if vol_fd is None:
        raise FileNotFoundError("Neither fd_rt.csv nor motion_rt.1D found in run_dir")

    # --- align series ---
    fd_aligned, dec_aligned, vols = align_by_volume_idx(vol_fd, fd, vol_dec, dec_z)

    # Optionally z-score decoder across the run (on aligned volumes)
    if args.no_zscore_decoder:
        dec_for_plot = dec_aligned.copy()
    else:
        dec_for_plot = zscore(dec_aligned)

    # z-score FD just for plotting on comparable scale
    fd_for_plot = zscore(fd_aligned)

    # --- censor mask & interpolated decoder ---
    censored = build_censor_mask(fd_aligned, fd_thresh=args.fd_thresh,
                                 censor_next=args.censor_next)

    dec_censored = dec_for_plot.copy()
    dec_censored[censored] = np.nan

    dec_interp = interpolate_over_censored(dec_for_plot, censored)

    # --- plotting ---
    if args.out is None:
        out_png = run_dir / "decoder_fd_censor.png"
    else:
        out_png = Path(args.out)

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Panel 1: FD
    ax[0].plot(vols, fd_for_plot, label="FD (z-scored)")
    # Mark censored TRs
    ax[0].scatter(
        vols[censored],
        fd_for_plot[censored],
        color="red",
        s=15,
        label="Censored (FD ≥ thr or neighbor)",
        zorder=5,
    )
    ax[0].axhline(
        zscore(fd_aligned)[fd_aligned >= args.fd_thresh].mean()
        if np.any(fd_aligned >= args.fd_thresh)
        else 0,
        linestyle="--",
        linewidth=1,
        label="Approx. threshold (z-space)",
    )
    ax[0].set_ylabel("FD (z-scored)")
    ax[0].set_title(
        f"FD and decoder censoring (thr = {args.fd_thresh} mm, "
        f"{'censor+1' if args.censor_next else 'censor only'} )"
    )
    ax[0].legend(loc="upper right")

    # Panel 2: decoder
    ax[1].plot(vols, dec_for_plot, label="Decoder (original)", alpha=0.4)
    ax[1].plot(
        vols, dec_interp, label="Decoder (interpolated over censored TRs)", linewidth=2
    )
    ax[1].scatter(
        vols[censored],
        dec_for_plot[censored],
        color="red",
        s=15,
        label="Censored TRs",
        zorder=5,
    )
    ax[1].set_xlabel("Volume index (TR)")
    ax[1].set_ylabel("Decoder value" + ("" if args.no_zscore_decoder else " (z)"))
    ax[1].legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"Saved plot → {out_png}")


if __name__ == "__main__":
    main()
