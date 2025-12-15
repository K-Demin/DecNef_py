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
    volume_idx, timestamp, score_raw
    (as written by your RT pipeline)
    """
    scores_path = run_dir / "scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"scores.csv not found at {scores_path}")

    data = np.genfromtxt(scores_path, delimiter=",", names=True)
    vol_idx = data["volume_idx"].astype(int)
    z_dec = data["score_raw"].astype(float)
    vol_idx = vol_idx[10:]
    z_dec = z_dec[10:]
    return vol_idx, z_dec


def load_fd_from_csv(run_dir: Path):
    """
    Preferred path: FD from fd_rt.csv.
    Expected header: volume_idx,fd
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
      cols 3–5: rotations (deg) -> convert to radians, then to mm via radius.

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


def main():
    parser = argparse.ArgumentParser(
        description="Plot z-scored FD and denoised decoder together."
    )
    parser.add_argument("--sub", required=True, help="Subject ID, e.g. 00085")
    parser.add_argument("--day", required=True, help="Day/session, e.g. 4")
    parser.add_argument("--run", required=True, help="Run/block, e.g. 13")
    parser.add_argument(
        "--base-data",
        default="/SSD2/DecNef_py/data",
        help="Base data folder (same as RT pipeline).",
    )
    parser.add_argument(
        "--radius-mm",
        type=float,
        default=50.0,
        help="Brain radius (mm) for converting rotations to FD (if using motion_rt.1D).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output PNG path (default: fd_vs_decoder_z.png in run dir).",
    )
    args = parser.parse_args()

    base = Path(args.base_data)
    run_dir = base / f"sub-{args.sub}" / args.day / "func" / args.run
    print(f"Using run_dir: {run_dir}")

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # --- load decoder scores ---
    vol_dec, z_dec = load_scores(run_dir)

    # --- load FD (fd_rt.csv preferred, otherwise motion_rt.1D) ---
    vol_fd, fd = load_fd_from_csv(run_dir)
    if vol_fd is None:
        print("fd_rt.csv not found; trying motion_rt.1D")
        vol_fd, fd = load_fd_from_motion(run_dir, radius_mm=args.radius_mm)
    if vol_fd is None:
        raise FileNotFoundError("Neither fd_rt.csv nor motion_rt.1D found in run_dir")

    # --- align FD and decoder series by volume index ---
    fd_aligned, z_dec_aligned, vols = align_by_volume_idx(vol_fd, fd, vol_dec, z_dec)

    # --- z-score both (ignoring NaNs) ---
    z_fd = zscore(fd_aligned)
    z_dec_norm = zscore(z_dec_aligned)

    # --- simple mask to drop positions where decoder is NaN (e.g., pre-baseline) ---
    valid = ~np.isnan(z_dec_norm)
    z_fd = z_fd[valid]
    z_dec_norm = z_dec_norm[valid]
    vols_valid = vols[valid]

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(vols_valid, z_fd, label="FD (z)", linewidth=1.5)
    ax.plot(vols_valid, z_dec_norm, label="Decoder z-score (z of z)", linewidth=1.5)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)

    ax.set_xlabel("Volume index (TR)")
    ax.set_ylabel("Z-scored value")
    ax.set_title(f"FD vs Decoder (z-scored)\nsub-{args.sub}, day {args.day}, run {args.run}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_png = Path(args.out) if args.out is not None else (run_dir / "fd_vs_decoder_z.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"Saved plot to {out_png}")


if __name__ == "__main__":
    main()
