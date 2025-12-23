#!/usr/bin/env python
# Example:
# python plot_decoder_dvars_filtered.py --sub 00000 --day 1 --run 2 --dvars-thresh 3 --dvars-mode robust_z --censor-next --skip-trs 25
#
# DVARS notes:
# - "raw" DVARS is sqrt(mean((ΔI)^2)) across voxels in a mask.
# - For thresholding across runs, use standardized DVARS (z or robust z).
#
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def zscore(x):
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if s == 0 or np.isnan(s):
        return np.zeros_like(x, dtype=float)
    return (x - m) / s


def robust_zscore(x):
    """Robust z using MAD. Good when you have spikes."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        return np.zeros_like(x, dtype=float)
    # 1.4826 rescales MAD to match std for Gaussian
    return (x - med) / (1.4826 * mad)


def load_scores(run_dir: Path):
    """
    Expect: scores.csv with header:
    volume_idx, timestamp, score_raw, score_z
    """
    scores_path = run_dir / "scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"scores.csv not found at {scores_path}")

    data = np.genfromtxt(scores_path, delimiter=",", names=True)
    vol_idx = data["volume_idx"].astype(int)
    dec = data["score_raw"].astype(float)  # or score_z if you prefer
    return vol_idx, dec


def find_4d_nifti(run_dir: Path):
    """
    Tries to find a 4D NIfTI for DVARS computation.
    You can override with --nii4d.
    """
    candidates = [
        run_dir / "func.nii.gz",
        run_dir / "func.nii",
        run_dir / "epi.nii.gz",
        run_dir / "epi.nii",
        run_dir / "bold.nii.gz",
        run_dir / "bold.nii",
        run_dir / "analysis/rs_4d_mc.nii.gz"
    ]
    for p in candidates:
        if p.exists():
            return p

    # fallback: any 4D nii in folder
    for p in sorted(run_dir.glob("*.nii*")):
        try:
            img = nib.load(str(p))
            if len(img.shape) == 4 and img.shape[3] > 3:
                return p
        except Exception:
            continue
    return None


def load_mask(mask_path: Path, target_shape3):
    """Load mask and force it to be 3D and same shape as data's XYZ."""
    m = np.squeeze(nib.load(str(mask_path)).get_fdata())
    if m.ndim != 3:
        raise ValueError(f"Mask must be 3D; got shape {m.shape} from {mask_path}")
    if m.shape != target_shape3:
        raise ValueError(f"Mask shape {m.shape} != data shape {target_shape3}")
    return m != 0


def compute_dvars_from_4d(
    nii_4d: Path,
    mask_3d: np.ndarray | None,
    skip_trs: int = 0,
    use_psc: bool = False,
):
    """
    Compute DVARS per volume.
    If use_psc=True, compute DVARS on percent signal change (PSC).
    """
    img = nib.load(str(nii_4d))
    data = np.asanyarray(img.dataobj).astype(np.float32)

    if data.ndim != 4:
        raise ValueError(f"Expected 4D NIfTI, got {data.shape}")

    X, Y, Z, T = data.shape
    if T < 2:
        raise ValueError("Need at least 2 volumes")

    if mask_3d is None:
        mask_3d = np.ones((X, Y, Z), dtype=bool)
    else:
        if mask_3d.shape != (X, Y, Z):
            raise ValueError("Mask shape mismatch")

    # --- PSC normalization (voxel-wise) ---
    if use_psc:
        mean_vol = np.mean(data, axis=3, keepdims=True)
        mean_vol[mean_vol == 0] = np.nan
        data = 100.0 * (data / mean_vol - 1.0)

    dvars = np.zeros(T, dtype=float)
    prev = data[..., 0]

    for t in range(1, T):
        cur = data[..., t]
        diff = cur - prev
        dv = diff[mask_3d]
        dvars[t] = float(np.sqrt(np.mean(dv * dv)))
        prev = cur

    vol_idx = np.arange(1, T + 1, dtype=int)
    return vol_idx, dvars



def align_by_volume_idx(vol_a, a, vol_b, b):
    """Align two series by common volume_idx intersection."""
    da = {int(v): float(x) for v, x in zip(vol_a, a)}
    db = {int(v): float(x) for v, x in zip(vol_b, b)}
    common = sorted(set(da.keys()) & set(db.keys()))
    if not common:
        raise RuntimeError("No overlapping volume indices between the two series")
    aa = np.array([da[v] for v in common], dtype=float)
    bb = np.array([db[v] for v in common], dtype=float)
    return aa, bb, np.array(common, dtype=int)


def build_censor_mask(metric, thresh, censor_next=False):
    """Censor TRs where metric >= thresh; optionally also censor TR+1."""
    metric = np.asarray(metric, dtype=float)
    censored = metric >= thresh
    if censor_next:
        nxt = np.zeros_like(censored)
        nxt[1:] = censored[:-1]
        censored = np.logical_or(censored, nxt)
    return censored


def interpolate_over_censored(series, censored_mask):
    """Linear interpolate series over censored points."""
    series = np.asarray(series, dtype=float)
    idx = np.arange(series.size)
    good = (~censored_mask) & ~np.isnan(series)
    if good.sum() < 2:
        return series.copy()
    return np.interp(idx, idx[good], series[good])


def main():
    parser = argparse.ArgumentParser(
        description="Compute DVARS from a 4D NIfTI and plot decoder with DVARS-based censoring."
    )
    parser.add_argument("--sub", required=True, help="Subject ID, e.g. 00086")
    parser.add_argument("--day", required=True, help="Day/session, e.g. 3")
    parser.add_argument("--run", required=True, help="Run/block, e.g. 13")
    parser.add_argument(
        "--base-data",
        default="/SSD2/DecNef_py/data",
        help="Base data folder.",
    )
    parser.add_argument(
        "--nii4d",
        type=str,
        default=None,
        help="Path to 4D NIfTI to compute DVARS from (default: auto-detect in run_dir).",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Optional 3D mask NIfTI (same space as 4D data). If omitted, use all voxels.",
    )
    parser.add_argument(
        "--dvars-mode",
        choices=["raw", "z", "robust_z"],
        default="robust_z",
        help="How to scale DVARS for thresholding/plotting.",
    )
    parser.add_argument(
        "--dvars-thresh",
        type=float,
        default=2.5,
        help=(
            "Threshold applied in the chosen DVARS mode. "
            "Typical: robust_z 2.5–3.5; z 2–3; raw is dataset-specific."
        ),
    )
    parser.add_argument(
        "--censor-next",
        action="store_true",
        help="Also censor TR+1 after any TR with DVARS >= threshold.",
    )
    parser.add_argument(
        "--no-zscore-decoder",
        action="store_true",
        help="If set, do NOT z-score decoder across run (use raw).",
    )
    parser.add_argument(
        "--skip-trs",
        type=int,
        default=25,
        help="Skip first N TRs (default: 25) for plotting/censoring.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path (default: run_dir/decoder_dvars_censor.png).",
    )
    parser.add_argument(
        "--dvars-psc",
        action="store_true",
        help="Compute DVARS on percent signal change (PSC) instead of raw intensity.",
    )
    args = parser.parse_args()

    base = Path(args.base_data)
    run_dir = base / f"sub-{args.sub}" / args.day / "func" / args.run
    print(f"Using run_dir: {run_dir}")
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # --- load decoder ---
    vol_dec, dec = load_scores(run_dir)

    # --- locate 4D nifti for DVARS ---
    if args.nii4d is None:
        nii4d = find_4d_nifti(run_dir)
        if nii4d is None:
            raise FileNotFoundError(
                "Could not auto-detect a 4D NIfTI in run_dir. Provide --nii4d /path/to/file.nii.gz"
            )
    else:
        nii4d = Path(args.nii4d)
        if not nii4d.exists():
            raise FileNotFoundError(f"--nii4d not found: {nii4d}")

    print(f"DVARS source 4D NIfTI: {nii4d}")

    # --- load optional mask ---
    mask_3d = None
    if args.mask is not None:
        mask_path = Path(args.mask)
        if not mask_path.exists():
            raise FileNotFoundError(f"--mask not found: {mask_path}")
        # need target shape; load header quickly
        img = nib.load(str(nii4d))
        target_shape3 = img.shape[:3]
        mask_3d = load_mask(mask_path, target_shape3)
        print(f"Using DVARS mask: {mask_path}")

    # --- compute DVARS (raw) ---
    vol_dv, dvars_raw = compute_dvars_from_4d(
        nii4d,
        mask_3d,
        skip_trs=args.skip_trs,
        use_psc=args.dvars_psc,
    )

    # --- scale DVARS for thresholding ---
    if args.dvars_mode == "raw":
        dvars_metric = dvars_raw
    elif args.dvars_mode == "z":
        dvars_metric = zscore(dvars_raw)
    else:  # robust_z
        dvars_metric = robust_zscore(dvars_raw)

    # --- align decoder and DVARS by volume index ---
    dvars_aligned, dec_aligned, vols = align_by_volume_idx(vol_dv, dvars_metric, vol_dec, dec)

    # skip initial TRs
    keep = vols > args.skip_trs
    dvars_aligned = dvars_aligned[keep]
    dec_aligned = dec_aligned[keep]
    vols = vols[keep]

    # decoder scaling
    if args.no_zscore_decoder:
        dec_for_plot = dec_aligned.copy()
    else:
        dec_for_plot = zscore(dec_aligned)

    # plot DVARS on comparable scale
    dvars_for_plot = zscore(dvars_aligned) if args.dvars_mode == "raw" else dvars_aligned

    # censor mask in chosen DVARS mode
    censored = build_censor_mask(dvars_aligned, args.dvars_thresh, censor_next=args.censor_next)

    dec_censored = dec_for_plot.copy()
    dec_censored[censored] = np.nan
    dec_interp = interpolate_over_censored(dec_for_plot, censored)

    # --- plotting ---
    out_png = Path(args.out) if args.out is not None else (run_dir / "decoder_dvars_censor.png")

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Panel 1: DVARS
    ax[0].plot(vols, dvars_for_plot, label=f"DVARS ({args.dvars_mode})")
    ax[0].scatter(
        vols[censored],
        dvars_for_plot[censored],
        color="red",
        s=15,
        label="Censored (DVARS ≥ thr or neighbor)",
        zorder=5,
    )
    ax[0].axhline(args.dvars_thresh, linestyle="--", linewidth=1, label="Threshold")
    ax[0].set_ylabel("DVARS (scaled)")
    psc_tag = "PSC" if args.dvars_psc else "raw"
    ax[0].set_title(
        f"DVARS ({psc_tag}) and decoder censoring "
        f"(mode={args.dvars_mode}, thr={args.dvars_thresh}, "
        f"{'censor+1' if args.censor_next else 'censor only'})"
    )
    ax[0].legend(loc="upper right")

    # Panel 2: decoder
    ax[1].plot(vols, dec_for_plot, label="Decoder (original)", alpha=0.4)
    ax[1].plot(vols, dec_interp, label="Decoder (interpolated over censored TRs)", linewidth=2)
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
