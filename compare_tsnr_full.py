#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, binary_erosion

def load_nii(p):
    img = nib.load(p)
    data = np.asanyarray(img.dataobj).astype(np.float32)
    return img, data

def as_mask(x):
    m = (x != 0) & np.isfinite(x)
    return m

def finite(x):
    return x[np.isfinite(x)]

def stat(x):
    x = finite(x)
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else float("nan"),
        "p05": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
    }

def fmt(d):
    if d.get("n", 0) == 0:
        return "n=0"
    return (f"n={d['n']:,} mean={d['mean']:.3f} med={d['median']:.3f} "
            f"std={d['std']:.3f} p05={d['p05']:.3f} p95={d['p95']:.3f}")

def dice(a, b):
    inter = np.count_nonzero(a & b)
    na = np.count_nonzero(a)
    nb = np.count_nonzero(b)
    return (2 * inter / (na + nb)) if (na + nb) > 0 else np.nan

def jaccard(a, b):
    inter = np.count_nonzero(a & b)
    uni = np.count_nonzero(a | b)
    return (inter / uni) if uni > 0 else np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsnr1", required=True)
    ap.add_argument("--mask1", required=True)
    ap.add_argument("--tsnr2", required=True)
    ap.add_argument("--mask2", required=True)
    ap.add_argument("--label1", default="run1")
    ap.add_argument("--label2", default="run2")
    ap.add_argument("--bins", default="0,2,4,6,10,999", help="distance-to-edge bins in voxels, comma-separated")
    ap.add_argument("--core-erosion", type=int, default=2, help="erosion voxels for a conservative core mask")
    args = ap.parse_args()

    img1, t1 = load_nii(args.tsnr1)
    img2, t2 = load_nii(args.tsnr2)
    _, m1d = load_nii(args.mask1)
    _, m2d = load_nii(args.mask2)

    if t1.shape != t2.shape or t1.shape != m1d.shape or t2.shape != m2d.shape:
        raise SystemExit(f"Shape mismatch: {t1.shape=} {t2.shape=} {m1d.shape=} {m2d.shape=}")

    m1 = as_mask(m1d)
    m2 = as_mask(m2d)

    inter = m1 & m2 & np.isfinite(t1) & np.isfinite(t2)
    uni = (m1 | m2)
    only1 = m1 & ~m2 & np.isfinite(t1)
    only2 = m2 & ~m1 & np.isfinite(t2)

    n1 = np.count_nonzero(m1)
    n2 = np.count_nonzero(m2)
    ni = np.count_nonzero(inter)
    nu = np.count_nonzero(uni)

    print("\n=== COVERAGE / OVERLAP ===")
    print(f"{args.label1}: mask voxels = {n1:,}")
    print(f"{args.label2}: mask voxels = {n2:,}")
    print(f"Intersection voxels  = {ni:,}  ({(ni/n1*100 if n1 else np.nan):.1f}% of {args.label1}, {(ni/n2*100 if n2 else np.nan):.1f}% of {args.label2})")
    print(f"Union voxels         = {nu:,}")
    print(f"Dice   = {dice(m1,m2):.4f}")
    print(f"Jaccard= {jaccard(m1,m2):.4f}")

    print("\n=== BASIC tSNR STATS IN REGIONS ===")
    print(f"{args.label1} in its mask:      {fmt(stat(t1[m1]))}")
    print(f"{args.label2} in its mask:      {fmt(stat(t2[m2]))}")
    print(f"{args.label1} in intersection:  {fmt(stat(t1[inter]))}")
    print(f"{args.label2} in intersection:  {fmt(stat(t2[inter]))}")

    if ni > 0:
        diff = t2[inter] - t1[inter]
        rel = diff / np.clip(t1[inter], 1e-6, None)
        corr = float(np.corrcoef(t1[inter], t2[inter])[0,1]) if (np.std(t1[inter])>0 and np.std(t2[inter])>0) else np.nan
        print(f"\nPaired Δ (intersection): mean={float(np.mean(diff)):.3f}  med={float(np.median(diff)):.3f}")
        print(f"Paired %Δ (intersection): mean={float(np.mean(rel))*100:.3f}%  med={float(np.median(rel))*100:.3f}%")
        print(f"Corr(tSNR1,tSNR2) in intersection: {corr:.5f}")

    print(f"\n{args.label1}-only region (coverage unique): {fmt(stat(t1[only1]))}  vox={np.count_nonzero(only1):,}")
    print(f"{args.label2}-only region (coverage unique): {fmt(stat(t2[only2]))}  vox={np.count_nonzero(only2):,}")

    # Conservative "core" comparison to reduce edge/mask artifacts
    if args.core_erosion > 0:
        core = binary_erosion(uni, iterations=args.core_erosion)
        core &= m1 & m2 & np.isfinite(t1) & np.isfinite(t2)
        nc = np.count_nonzero(core)
        print(f"\n=== CONSERVATIVE CORE (erosion={args.core_erosion} vox) ===")
        print(f"Core voxels: {nc:,}")
        if nc > 0:
            print(f"{args.label1} core: {fmt(stat(t1[core]))}")
            print(f"{args.label2} core: {fmt(stat(t2[core]))}")
            d = t2[core] - t1[core]
            print(f"Δ core: mean={float(np.mean(d)):.3f}  med={float(np.median(d)):.3f}")

    # Center vs periphery bins based on distance-to-edge within UNION (consistent definition)
    print("\n=== CENTER vs PERIPHERY (distance-to-edge bins, in voxels; defined on UNION mask) ===")
    bins = [float(x) for x in args.bins.split(",")]
    dist_in = distance_transform_edt(uni)  # distance to background for voxels inside union
    # Only analyze where each map is valid within its own mask
    for lo, hi in zip(bins[:-1], bins[1:]):
        ring = (dist_in >= lo) & (dist_in < hi) & uni
        r1 = ring & m1 & np.isfinite(t1)
        r2 = ring & m2 & np.isfinite(t2)
        ri = ring & inter  # paired region

        n_r1 = np.count_nonzero(r1)
        n_r2 = np.count_nonzero(r2)
        n_ri = np.count_nonzero(ri)

        line = f"bin[{lo:g},{hi:g}): vox {n_r1:,}/{n_r2:,} (paired {n_ri:,})"
        if n_ri > 0:
            d = t2[ri] - t1[ri]
            line += f"  Δmean={float(np.mean(d)):.3f}  Δmed={float(np.median(d)):.3f}"
        print(line)

    # Coverage bias diagnostic: what fraction of union each direction covers in periphery
    print("\n=== COVERAGE FRACTION IN UNION (by bins) ===")
    for lo, hi in zip(bins[:-1], bins[1:]):
        ring = (dist_in >= lo) & (dist_in < hi) & uni
        nr = np.count_nonzero(ring)
        if nr == 0:
            continue
        frac1 = np.count_nonzero(ring & m1) / nr
        frac2 = np.count_nonzero(ring & m2) / nr
        print(f"bin[{lo:g},{hi:g}): union vox={nr:,}  coverage {args.label1}={frac1*100:.1f}%  {args.label2}={frac2*100:.1f}%")

if __name__ == "__main__":
    main()
