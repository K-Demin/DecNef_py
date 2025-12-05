#!/usr/bin/env python3
"""
compare_decoders.py

Usage:
    python compare_decoders.py \
      --matlab folder1 \
      --python folder2 \
      --decoder decoder.nii \
      --roi-matlab /path/to/ROI_NPS.txt \
      --roi-python /path/to/ROI_DECODER.txt \
      --matlab-mat /path/to/Collector_Curve_RT_*.mat

Both folders must contain single-volume NIfTI files,
named in alphabetical order (vol_00001.nii, vol_00002.nii, ...).
"""

import argparse
from pathlib import Path
import h5py

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.io import loadmat   # for reading gData *.mat


# ----------------- basic helpers ----------------- #

def load_sorted(folder: Path):
    files = sorted([f for f in folder.glob("*.nii*")])
    if not files:
        raise RuntimeError(f"No NIfTI files in {folder}")
    return files


def flatten(img_path: Path):
    img = nib.load(str(img_path))
    arr = np.asarray(img.dataobj, dtype=np.float32)
    return arr.ravel()


def dot(a, b):
    return float(np.dot(a, b))


def corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def zscore(arr):
    m = np.nanmean(arr)
    s = np.nanstd(arr)
    if s == 0:
        return np.zeros_like(arr)
    return (arr - m) / s


# ----------------- ROI text parsing (ATR style) ----------------- #

def load_roi_text(path: Path):
    """
    Parse ROI_NPS.txt / ROI_DECODER.txt like MATLAB get_roi_array_text.

    Returns
    -------
    idx : 1D int array
        Indices of ROI voxels (flattened).
    weights : 1D float array
        Length = n_vox + 1. First n_vox are voxel weights, last element is bias.
    template : 1D float array
        Template values for each ROI voxel.
    dims : tuple
        Original 3D dimensions.
    """
    with open(path, "r") as f:
        dim = list(map(int, f.readline().split()))
        dims = tuple(dim)
        vox_num = int(f.readline())

        mask = np.zeros(dims, dtype=bool)
        weight_img = np.zeros(dims, dtype=np.float32)
        template_img = np.zeros(dims, dtype=np.float32)

        # voxel lines
        for _ in range(vox_num):
            d1, d2, d3, w, t = f.readline().split()
            x = int(d1) - 1
            y = int(d2) - 1
            z = int(d3) - 1
            mask[x, y, z] = True
            weight_img[x, y, z] = float(w)
            template_img[x, y, z] = float(t)

        # last line: bias (intercept) in 4th column
        _d1, _d2, _d3, bias_str, _t = f.readline().split()
        bias = float(bias_str)

    flat_mask = mask.ravel()
    idx = np.where(flat_mask)[0]

    voxel_w = weight_img.ravel()[idx]
    weights = np.concatenate([voxel_w, np.array([bias], dtype=np.float32)])

    template = template_img.ravel()[idx]

    return idx, weights.astype(np.float32), template.astype(np.float32), dims


# ----------------- ATR-style decoder computation ----------------- #

def atr_compute_decoder(volumes_flat, idx, weights, template, baseline_len=20):
    """
    Approximate ATR-style decoder:

    - Extract ROI voxels.
    - Compute voxel-wise baseline mean/std from first baseline_len TRs.
    - Z-score each TR per voxel (like roi_baseline_mean/std logic).
    - Decoder value per TR = z_vector · voxel_weights + bias
    - corr_roi_template per TR = corr(z_vector, template)
    """
    # Stack TR × vox
    V = np.vstack([v[idx] for v in volumes_flat])  # shape: [TR, vox]

    n_tr, _ = V.shape

    # baseline stats
    bl = V[:min(baseline_len, n_tr), :]
    mu = bl.mean(axis=0)
    sigma = bl.std(axis=0)
    sigma[sigma == 0] = 1.0

    # voxelwise z-score
    Z = (V - mu) / sigma  # [TR, vox]

    voxel_w = weights[:-1]
    bias = weights[-1]

    # decoder timecourse
    dec = Z.dot(voxel_w) + bias

    # correlation with template
    corrT = np.empty(n_tr, dtype=np.float32)
    for t in range(n_tr):
        corrT[t] = corr(Z[t], template)

    return dec, corrT


# ----------------- MATLAB decoding_windows loader ----------------- #

def load_matlab_decoding_windows(mat_path: Path, start_index: int = 11):
    """
    Load gData.data.decoding_windows from a MATLAB v7.3 HDF5 .mat file.

    Handles both cases where 'gData' / 'data' are:
      - groups directly, or
      - datasets that contain object references.
    """
    import h5py

    def deref_to_group(h5file, obj):
        """
        Given an h5py Group or Dataset that encodes a MATLAB struct/object,
        return the underlying Group.

        - If obj is a Group: return as is.
        - If obj is a Dataset of references: take [0,0] and dereference.
        """
        if isinstance(obj, h5py.Group):
            return obj
        elif isinstance(obj, h5py.Dataset):
            # Typical MATLAB v7.3: 2D array of object refs
            ref = obj[0, 0]
            return h5file[ref]
        else:
            raise TypeError(f"Unexpected HDF5 type: {type(obj)}")

    with h5py.File(str(mat_path), "r") as f:
        # 1) Get gData as a group (direct or via ref)
        gData_obj = f["gData"]
        gData_grp = deref_to_group(f, gData_obj)

        # 2) Inside gData, get the 'data' struct
        data_obj = gData_grp["data"]
        data_grp = deref_to_group(f, data_obj)

        # 3) Find the decoding_windows-like field
        keys = list(data_grp.keys())
        target_key = None
        for k in keys:
            if "decoding" in k.lower():
                target_key = k
                break

        if target_key is None:
            raise KeyError(f"No decoding_windows-like field found. Keys: {keys}")

        dw_ds = data_grp[target_key]

        # 4) Get numeric array
        dw = np.array(dw_ds, dtype=np.float32).ravel()

    # MATLAB is 1-based indexing → drop the first start_index-1 elements
    if start_index > 1:
        dw = dw[start_index - 1:]

    return dw


# ----------------- main comparison ----------------- #

def main(args):
    f_mat = Path(args.matlab)
    f_pyt = Path(args.python)
    decoder_img = flatten(Path(args.decoder))

    mat_files = load_sorted(f_mat)
    pyt_files = load_sorted(f_pyt)

    n = min(len(mat_files), len(pyt_files))

    # results
    dot_mat = []
    dot_pyt = []
    corr_mat = []
    corr_pyt = []
    dot_diff = []
    corr_diff = []

    voxel_means_mat = []
    voxel_means_pyt = []
    voxel_stds_mat = []
    voxel_stds_pyt = []

    # pre-load volumes as flat arrays for ATR part later
    vols_mat_flat = []
    vols_pyt_flat = []

    # align: MATLAB vol i ↔ Python vol i+10 (Python starts at TR 11)
    for i in range(n - 10):
        a = flatten(mat_files[i])
        b = flatten(pyt_files[i + 10])

        vols_mat_flat.append(a)
        vols_pyt_flat.append(b)

        # dot products (decoder·volume)
        dm = dot(a, decoder_img)
        dp = dot(b, decoder_img)

        # correlations with decoder image
        cm = corr(a, decoder_img)
        cp = corr(b, decoder_img)

        dot_mat.append(dm)
        dot_pyt.append(dp)
        corr_mat.append(cm)
        corr_pyt.append(cp)

        dot_diff.append(dp - dm)
        corr_diff.append(cp - cm)

        voxel_means_mat.append(np.mean(a))
        voxel_means_pyt.append(np.mean(b))
        voxel_stds_mat.append(np.std(a))
        voxel_stds_pyt.append(np.std(b))

    dot_mat = np.asarray(dot_mat)
    dot_pyt = np.asarray(dot_pyt)
    corr_mat = np.asarray(corr_mat)
    corr_pyt = np.asarray(corr_pyt)

    # --------- print original global metrics --------- #
    print("===== GLOBAL METRICS (decoder.nii · volume) =====")
    print(f"Dot-product correlation:     {np.corrcoef(dot_mat, dot_pyt)[0, 1]:.4f}")
    print(f"Corr-with-template corr:     {np.corrcoef(corr_mat, corr_pyt)[0, 1]:.4f}")
    print(f"Dot-product MSE:             {np.mean((dot_mat - dot_pyt) ** 2):.4f}")
    print(f"Corr-with-template MSE:      {np.mean((corr_mat - corr_pyt) ** 2):.4f}")
    print(f"Mean dot diff (pyt - mat):   {np.mean(dot_pyt - dot_mat):.4f}")
    mean_corr_diff = np.mean(corr_pyt - corr_mat)
    print(f"Mean corr diff (pyt - mat):  {mean_corr_diff:.4f}")
    print()

    # --------- ATR-style decoder using ROI text files --------- #
    dec_mat = dec_pyt = corrT_mat = corrT_pyt = None

    if args.roi_matlab and args.roi_python:
        roi_mat_path = Path(args.roi_matlab)
        roi_pyt_path = Path(args.roi_python)

        idx_mat, w_mat, tmpl_mat, _ = load_roi_text(roi_mat_path)
        idx_pyt, w_pyt, tmpl_pyt, _ = load_roi_text(roi_pyt_path)

        dec_mat, corrT_mat = atr_compute_decoder(
            vols_mat_flat, idx_mat, w_mat, tmpl_mat
        )
        dec_pyt, corrT_pyt = atr_compute_decoder(
            vols_pyt_flat, idx_pyt, w_pyt, tmpl_pyt
        )

        print("===== ATR-style DECODER METRICS (ROI txt) =====")
        print(f"Decoder corr (Python vs MATLAB):          {np.corrcoef(dec_mat, dec_pyt)[0, 1]:.4f}")
        print(f"Template-corr corr (Python vs MATLAB):    {np.corrcoef(corrT_mat, corrT_pyt)[0, 1]:.4f}")
        print()
    else:
        print("No ROI txt paths provided → skipping ATR-style decoder comparison.\n")

    # --------- NEW: compare with MATLAB decoding_windows from .mat --------- #
    # --------- NEW: compare with MATLAB decoding_windows from .mat --------- #
    # --------- NEW: compare with MATLAB decoding_windows from .mat --------- #
    if args.matlab_mat and (dec_mat is not None) and (dec_pyt is not None):
        mat_dec_path = Path(args.matlab_mat)
        dw = load_matlab_decoding_windows(mat_dec_path, start_index=11)

        L = min(len(dw), len(dec_mat), len(dec_pyt))
        dw = dw[:L]
        d_mat = dec_mat[:L]
        d_pyt = dec_pyt[:L]

        # mask out invalid TRs: MATLAB decoding_windows == 0
        valid_mask = dw != 0
        dw_valid = dw[valid_mask]
        d_mat_valid = d_mat[valid_mask]
        d_pyt_valid = d_pyt[valid_mask]

        print("===== COMPARISON WITH MATLAB gData.data.decoding_windows =====")
        print(f"Total TRs: {L},  Valid (dw!=0): {np.sum(valid_mask)}")
        print(f"MATLAB indices: 11..{10 + L}")
        print()

        # --------- numeric metrics --------- #
        # raw correlations on valid samples only
        r_mat = corr(dw_valid, d_mat_valid)
        r_pyt = corr(dw_valid, d_pyt_valid)
        print(f"Corr(decoding_windows, ATR MATLAB ROI) [valid]:   {r_mat:.4f}")
        print(f"Corr(decoding_windows, ATR Python ROI) [valid]:   {r_pyt:.4f}")

        # z-scored correlations on valid samples only
        z_dw = zscore(dw_valid)
        z_d_mat = zscore(d_mat_valid)
        z_d_pyt = zscore(d_pyt_valid)
        rz_mat = corr(z_dw, z_d_mat)
        rz_pyt = corr(z_dw, z_d_pyt)
        print("Z-scored correlations (valid only):")
        print(f"  z(dw) vs z(ATR MATLAB ROI):                    {rz_mat:.4f}")
        print(f"  z(dw) vs z(ATR Python ROI):                    {rz_pyt:.4f}")
        print()

        # --------- plots --------- #
        fig3, ax3 = plt.subplots(2, 2, figsize=(14, 10))

        # (0,0) raw timecourses, keep full series
        ax3[0, 0].plot(dw, label="MATLAB decoding_windows (full)")
        ax3[0, 0].plot(d_mat, label="ATR MATLAB ROI")
        ax3[0, 0].plot(d_pyt, label="ATR Python ROI")
        ax3[0, 0].set_title("Raw decoder timecourses")
        ax3[0, 0].set_xlabel("TR")
        ax3[0, 0].set_ylabel("Decoder value")
        ax3[0, 0].legend()

        # (0,1) z-scored VALID timecourses only
        ax3[0, 1].plot(z_dw, label="z(dw) [valid]")
        ax3[0, 1].plot(z_d_mat, label="z(ATR MATLAB ROI) [valid]")
        ax3[0, 1].plot(z_d_pyt, label="z(ATR Python ROI) [valid]")
        ax3[0, 1].set_title("Z-scored decoder timecourses (valid only)")
        ax3[0, 1].set_xlabel("Valid TR index")
        ax3[0, 1].set_ylabel("z-score")
        ax3[0, 1].legend()

        # (1,0) scatter: dw vs ATR MATLAB / Python (valid only)
        ax3[1, 0].scatter(dw_valid, d_mat_valid, s=10, label="MATLAB ROI")
        ax3[1, 0].scatter(dw_valid, d_pyt_valid, s=10, label="Python ROI", alpha=0.7)
        ax3[1, 0].set_title("decoding_windows vs ATR decoders (valid only)")
        ax3[1, 0].set_xlabel("decoding_windows")
        ax3[1, 0].set_ylabel("ATR decoder")
        ax3[1, 0].legend()

        # (1,1) differences (valid only)
        ax3[1, 1].plot(dw_valid - d_mat_valid, label="dw - ATR MATLAB")
        ax3[1, 1].plot(dw_valid - d_pyt_valid, label="dw - ATR Python")
        ax3[1, 1].set_title("Differences to decoding_windows (valid only)")
        ax3[1, 1].set_xlabel("Valid TR index")
        ax3[1, 1].set_ylabel("Difference")
        ax3[1, 1].legend()

        plt.tight_layout()
        out3 = Path("decoder_dw_comparison_valid.png")
        plt.savefig(out3, dpi=200)
        print(f"Saved decoding_windows comparison figure (valid only) → {out3}")



    # --------- plots (simple metrics; ATR plots unchanged) --------- #
    fig, ax = plt.subplots(3, 2, figsize=(14, 16))

    # TR-by-TR dot product timecourses
    ax[0, 0].plot(dot_mat, label="MATLAB", color="blue")
    ax[0, 0].plot(dot_pyt, label="Python", color="orange")
    ax[0, 0].set_title("Dot Product (decoder.nii · volume)")
    ax[0, 0].legend()

    # TR-by-TR correlation timecourses
    ax[0, 1].plot(corr_mat, label="MATLAB", color="blue")
    ax[0, 1].plot(corr_pyt, label="Python", color="orange")
    ax[0, 1].set_title("Correlation with decoder.nii")
    ax[0, 1].legend()

    # Scatter: dot products
    ax[1, 0].scatter(dot_mat, dot_pyt, s=10)
    ax[1, 0].set_title("Dot Products: MATLAB vs Python")
    ax[1, 0].set_xlabel("MATLAB")
    ax[1, 0].set_ylabel("Python")

    # Scatter: correlations
    ax[1, 1].scatter(corr_mat, corr_pyt, s=10)
    ax[1, 1].set_title("Template Correlation: MATLAB vs Python")
    ax[1, 1].set_xlabel("MATLAB")
    ax[1, 1].set_ylabel("Python")

    # Differences
    ax[2, 0].plot(dot_diff)
    ax[2, 0].set_title("Dot Product Difference (Python - MATLAB)")

    ax[2, 1].plot(corr_diff)
    ax[2, 1].set_title("Correlation Difference (Python - MATLAB)")

    plt.tight_layout()
    out = Path("decoder_comparison.png")
    plt.savefig(out, dpi=200)
    print(f"Saved figure → {out}")

    # Optional: ATR-style decoder traces as a separate figure
    if dec_mat is not None and dec_pyt is not None:
        fig2, ax2 = plt.subplots(2, 1, figsize=(12, 8))
        ax2[0].plot(dec_mat, label="MATLAB ROI txt", color="blue")
        ax2[0].plot(dec_pyt, label="Python ROI txt", color="orange")
        ax2[0].set_title("ATR-style decoder timecourse")
        ax2[0].legend()

        ax2[1].plot(corrT_mat, label="MATLAB ROI txt", color="blue")
        ax2[1].plot(corrT_pyt, label="Python ROI txt", color="orange")
        ax2[1].set_title("ATR-style corr_roi_template timecourse")
        ax2[1].legend()

        plt.tight_layout()
        out2 = Path("decoder_ATR_style.png")
        plt.savefig(out2, dpi=200)
        print(f"Saved ATR-style figure → {out2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matlab", required=True,
                        help="Folder with MATLAB-processed volumes")
    parser.add_argument("--python", required=True,
                        help="Folder with Python-processed volumes")
    parser.add_argument("--decoder", required=True,
                        help="Decoder/template NIfTI (rweights_*.nii)")
    parser.add_argument("--roi-matlab", required=False,
                        help="ROI txt for MATLAB pipeline (e.g., ROI_NPS.txt)")
    parser.add_argument("--roi-python", required=False,
                        help="ROI txt for Python pipeline (e.g., ROI_DECODER.txt)")
    parser.add_argument("--matlab-mat", required=False,
                        help="MATLAB .mat file with gData.data.decoding_windows")
    args = parser.parse_args()
    main(args)
