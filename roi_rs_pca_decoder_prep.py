"""
roi_rs_pca_decoder_prep.py

HOW TO USE

# python roi_rs_pca_decoder_prep.py -subj 00085 -day 2 -run 7
----------
1) Keep the USER SETTINGS block below as-is unless you want to tweak ROI/KVOX/etc.
2) Run the script by specifying subject, day, and run IDs (data are under ./data):
       python roi_rs_pca_decoder_prep.py -subj 00085 -day 3 -run 1
3) Inputs are discovered under ./data/sub-<subj>/<day>/func/<run>/ and outputs
   are written to ./data/sub-<subj>/<day>/PCA/<run>/<ROI>/.

This script auto-discovers resting-state (RS) fMRI inputs, ROI masks in EPI space,
optional FD vectors, and optional brain/GM masks. It computes voxelwise tSNR,
selects top voxels per ROI, runs PCA (preferring denoised RS), and writes
"decoder-ready" artifacts including voxel indices, weights, component maps,
and metadata. A helper function `score_components` is provided for applying the
weights to new EPI volumes.
"""

from __future__ import annotations

# ----------------------
# USER SETTINGS
# ----------------------
ROI_NAMES = ["EVC", "LPFC", "Sensorimotor"]
KVOX = 2000
N_COMPONENTS = 10
FD_THRESH = 0.2
MIN_NEIGHBORS = 0  # 0 disables neighbor filtering
USE_ZSCORE = True
SEED = 0


AUTO_FSLMERGE_4D = True
FSLMERGE_OUTDIR_NAME = "analysis"  # where to write the merged 4D under run_dir
PREFER_MERGED_BASENAME = "rs_4d"   # filename stem: rs_4d_mc.nii.gz / rs_4d_reg.nii.gz
USE_MOTION_QC = True              # load motion_rt.1D and compute PC1 motion corr

# ----------------------

import csv
import json
import argparse
from dataclasses import dataclass
from importlib import util
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import re
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt

@dataclass
class DiscoveredInputs:
    pca_rs: Path
    tsnr_rs: Path
    fd: Optional[np.ndarray]
    brain_mask: Optional[Path]
    gm_mask: Optional[Path]
    roi_masks: dict


@dataclass
class QCRow:
    roi: str
    vox_orig: int
    vox_after_masks: int
    vox_selected: int
    t_all: int
    t_used: int
    tsnr_mean: float
    tsnr_median: float
    pc1_fd_corr: Optional[float]


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def score_components(volume_3d: np.ndarray, voxel_indices: np.ndarray, weights: np.ndarray, normalize: Optional[str] = None, norm_mean: Optional[np.ndarray] = None, norm_std: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply decoder weights to a single 3D EPI volume.

    Parameters
    ----------
    volume_3d : np.ndarray
        3D volume in the same space as the decoder (x, y, z).
    voxel_indices : np.ndarray
        Flat indices (np.ravel order) of voxels used by the decoder.
    weights : np.ndarray
        Array of shape (n_components, n_voxels) containing component weights.
    normalize : str | None
        None (default): use raw voxel values.
        "z": z-score using provided norm_mean and norm_std before scoring.
    norm_mean : np.ndarray | None
        Mean per voxel from training data (needed if normalize == "z").
    norm_std : np.ndarray | None
        Std per voxel from training data (needed if normalize == "z").

    Returns
    -------
    scores : np.ndarray
        Component scores of shape (n_components,).
    """

    flat = volume_3d.reshape(-1).astype(np.float32)
    x = flat[voxel_indices]

    if normalize == "z":
        if norm_mean is None or norm_std is None:
            raise ValueError("norm_mean and norm_std required for z-normalization")
        safe_std = np.where(norm_std == 0, 1.0, norm_std)
        x = (x - norm_mean) / safe_std

    return weights @ x


def _list_imaging_files(data_dir: Path) -> List[Path]:
    exts = (".nii", ".nii.gz", ".mgz")
    return [p for p in data_dir.rglob("*") if p.is_file() and any(p.name.lower().endswith(ext) for ext in exts)]


def _is_rest_name(name: str) -> bool:
    low = name.lower()
    keywords = ["rest", "rs", "bold", "func", "task-rest", "resting"]
    return any(k in low for k in keywords)


def _rs_priority_for_pca(path: Path) -> Tuple[int, int, int]:
    low = path.name.lower()
    parent_low = path.parent.name.lower()
    denoise_keys = ["denoise", "resid", "clean", "regress", "regressed", "nuis", "filtered", "noise"]
    mc_keys = ["mc", "motioncorr", "distcorr", "preproc"]
    has_denoise = any(k in low for k in denoise_keys) or parent_low == "reg"
    has_mc = any(k in low for k in mc_keys) or parent_low == "mc"
    # Prefer reg/denoised first, then mc, then everything else
    dir_score = 0 if parent_low == "reg" else (1 if parent_low == "mc" else 2)
    return (dir_score, 0 if has_denoise else 1, 0 if has_mc else 1)


def _rs_priority_for_tsnr(path: Path) -> Tuple[int, int, int]:
    low = path.name.lower()
    parent_low = path.parent.name.lower()
    pre_keys = ["mc", "distcorr", "stc", "preproc", "raw"]
    denoise_keys = ["denoise", "resid", "clean", "regress", "regressed", "nuis", "filtered", "noise"]
    has_pre = any(k in low for k in pre_keys) or parent_low == "mc"
    has_denoise = any(k in low for k in denoise_keys) or parent_low == "reg"
    # Prefer mc/preproc for tSNR, avoid denoised when possible
    dir_score = 0 if parent_low == "mc" else (1 if parent_low == "reg" else 2)
    return (dir_score, 0 if has_pre else 1, 1 if has_denoise else 0)


def _load_nifti(path: Path) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    img = nib.load(str(path))
    data = np.asanyarray(img.get_fdata(), dtype=np.float32)
    return data, img.affine, img.header


def _discover_rs_inputs(data_dir: Path) -> Tuple[Path, Path]:
    # ---- 1) Original logic: find 4D candidates anywhere under run_dir ----
    candidates: List[Tuple[Path, nib.Nifti1Header]] = []
    priority_dirs = {"reg", "mc"}

    for path in _list_imaging_files(data_dir):
        parent_low = path.parent.name.lower()
        is_reg_or_mc = parent_low in priority_dirs or any(part.lower() in priority_dirs for part in path.parts[-3:])
        if not (_is_rest_name(path.name) or is_reg_or_mc):
            continue
        try:
            hdr = nib.load(str(path)).header
        except Exception:
            continue
        if len(hdr.get_data_shape()) != 4:
            continue
        candidates.append((path, hdr))

    if candidates:
        pca_sorted = sorted(candidates, key=lambda x: _rs_priority_for_pca(x[0]))
        tsnr_sorted = sorted(candidates, key=lambda x: _rs_priority_for_tsnr(x[0]))
        pca_rs = pca_sorted[0][0]
        tsnr_rs = tsnr_sorted[0][0] if tsnr_sorted else pca_rs
        return pca_rs, tsnr_rs

    # ---- 2) Fallback: build 4D from 3D-per-TR in mc/ or reg/ ----
    if not AUTO_FSLMERGE_4D:
        raise FileNotFoundError(f"No resting-state 4D NIfTI found under {data_dir} (and AUTO_FSLMERGE_4D is False)")

    mc_dir = data_dir / "mc"
    reg_dir = data_dir / "reg"

    # Prefer reg (denoised) for PCA, but tSNR wants mc if available.
    reg_vols = _find_3d_series(reg_dir) if reg_dir.exists() else []
    mc_vols = _find_3d_series(mc_dir) if mc_dir.exists() else []

    if not reg_vols and not mc_vols:
        raise FileNotFoundError(f"No resting-state NIfTI found under {data_dir} (no 4D files; no 3D series in mc/ or reg/)")

    out_dir = data_dir / FSLMERGE_OUTDIR_NAME
    merged_reg = out_dir / f"{PREFER_MERGED_BASENAME}_reg.nii.gz"
    merged_mc  = out_dir / f"{PREFER_MERGED_BASENAME}_mc.nii.gz"

    # Merge what exists
    if reg_vols and (not merged_reg.exists()):
        print(f"Building 4D from reg/ ({len(reg_vols)} vols) -> {merged_reg}")
        ok = _run_fslmerge(reg_vols, merged_reg)
        if not ok:
            raise RuntimeError("Failed to fslmerge reg/ 3D volumes into 4D.")
    if mc_vols and (not merged_mc.exists()):
        print(f"Building 4D from mc/ ({len(mc_vols)} vols) -> {merged_mc}")
        ok = _run_fslmerge(mc_vols, merged_mc)
        if not ok:
            raise RuntimeError("Failed to fslmerge mc/ 3D volumes into 4D.")

    # Decide PCA vs tSNR inputs
    # PCA: prefer reg merged if exists; otherwise mc merged
    pca_rs = merged_reg if merged_reg.exists() else merged_mc
    # tSNR: prefer mc merged if exists; otherwise reg merged
    tsnr_rs = merged_mc if merged_mc.exists() else merged_reg

    # sanity check they are 4D
    for p in [pca_rs, tsnr_rs]:
        hdr = nib.load(str(p)).header
        if len(hdr.get_data_shape()) != 4:
            raise RuntimeError(f"Merged file is not 4D: {p} shape={hdr.get_data_shape()}")

    return pca_rs, tsnr_rs



def _discover_mask(paths: Iterable[Path], keywords: Sequence[str]) -> Optional[Path]:
    best = None
    best_score = (10, 10)
    for p in paths:
        low = p.name.lower()
        if not any(k in low for k in keywords):
            continue
        folder = p.parent.name.lower()
        folder_score = 0 if any(k in folder for k in ["roi", "mask", "epi", "func"]) else 1
        name_score = 0 if any(k in low for k in ["roi", "mask"]) else 1
        score = (folder_score, name_score)
        if score < best_score:
            best = p
            best_score = score
    return best


def _discover_roi_masks(search_dirs: Sequence[Path], roi_names: Sequence[str]) -> dict:
    roi_masks = {}
    all_paths: List[Path] = []
    for d in search_dirs:
        if d.exists():
            all_paths.extend(_list_imaging_files(d))

    for roi in roi_names:
        roi_low = roi.lower()
        matches = []
        for p in all_paths:
            name_low = p.name.lower()
            if roi_low in name_low and any(key in name_low for key in ["roi", "mask"]):
                matches.append(p)
        if not matches:
            # allow looser match
            for p in all_paths:
                if roi_low in p.name.lower():
                    matches.append(p)
        if matches:
            chosen = _discover_mask(matches, [roi_low])
            roi_masks[roi] = chosen or matches[0]
    return roi_masks


def _discover_optional_mask(search_dirs: Sequence[Path], name_keywords: Sequence[str]) -> Optional[Path]:
    all_paths: List[Path] = []
    for d in search_dirs:
        if d.exists():
            all_paths.extend(_list_imaging_files(d))
    return _discover_mask(all_paths, name_keywords)


def _discover_fd_vector(data_dir: Path, t_expected: int) -> Optional[np.ndarray]:
    fdt = data_dir / "fd_rt.csv"
    if not fdt.exists():
        print("No fd_rt.csv found in run folder; FD censoring disabled.")
        return None

    try:
        # Try DictReader first (header-aware)
        with open(fdt, "r", newline="") as f:
            sample = f.read(2048)
            f.seek(0)
            has_header = any(h in sample.lower() for h in ["fd", "framewise"]) and ("," in sample or "\t" in sample)

            if has_header:
                reader = csv.DictReader(f)
                vals = []
                for row in reader:
                    if "fd" in row and row["fd"] != "":
                        vals.append(float(row["fd"]))
                    else:
                        # fallback: second column if fd missing
                        items = list(row.values())
                        if len(items) >= 2 and items[1] != "":
                            vals.append(float(items[1]))
                fd = np.array(vals, dtype=np.float32)
            else:
                # No reliable header: read as plain CSV and take 2nd column
                f.seek(0)
                raw = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 2:
                        continue
                    # skip non-numeric header-ish lines
                    try:
                        raw.append(float(parts[1]))
                    except ValueError:
                        continue
                fd = np.array(raw, dtype=np.float32)

        fd = np.atleast_1d(fd).reshape(-1)

        # Handle common FD length conventions
        if fd.size == t_expected - 1:
            print(f"FD length is T-1 ({fd.size}); prepending 0 to match T={t_expected}.")
            fd = np.concatenate([[0.0], fd]).astype(np.float32)

        if fd.size != t_expected:
            print(f"Warning: FD length mismatch in fd_rt.csv (found {fd.size}, expected {t_expected}); ignoring FD.")
            return None

        print(f"Found FD in: {fdt} (length {fd.size})")
        return fd

    except Exception as e:
        print(f"Warning: failed to read FD from {fdt}: {e}")
        return None



def _affine_close(a1: np.ndarray, a2: np.ndarray, tol: float = 1e-3) -> bool:
    return np.allclose(a1, a2, atol=tol)


def _count_neighbors(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.int8), 1, mode="constant", constant_values=0)
    counts = np.zeros_like(mask, dtype=np.int16)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                counts += padded[
                    1 + dx : 1 + dx + mask.shape[0],
                    1 + dy : 1 + dy + mask.shape[1],
                    1 + dz : 1 + dz + mask.shape[2],
                ]
    return counts


def _save_nifti_like(reference: nib.Nifti1Image, data: np.ndarray, path: Path):
    img = nib.Nifti1Image(data.astype(np.float32), reference.affine, reference.header)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))


def _compute_pca(data: np.ndarray, n_components: int, random_state: int, use_sklearn: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    data: (T, V)
    Returns components (n_components, V), explained_ratio (n_components,), timecourses (T, n_components)
    """
    n_components = min(n_components, data.shape[0], data.shape[1])
    if n_components == 0:
        return np.empty((0, data.shape[1]), dtype=np.float32), np.empty(0, dtype=np.float32), np.empty((data.shape[0], 0), dtype=np.float32)

    if use_sklearn:
        from sklearn.decomposition import PCA  # type: ignore

        pca = PCA(n_components=n_components, svd_solver="auto", random_state=random_state)
        timecourses = pca.fit_transform(data)
        components = pca.components_
        explained = pca.explained_variance_ratio_
    else:
        # Center data
        data_centered = data - np.mean(data, axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(data_centered, full_matrices=False)
        components = vt[:n_components]
        s = s[:n_components]
        u = u[:, :n_components]
        total_var = np.sum(data_centered ** 2) / (data_centered.shape[0] - 1)
        explained_var = (s ** 2) / (data_centered.shape[0] - 1)
        explained = explained_var / total_var
        timecourses = u * s

    return (
        components.astype(np.float32),
        explained.astype(np.float32),
        timecourses.astype(np.float32),
    )


def _prepare_roi(
    roi: str,
    roi_mask_path: Path,
    tsnr_map: np.ndarray,
    reference_img: nib.Nifti1Image,
    brain_mask: Optional[np.ndarray],
    gm_mask: Optional[np.ndarray],
    output_dir: Path,
    run_dir: Path,  # <-- add this new argument
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    roi_mask, roi_affine = _load_mask_in_epi_space(
        roi=roi,
        roi_mask_path=roi_mask_path,
        tsnr_map_3d=tsnr_map,
        tsnr_img_3d=reference_img,
        run_dir=run_dir,
        roi_out_dir=output_dir,
    )
    if roi_mask is None:
        raise ValueError(f"ROI {roi} could not be loaded/warped into EPI space: {roi_mask_path}")

    if roi_affine is not None and not _affine_close(roi_affine, reference_img.affine):
        print(f"  Warning: ROI {roi} affine differs from RS affine; proceeding with caution.")

    _save_nifti_like(reference_img, roi_mask.astype(np.float32), output_dir / "roi_original_mask.nii.gz")

    combined_mask = roi_mask.copy()
    if brain_mask is not None:
        combined_mask &= brain_mask
    if gm_mask is not None:
        combined_mask &= gm_mask

    vox_after_masks = int(np.sum(combined_mask))

    tsnr_in_roi = tsnr_map * combined_mask
    _save_nifti_like(reference_img, tsnr_in_roi, output_dir / "roi_tsnr_in_roi.nii.gz")

    selected_mask = np.zeros_like(combined_mask, dtype=bool)
    if vox_after_masks > 0:
        tsnr_vals = tsnr_map[combined_mask]
        if KVOX > 0 and tsnr_vals.size > KVOX:
            top_idx = np.argpartition(-tsnr_vals, KVOX - 1)[:KVOX]
            selected_flat_indices = np.flatnonzero(combined_mask)[top_idx]
            selected_mask_flat = np.zeros_like(combined_mask.reshape(-1), dtype=bool)
            selected_mask_flat[selected_flat_indices] = True
            selected_mask = selected_mask_flat.reshape(combined_mask.shape)
        else:
            selected_mask = combined_mask.copy()

        if MIN_NEIGHBORS > 0:
            neighbor_counts = _count_neighbors(selected_mask)
            selected_mask &= neighbor_counts >= MIN_NEIGHBORS

    _save_nifti_like(reference_img, selected_mask.astype(np.float32), output_dir / "roi_selected_mask.nii.gz")

    tsnr_selected = tsnr_map[selected_mask]
    return selected_mask, tsnr_selected, tsnr_in_roi, roi_mask, combined_mask


def _load_optional_mask(mask_path: Optional[Path]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if mask_path is None:
        return None, None
    data, affine, _ = _load_nifti(mask_path)
    return data > 0, affine


def _corr_pc1_fd(pc1: np.ndarray, fd: np.ndarray) -> Optional[float]:
    if pc1.size != fd.size:
        return None
    if pc1.size == 0:
        return None
    if np.std(pc1) == 0 or np.std(fd) == 0:
        return None
    return float(np.corrcoef(pc1, fd)[0, 1])


def process_dataset(
    run_dir: Path,
    roi_names: Sequence[str],
    roi_search_dirs: Optional[Sequence[Path]] = None,
    mask_search_dirs: Optional[Sequence[Path]] = None,
    out_root: Optional[Path] = None,
) -> None:
    print(f"\nProcessing dataset: {run_dir}")
    pca_rs_path, tsnr_rs_path = _discover_rs_inputs(run_dir)
    tsnr_data, tsnr_affine, tsnr_hdr = _load_nifti(tsnr_rs_path)
    if len(tsnr_data.shape) != 4:
        raise ValueError(f"tSNR input is not 4D: {tsnr_rs_path}")
    t_all = tsnr_data.shape[3]

    tsnr_map = np.divide(
        np.mean(tsnr_data, axis=3),
        np.where(np.std(tsnr_data, axis=3) == 0, np.inf, np.std(tsnr_data, axis=3)),
    )
    tsnr_map = np.nan_to_num(tsnr_map, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    tsnr_img = nib.Nifti1Image(tsnr_map, tsnr_affine, tsnr_hdr)

    pca_data, pca_affine, pca_hdr = _load_nifti(pca_rs_path)
    if pca_data.shape[:3] != tsnr_map.shape:
        raise ValueError("Spatial dimensions of PCA RS and tSNR RS differ")
    if not _affine_close(pca_affine, tsnr_affine):
        print("Warning: PCA RS affine differs from tSNR RS affine; proceeding with PCA input affine.")

    fd_vec = _discover_fd_vector(run_dir, t_all)
    motion_mat = _load_motion_1d(run_dir, t_all) if USE_MOTION_QC else None

    roi_dirs = list(roi_search_dirs or [run_dir])
    mask_dirs = list(mask_search_dirs or roi_dirs)

    brain_mask_path = _discover_optional_mask(mask_dirs, ["brainmask", "mask_brain", "brain_mask"])
    gm_mask_path = _discover_optional_mask(mask_dirs, ["gm_mask", "gmmask", "ribbon", "gm"])

    roi_masks = _discover_roi_masks(roi_dirs, roi_names)
    if not roi_masks:
        print("No ROI masks found; skipping dataset.")
        return

    out_root = out_root or run_dir / "PCA"
    out_root.mkdir(parents=True, exist_ok=True)
    _save_nifti_like(tsnr_img, tsnr_map, out_root / "tsnr_full.nii.gz")

    brain_mask, brain_affine = _load_optional_mask(brain_mask_path)
    gm_mask, gm_affine = _load_optional_mask(gm_mask_path)

    if brain_mask is not None:
        if brain_mask.shape != tsnr_map.shape:
            print("Warning: brain mask shape mismatch; ignoring brain mask.")
            brain_mask = None
        elif brain_affine is not None and not _affine_close(brain_affine, tsnr_affine):
            print("Warning: brain mask affine mismatch; ignoring brain mask.")
            brain_mask = None

    if gm_mask is not None:
        if gm_mask.shape != tsnr_map.shape:
            print("Warning: GM mask shape mismatch; ignoring GM mask.")
            gm_mask = None
        elif gm_affine is not None and not _affine_close(gm_affine, tsnr_affine):
            print("Warning: GM mask affine mismatch; ignoring GM mask.")
            gm_mask = None

    qc_rows: List[QCRow] = []
    use_sklearn = util.find_spec("sklearn") is not None

    for roi in roi_names:
        if roi not in roi_masks:
            print(f"ROI {roi} not found; skipping.")
            continue
        roi_path = roi_masks[roi]
        roi_dir = out_root / roi
        roi_dir.mkdir(exist_ok=True)

        print(f"  ROI: {roi} | mask: {roi_path}")
        selected_mask, tsnr_selected, tsnr_in_roi, roi_original_mask, combined_mask = _prepare_roi(
            roi,
            roi_path,
            tsnr_map,
            tsnr_img,
            brain_mask,
            gm_mask,
            roi_dir,
            run_dir,
        )

        vox_orig = int(np.sum(roi_original_mask))
        vox_after_masks = int(np.sum(combined_mask))
        vox_selected = int(np.sum(selected_mask))

        if vox_selected < 50:
            print(f"  Warning: ROI {roi} has only {vox_selected} voxels; skipping PCA.")
            qc_rows.append(
                QCRow(
                    roi,
                    vox_orig,
                    vox_after_masks,
                    vox_selected,
                    t_all,
                    0,
                    float(np.mean(tsnr_selected)) if tsnr_selected.size else 0.0,
                    float(np.median(tsnr_selected)) if tsnr_selected.size else 0.0,
                    None,
                )
            )
            continue

        flat_indices = np.flatnonzero(selected_mask)
        data_2d = pca_data.reshape(-1, pca_data.shape[3]).astype(np.float32)
        roi_ts = data_2d[flat_indices].T  # (T, V)

        if fd_vec is not None and FD_THRESH is not None:
            keep = fd_vec <= FD_THRESH
            if np.sum(keep) < 30:
                print(f"  Warning: FD censoring leaves only {np.sum(keep)} TRs; skipping PCA.")
                t_used = int(np.sum(keep))
                qc_rows.append(
                    QCRow(
                        roi,
                        vox_orig,
                        vox_after_masks,
                        vox_selected,
                        t_all,
                        t_used,
                        float(np.mean(tsnr_selected)) if tsnr_selected.size else 0.0,
                        float(np.median(tsnr_selected)) if tsnr_selected.size else 0.0,
                        None,
                    )
                )
                continue
            roi_ts = roi_ts[keep, :]
            fd_used = fd_vec[keep]
        else:
            fd_used = None

        t_used = roi_ts.shape[0]

        if USE_ZSCORE:
            mean_vox = np.mean(roi_ts, axis=0)
            std_vox = np.std(roi_ts, axis=0)
            std_safe = np.where(std_vox == 0, 1.0, std_vox)
            roi_ts_proc = (roi_ts - mean_vox) / std_safe
        else:
            mean_vox = np.mean(roi_ts, axis=0)
            std_vox = np.std(roi_ts, axis=0)
            roi_ts_proc = roi_ts - mean_vox
            std_safe = np.where(std_vox == 0, 1.0, std_vox)

        components, explained, timecourses = _compute_pca(
            roi_ts_proc,
            N_COMPONENTS,
            SEED,
            use_sklearn,
        )

        if timecourses.shape[0] == 0:
            print(f"  Warning: PCA failed for ROI {roi}; skipping outputs.")
            continue

        pc_corr = _corr_pc1_fd(timecourses[:, 0], fd_used) if fd_used is not None else None
        motion_qc = _pc1_motion_qc(timecourses[:, 0], motion_mat[keep, :] if (
                    motion_mat is not None and fd_used is not None) else motion_mat) if USE_MOTION_QC else None

        # Save outputs
        np.save(roi_dir / "pca_explained.npy", explained)
        np.save(roi_dir / "pca_timecourses.npy", timecourses)
        np.save(roi_dir / "decoder_voxel_indices.npy", flat_indices)
        np.save(roi_dir / "decoder_weights.npy", components)
        np.save(roi_dir / "decoder_norm_mean.npy", mean_vox.astype(np.float32))
        np.save(roi_dir / "decoder_norm_std.npy", std_safe.astype(np.float32))

        reference_img = tsnr_img
        for i in range(components.shape[0]):
            comp_map = np.zeros(selected_mask.shape, dtype=np.float32)
            comp_map[selected_mask] = components[i]
            comp_path = roi_dir / f"PC{i+1:02d}.nii.gz"
            _save_nifti_like(reference_img, comp_map, comp_path)

        metadata = {
            "roi": roi,
            "pca_rs": str(pca_rs_path),
            "tsnr_rs": str(tsnr_rs_path),
            "roi_mask": str(roi_path),
            "brain_mask": str(brain_mask_path) if brain_mask_path else None,
            "gm_mask": str(gm_mask_path) if gm_mask_path else None,
            "kvox": KVOX,
            "n_components": int(components.shape[0]),
            "fd_thresh": FD_THRESH,
            "use_zscore": USE_ZSCORE,
            "min_neighbors": MIN_NEIGHBORS,
            "voxel_count": vox_selected,
            "t_original": t_all,
            "t_used": t_used,
            "motion_file": str(run_dir / "motion_rt.1D") if (run_dir / "motion_rt.1D").exists() else None,
            "pc1_motion_corr": motion_qc,
            "decoder_ready": "scores = weights @ voxel_values (optionally z-scored)",
        }
        with open(roi_dir / "decoder_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        qc_rows.append(
            QCRow(
                roi,
                vox_orig,
                vox_after_masks,
                vox_selected,
                t_all,
                t_used,
                float(np.mean(tsnr_selected)),
                float(np.median(tsnr_selected)),
                pc_corr,
            )
        )

    # Write QC CSV
    qc_path = out_root / "qc_summary.csv"
    with open(qc_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "ROI",
            "voxels_original",
            "voxels_after_masks",
            "voxels_selected",
            "T_original",
            "T_used_for_PCA",
            "tSNR_mean_selected",
            "tSNR_median_selected",
            "corr_PC1_FD",
        ])
        for row in qc_rows:
            writer.writerow([
                row.roi,
                row.vox_orig,
                row.vox_after_masks,
                row.vox_selected,
                row.t_all,
                row.t_used,
                f"{row.tsnr_mean:.4f}",
                f"{row.tsnr_median:.4f}",
                "" if row.pc1_fd_corr is None else f"{row.pc1_fd_corr:.4f}",
            ])
    print(f"QC summary saved to {qc_path}")


def _build_run_dir(base_data: Path, subj: str, day: str, run: str) -> Path:
    subj_dir = base_data / f"sub-{subj}"
    day_dir = subj_dir / day
    func_dir = day_dir / "func"
    # Allow run id with or without prefix (e.g., "1" or "run-01")
    candidate_runs = [func_dir / run]
    if not run.startswith("run-"):
        candidate_runs.append(func_dir / f"run-{run}")
    for c in candidate_runs:
        if c.exists():
            return c
    return candidate_runs[0]


def _load_motion_1d(run_dir: Path, t_expected: int) -> Optional[np.ndarray]:
    f = run_dir / "motion_rt.1D"
    if not f.exists():
        return None
    try:
        arr = np.loadtxt(f, dtype=np.float32)
        arr = np.atleast_2d(arr)
        if arr.shape[0] != t_expected and arr.shape[1] == t_expected:
            arr = arr.T
        if arr.shape[0] == t_expected:
            return arr
        # allow T-1 by prepending zeros
        if arr.shape[0] == t_expected - 1:
            z = np.zeros((1, arr.shape[1]), dtype=np.float32)
            return np.vstack([z, arr])
        print(f"Warning: motion_rt.1D length mismatch (got {arr.shape}, expected T={t_expected}); ignoring.")
        return None
    except Exception as e:
        print(f"Warning: failed to read motion_rt.1D: {e}")
        return None

def _pc1_motion_qc(pc1: np.ndarray, motion: np.ndarray) -> Optional[dict]:
    if motion is None or pc1 is None:
        return None
    if motion.shape[0] != pc1.size or pc1.size < 2:
        return None
    out = {}
    for j in range(motion.shape[1]):
        x = motion[:, j]
        if np.std(x) == 0 or np.std(pc1) == 0:
            out[f"mot{j+1:02d}"] = None
        else:
            out[f"mot{j+1:02d}"] = float(np.corrcoef(pc1, x)[0, 1])
    # convenience: max abs corr across cols
    vals = [abs(v) for v in out.values() if v is not None]
    out["max_abs_corr"] = float(max(vals)) if vals else None
    return out


def _find_epi2t1_composite(run_dir: Path) -> Optional[Path]:
    """
    Expected layout (your pipeline):
      <day>/func/trans/epi2t1_Composite.h5
    run_dir = <day>/func/<run>
    """
    try:
        day_dir = run_dir.parent.parent  # <day>
        trans_dir = day_dir / "func" / "trans"
        comp = trans_dir / "epi2t1_Composite.h5"
        return comp if comp.exists() else None
    except Exception:
        return None


def _maybe_convert_to_nifti(mask_path: Path, out_dir: Path) -> Path:
    """
    antsApplyTransforms is happiest with NIfTI. FastSurfer ROIs are often .mgz.
    Convert .mgz/.mgh -> .nii.gz using mri_convert if available; otherwise try to use as-is.
    """
    low = mask_path.name.lower()
    if low.endswith((".nii", ".nii.gz")):
        return mask_path

    out_dir.mkdir(parents=True, exist_ok=True)
    out_nii = out_dir / (mask_path.stem.replace(".mgz", "").replace(".mgh", "") + ".nii.gz")

    if out_nii.exists():
        return out_nii

    # Try FreeSurfer mri_convert
    try:
        subprocess.run(
            ["mri_convert", str(mask_path), str(out_nii)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return out_nii
    except Exception as e:
        print(f"Warning: failed to mri_convert {mask_path} -> {out_nii}: {e}")
        # Fall back to original path (might still work)
        return mask_path


def warp_t1_mask_to_epi(
    roi_t1_path: Path,
    run_dir: Path,          # MUST be .../day/func/<run>
    epi_ref_3d: Path,
    out_roi_epi: Path,
) -> Path:
    import subprocess

    out_roi_epi.parent.mkdir(parents=True, exist_ok=True)
    if out_roi_epi.exists():
        return out_roi_epi

    # run_dir = .../sub-XXXX/<day>/func/<run>
    day_dir = run_dir.parent.parent # .../sub-XXXX/<day>
    trans_dir = day_dir / "func" / "trans"          # .../sub-XXXX/<day>/func/trans
    inv_comp = trans_dir / "epi2t1_InverseComposite.h5"
    if not inv_comp.exists():
        raise FileNotFoundError(f"Missing T1→EPI transform: {inv_comp}")

    # Convert mgz -> nii.gz if needed
    roi_in = roi_t1_path
    if roi_t1_path.suffix.lower() == ".mgz":
        roi_in = out_roi_epi.parent / (roi_t1_path.stem + ".nii.gz")
        if not roi_in.exists():
            subprocess.run(["mri_convert", str(roi_t1_path), str(roi_in)], check=True)

    cmd = [
        "bash", "-lc",
        f"""
        antsApplyTransforms -d 3 \
          -i {roi_in} \
          -r {epi_ref_3d} \
          -o {out_roi_epi} \
          -t {inv_comp} \
          -n NearestNeighbor --float 1
        """
    ]
    subprocess.run(cmd, check=True)
    return out_roi_epi



def _load_mask_in_epi_space(
    roi: str,
    roi_mask_path: Path,
    tsnr_map_3d: np.ndarray,
    tsnr_img_3d: nib.Nifti1Image,
    run_dir: Path,
    roi_out_dir: Path,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns (mask_bool_3d, affine) in EPI space.
    If roi_mask_path is not in EPI space, tries to warp it using inverse epi2t1_Composite.h5.
    """
    # First try direct load
    data, aff, _ = _load_nifti(roi_mask_path)
    if data.shape == tsnr_map_3d.shape:
        return (data > 0), aff

    # Try warp T1->EPI using inverse of epi2t1 composite
    comp = _find_epi2t1_composite(run_dir)
    if comp is None:
        print(f"  ROI {roi}: mask is not in EPI space and epi2t1_Composite.h5 not found; cannot warp.")
        return None, None

    # Use the already-saved tSNR 3D as reference grid for warping
    # (we ensure it exists below in process_dataset)
    epi_ref = run_dir.parent.parent / "func" / "trans" / "rt_ref_epi.nii"
    if not epi_ref.exists():
        epi_ref = run_dir.parent.parent / "func" / "trans" / "epi_unwarped_mean.nii"

    warped_path = roi_out_dir / "roi_in_epi.nii.gz"
    warped = warp_t1_mask_to_epi(
        roi_t1_path=roi_mask_path,
        run_dir=run_dir,
        epi_ref_3d=epi_ref,
        out_roi_epi=warped_path,
    )
    if warped is None:
        return None, None

    # Load warped ROI and validate
    wdata, waff, _ = _load_nifti(warped)
    if wdata.shape != tsnr_map_3d.shape:
        print(f"  ROI {roi}: warped mask shape {wdata.shape} still != EPI shape {tsnr_map_3d.shape}; skipping.")
        return None, None

    return (wdata > 0), waff

def make_qc_plots(out_root: Path):
    """
    Saves QC plots for each ROI into:
      out_root/<ROI>/qc_*.png
    """
    import matplotlib
    matplotlib.use("Agg")  # force non-interactive backend (prevents "flash then disappear")
    import matplotlib.pyplot as plt
    import numpy as np
    import nibabel as nib

    for roi_dir in sorted([p for p in out_root.iterdir() if p.is_dir()]):
        roi = roi_dir.name
        # required files
        p_expl = roi_dir / "pca_explained.npy"
        p_tc   = roi_dir / "pca_timecourses.npy"
        p_tsnr = roi_dir / "roi_tsnr_in_roi.nii.gz"
        p_roi  = roi_dir / "roi_in_epi.nii.gz"
        p_sel  = roi_dir / "roi_selected_mask.nii.gz"

        if not (p_expl.exists() and p_tc.exists() and p_tsnr.exists() and p_roi.exists() and p_sel.exists()):
            continue

        # ---------- PCA scree ----------
        expl = np.load(p_expl)
        plt.figure()
        plt.plot(np.arange(1, len(expl) + 1), expl, "o-")
        plt.xlabel("PC index")
        plt.ylabel("Explained variance ratio")
        plt.title(f"{roi}: PCA scree")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(roi_dir / "qc_pca_scree.png", dpi=200)
        plt.close()

        # ---------- PCA timecourses ----------
        tc = np.load(p_tc)  # (T, nPC)
        plt.figure(figsize=(10, 6))
        for i in range(min(5, tc.shape[1])):
            plt.plot(tc[:, i] + i * 5, label=f"PC{i + 1}")
        plt.xlabel("Time (TRs)")
        plt.title(f"{roi}: Top PCA timecourses (offset)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(roi_dir / "qc_pca_timecourses.png", dpi=200)
        plt.close()

        # ---------- tSNR distributions ----------
        tsnr = nib.load(str(p_tsnr)).get_fdata()
        roi_mask = nib.load(str(p_roi)).get_fdata() > 0
        sel_mask = nib.load(str(p_sel)).get_fdata() > 0

        tsnr_vals = tsnr[roi_mask]
        sel_vals  = tsnr[sel_mask]

        plt.figure()
        plt.hist(tsnr_vals, bins=100, log=True)
        plt.xlabel("tSNR")
        plt.ylabel("Voxel count (log)")
        plt.title(f"{roi}: tSNR distribution (ROI)")
        plt.tight_layout()
        plt.savefig(roi_dir / "qc_tsnr_hist.png", dpi=200)
        plt.close()

        # ---------- tSNR decay curve ----------
        tsnr_sorted = np.sort(tsnr_vals)[::-1]
        cum = np.arange(1, len(tsnr_sorted) + 1)
        plt.figure()
        plt.plot(cum, tsnr_sorted)
        plt.axvline(min(KVOX, len(tsnr_sorted)), color="gray", ls="--", label=f"KVOX={KVOX}")
        plt.xlabel("Voxel rank (sorted by tSNR)")
        plt.ylabel("tSNR")
        plt.title(f"{roi}: tSNR decay curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(roi_dir / "qc_tsnr_decay.png", dpi=200)
        plt.close()

        # ---------- ROI vs selected comparison ----------
        plt.figure()
        plt.hist(tsnr_vals, bins=100, alpha=0.5, label="All ROI", log=True)
        plt.hist(sel_vals, bins=100, alpha=0.5, label="Selected voxels", log=True)
        plt.xlabel("tSNR")
        plt.ylabel("Voxel count (log)")
        plt.legend()
        plt.title(f"{roi}: tSNR ROI vs selected")
        plt.tight_layout()
        plt.savefig(roi_dir / "qc_tsnr_roi_vs_selected.png", dpi=200)
        plt.close()

    print(f"[QC] Saved plots under: {out_root}")



def main():
    parser = argparse.ArgumentParser(description="Prepare ROI PCA decoder inputs")
    parser.add_argument("-subj", required=True, help="Subject ID (e.g., 00085)")
    parser.add_argument("-day", required=True, help="Day/session ID (e.g., 3)")
    parser.add_argument("-run", required=True, help="Run ID within the day (e.g., 1 or run-01)")
    parser.add_argument(
        "--base-data",
        default=Path(__file__).resolve().parent / "data",
        type=Path,
        help="Root data directory containing sub-*/day/func/run folders (default: ./data)",
    )
    args = parser.parse_args()

    run_dir = _build_run_dir(Path(args.base_data), args.subj, args.day, args.run)
    day_dir = run_dir.parent.parent  # .../day/func/run -> day
    subj_dir = day_dir.parent

    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return

    try:
        anat_dir = subj_dir / "anat"  # sub-00085/anat
        roi_search_dirs = [anat_dir]  # ONLY subject anatomy
        out_root = day_dir / "PCA" / run_dir.name
        process_dataset(
            run_dir,
            ROI_NAMES,
            roi_search_dirs=roi_search_dirs,
            mask_search_dirs=[run_dir, day_dir],  # masks (if any) can still be searched near run
            out_root=out_root,
        )
        make_qc_plots(out_root)
    except Exception as e:
        print(f"Error processing {run_dir}: {e}")
        raise


def _natural_sort_key(p: Path):
    # Sort vol_00001_mc.nii, vol_00002_mc.nii, ... safely
    nums = re.findall(r"\d+", p.name)
    return (int(nums[-1]) if nums else 10**12, p.name)

def _find_3d_series(dirpath: Path) -> List[Path]:
    exts = (".nii", ".nii.gz", ".mgz")
    vols = [p for p in dirpath.glob("*") if p.is_file() and p.name.lower().endswith(exts)]
    vols = sorted(vols, key=_natural_sort_key)
    # Keep only 3D volumes (fast header check)
    out = []
    for p in vols:
        try:
            shp = nib.load(str(p)).header.get_data_shape()
        except Exception:
            continue
        if len(shp) == 3:
            out.append(p)
    return out

def _run_fslmerge(vols: List[Path], out_4d: Path) -> bool:
    """
    Returns True if merged successfully.
    Requires FSL in PATH (fslmerge).
    """
    if not vols:
        return False
    out_4d.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["fslmerge", "-t", str(out_4d)] + [str(v) for v in vols]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return True
    except FileNotFoundError:
        print("Warning: fslmerge not found in PATH. Install FSL or disable AUTO_FSLMERGE_4D.")
        return False
    except subprocess.CalledProcessError as e:
        print("Warning: fslmerge failed.")
        print("STDERR:", e.stderr.strip())
        return False



if __name__ == "__main__":
    main()
