"""
roi_rs_pca_decoder_prep.py

HOW TO USE
----------
1) Keep the USER SETTINGS block below as-is unless you want to tweak ROI/KVOX/etc.
2) Run the script by specifying subject and run/day IDs (data are under ./data):
       python roi_rs_pca_decoder_prep.py -subj 00085 -run 3
3) Inputs are discovered under ./data/sub-<subj>/<run>/ and outputs are written
   to ./data/sub-<subj>/PCA/<run>/<ROI>/.

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
FD_THRESH = 0.3
MIN_NEIGHBORS = 0  # 0 disables neighbor filtering
USE_ZSCORE = True
SEED = 0
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


def _list_nifti_files(data_dir: Path) -> List[Path]:
    return [p for p in data_dir.rglob("*.nii*") if p.is_file()]


def _is_rest_name(name: str) -> bool:
    low = name.lower()
    keywords = ["rest", "rs", "bold", "func", "task-rest", "resting"]
    return any(k in low for k in keywords)


def _rs_priority_for_pca(path: Path) -> Tuple[int, int]:
    low = path.name.lower()
    denoise_keys = ["denoise", "resid", "clean", "regress", "regressed", "nuis", "filtered", "noise"]
    mc_keys = ["mc", "motioncorr", "distcorr", "preproc"]
    has_denoise = any(k in low for k in denoise_keys)
    has_mc = any(k in low for k in mc_keys)
    return (0 if has_denoise else 1, 0 if has_mc else 1)


def _rs_priority_for_tsnr(path: Path) -> Tuple[int, int]:
    low = path.name.lower()
    pre_keys = ["mc", "distcorr", "stc", "preproc", "raw"]
    denoise_keys = ["denoise", "resid", "clean", "regress", "regressed", "nuis", "filtered", "noise"]
    has_pre = any(k in low for k in pre_keys)
    has_denoise = any(k in low for k in denoise_keys)
    return (0 if has_pre else 1, 1 if has_denoise else 0)


def _load_nifti(path: Path) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    img = nib.load(str(path))
    data = np.asanyarray(img.get_fdata(), dtype=np.float32)
    return data, img.affine, img.header


def _discover_rs_inputs(data_dir: Path) -> Tuple[Path, Path]:
    candidates: List[Tuple[Path, nib.Nifti1Header]] = []
    for path in _list_nifti_files(data_dir):
        if not _is_rest_name(path.name):
            continue
        try:
            hdr = nib.load(str(path)).header
        except Exception:
            continue
        if len(hdr.get_data_shape()) != 4:
            continue
        candidates.append((path, hdr))

    if not candidates:
        raise FileNotFoundError(f"No resting-state NIfTI found under {data_dir}")

    pca_sorted = sorted(candidates, key=lambda x: _rs_priority_for_pca(x[0]))
    tsnr_sorted = sorted(candidates, key=lambda x: _rs_priority_for_tsnr(x[0]))
    pca_rs = pca_sorted[0][0]
    tsnr_rs = tsnr_sorted[0][0] if tsnr_sorted else pca_rs
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


def _discover_roi_masks(data_dir: Path, roi_names: Sequence[str]) -> dict:
    all_nii = _list_nifti_files(data_dir)
    roi_masks = {}
    for roi in roi_names:
        roi_low = roi.lower()
        matches = []
        for p in all_nii:
            name_low = p.name.lower()
            if roi_low in name_low and any(key in name_low for key in ["roi", "mask"]):
                matches.append(p)
        if not matches:
            # allow looser match
            for p in all_nii:
                if roi_low in p.name.lower():
                    matches.append(p)
        if matches:
            chosen = _discover_mask(matches, [roi_low])
            roi_masks[roi] = chosen or matches[0]
    return roi_masks


def _discover_optional_mask(data_dir: Path, name_keywords: Sequence[str]) -> Optional[Path]:
    all_nii = _list_nifti_files(data_dir)
    return _discover_mask(all_nii, name_keywords)


def _discover_fd_vector(data_dir: Path, t_expected: int) -> Optional[np.ndarray]:
    text_exts = [".txt", ".tsv", ".csv"]
    fd_files = []
    for ext in text_exts:
        fd_files.extend(data_dir.rglob(f"*fd*{ext}"))
        fd_files.extend(data_dir.rglob(f"*framewise*{ext}"))

    for f in fd_files:
        try:
            fd = np.loadtxt(f, dtype=np.float32, delimiter=None)
        except Exception:
            continue
        fd = np.atleast_1d(fd)
        if fd.ndim > 1:
            fd = fd.reshape(-1)
        if fd.size == t_expected:
            print(f"Found FD file: {f} (length {fd.size})")
            return fd
        else:
            print(f"Warning: FD length mismatch for {f} (found {fd.size}, expected {t_expected}); ignoring.")
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    roi_data, roi_affine, _ = _load_nifti(roi_mask_path)
    if roi_data.shape != tsnr_map.shape:
        raise ValueError(f"ROI {roi} mask shape {roi_data.shape} does not match RS shape {tsnr_map.shape}")
    if not _affine_close(roi_affine, reference_img.affine):
        print(f"  Warning: ROI {roi} affine differs from RS affine; proceeding with caution.")
    roi_mask = roi_data > 0

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


def process_dataset(data_dir: Path, roi_names: Sequence[str], out_root: Optional[Path] = None) -> None:
    print(f"\nProcessing dataset: {data_dir}")
    pca_rs_path, tsnr_rs_path = _discover_rs_inputs(data_dir)
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

    fd_vec = _discover_fd_vector(data_dir, t_all)

    brain_mask_path = _discover_optional_mask(data_dir, ["brainmask", "mask_brain", "brain_mask"])
    gm_mask_path = _discover_optional_mask(data_dir, ["gm_mask", "gmmask", "ribbon", "gm"])

    roi_masks = _discover_roi_masks(data_dir, roi_names)
    if not roi_masks:
        print("No ROI masks found; skipping dataset.")
        return

    out_root = out_root or data_dir / "PCA"
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


def _build_run_dir(base_data: Path, subj: str, run: str) -> Path:
    subj_dir = base_data / f"sub-{subj}"
    # Allow run id with or without prefix (e.g., "3" or "run-03")
    candidate_runs = [subj_dir / run]
    if not run.startswith("run-"):
        candidate_runs.append(subj_dir / f"run-{run}")
    for c in candidate_runs:
        if c.exists():
            return c
    # Fallback to first candidate even if missing so error surfaces downstream
    return candidate_runs[0]


def main():
    parser = argparse.ArgumentParser(description="Prepare ROI PCA decoder inputs")
    parser.add_argument("-subj", required=True, help="Subject ID (e.g., 00085)")
    parser.add_argument("-run", required=True, help="Run/day ID (e.g., 3 or run-03)")
    parser.add_argument(
        "--base-data",
        default=Path(__file__).resolve().parent / "data",
        type=Path,
        help="Root data directory containing sub-*/<run>/ (default: ./data)",
    )
    args = parser.parse_args()

    run_dir = _build_run_dir(Path(args.base_data), args.subj, args.run)
    subj_dir = run_dir.parent

    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return

    try:
        out_root = subj_dir / "PCA" / run_dir.name
        process_dataset(run_dir, ROI_NAMES, out_root=out_root)
    except Exception as e:
        print(f"Error processing {run_dir}: {e}")


if __name__ == "__main__":
    main()
