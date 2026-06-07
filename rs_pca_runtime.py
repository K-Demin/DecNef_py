from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from multiprocessing import Queue
from pathlib import Path
from queue import Full
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PCACondition:
    condition_id: str
    symbol: str
    roi: str
    pc: str
    direction: str

    @property
    def score_column(self) -> str:
        return f"{self.roi}_{self.pc}"

    @property
    def direction_sign(self) -> float:
        return -1.0 if self.direction.lower() in {"down", "decrease", "-"} else 1.0


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def build_run_dir(base_data: Path, subj: str, day: str, run: str) -> Path:
    sub_tag = subj if str(subj).startswith("sub-") else f"sub-{subj}"
    day_dir = Path(base_data) / sub_tag / str(day)
    func_dir = day_dir / "func"
    candidates = [func_dir / str(run)]
    if not str(run).startswith("run-"):
        candidates.append(func_dir / f"run-{run}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_pca_root(day_dir: Path, run: str, pca_input: str) -> Path:
    run_name = Path(str(run)).name
    return day_dir / "PCA" / run_name / f"pca_{pca_input}"


def _find_first_existing(paths: list[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _same_zooms(a: tuple[float, ...], b: tuple[float, ...], tol: float = 0.05) -> bool:
    if len(a) < 3 or len(b) < 3:
        return False
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a[:3], b[:3]))


def ensure_pca_t1_reference(
    subject_root: Path,
    trans_dir: Path,
    explicit_path: Optional[Path] = None,
    resolution: str = "epi",
) -> Path:
    if explicit_path is not None:
        explicit_path = Path(explicit_path)
        if not explicit_path.exists():
            raise FileNotFoundError(f"PCA reference image not found: {explicit_path}")
        return explicit_path

    resolution = str(resolution).lower()
    if resolution not in {"epi", "t1"}:
        raise ValueError(f"Unknown PCA reference resolution: {resolution}")

    t1_path = _find_first_existing([
        subject_root / "anat" / "T1_N4.nii",
        subject_root / "anat" / "T1_N4.nii.gz",
        subject_root / "anat" / "T1.nii",
        subject_root / "anat" / "T1.nii.gz",
    ])
    if t1_path is None:
        raise FileNotFoundError(
            f"Could not find a T1 image for PCA reference under {subject_root / 'anat'}"
        )

    if resolution == "t1":
        out_path = trans_dir / "pca_t1_reference_t1res.nii.gz"
        if not out_path.exists():
            import shutil

            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(t1_path, out_path)
        return out_path

    out_path = trans_dir / "pca_t1_reference_epires.nii.gz"
    epi_ref = _find_first_existing([
        trans_dir / "epi_unwarped_mean.nii",
        trans_dir / "epi_unwarped_mean.nii.gz",
        trans_dir / "rt_ref_epi.nii",
        trans_dir / "rt_ref_epi.nii.gz",
    ])
    if epi_ref is None:
        raise FileNotFoundError(
            f"Could not find an EPI reference in {trans_dir} to define PCA reference resolution."
        )

    import nibabel as nib
    from nibabel.processing import resample_to_output

    epi_zooms = nib.load(str(epi_ref)).header.get_zooms()[:3]
    if out_path.exists():
        existing_zooms = nib.load(str(out_path)).header.get_zooms()[:3]
        if _same_zooms(existing_zooms, epi_zooms):
            return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t1_img = nib.load(str(t1_path))
    ref_img = resample_to_output(t1_img, voxel_sizes=epi_zooms, order=1)
    ref_img.set_data_dtype(np.float32)
    nib.save(ref_img, str(out_path))
    return out_path


def load_decoder_artifacts(roi_dir: Path) -> dict[str, np.ndarray]:
    bundle_path = roi_dir / "decoder_bundle.npz"
    if bundle_path.exists():
        z = np.load(bundle_path)
        return {
            "voxel_indices": z["voxel_indices"].astype(np.int64, copy=False),
            "weights": z["weights"].astype(np.float32, copy=False),
            "norm_mean": z["norm_mean"].astype(np.float32, copy=False),
            "norm_std": z["norm_std"].astype(np.float32, copy=False),
        }

    return {
        "voxel_indices": np.load(roi_dir / "decoder_voxel_indices.npy").astype(np.int64, copy=False),
        "weights": np.load(roi_dir / "decoder_weights.npy").astype(np.float32, copy=False),
        "norm_mean": np.load(roi_dir / "decoder_norm_mean.npy").astype(np.float32, copy=False),
        "norm_std": np.load(roi_dir / "decoder_norm_std.npy").astype(np.float32, copy=False),
    }


def score_pca_volume(
    volume_3d: np.ndarray,
    decoder: dict[str, np.ndarray],
    *,
    normalization: str = "zscore",
    score_metric: str = "projection",
) -> np.ndarray:
    flat = volume_3d.reshape(-1).astype(np.float32, copy=False)
    x = flat[decoder["voxel_indices"]]

    if normalization == "zscore":
        safe_std = np.where(decoder["norm_std"] == 0, 1.0, decoder["norm_std"])
        x = (x - decoder["norm_mean"]) / safe_std
    elif normalization == "demean":
        x = x - decoder["norm_mean"]
    elif normalization == "none":
        pass
    else:
        raise ValueError(f"Unknown PCA normalization: {normalization}")

    weights = decoder["weights"]
    if score_metric == "projection":
        return (weights @ x).astype(np.float32, copy=False)
    if score_metric == "cosine":
        x_norm = float(np.linalg.norm(x))
        if x_norm == 0:
            return np.zeros((weights.shape[0],), dtype=np.float32)
        w_norm = np.linalg.norm(weights, axis=1)
        w_norm_safe = np.where(w_norm == 0, 1.0, w_norm)
        return ((weights @ x) / (w_norm_safe * x_norm)).astype(np.float32, copy=False)
    raise ValueError(f"Unknown PCA score metric: {score_metric}")


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def load_reference_stats(path: Optional[Path]) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("columns", payload)


def feedback_from_score(
    raw_score: float,
    condition: PCACondition,
    reference_stats: dict[str, dict[str, float]],
) -> dict[str, float]:
    directed_score = raw_score * condition.direction_sign
    stats = reference_stats.get(condition.score_column)
    if not stats:
        return {
            "directed_score": directed_score,
            "score_z": float("nan"),
            "feedback_score": directed_score,
        }
    std = float(stats["std"])
    if not np.isfinite(std) or std == 0.0:
        raise ValueError(f"Invalid reference std for {condition.score_column}: {std}")
    raw_z = (raw_score - float(stats["mean"])) / std
    directed_z = raw_z * condition.direction_sign
    return {
        "directed_score": directed_score,
        "score_z": directed_z,
        "feedback_score": float(np.clip(_norm_cdf(directed_z) * 100.0, 0.0, 100.0)),
    }


def _make_conditions(
    roi_labels: list[str],
    direction_labels: list[str],
    symbols: list[str],
    target_pc: str,
    symbol_seed: Optional[int],
) -> list[PCACondition]:
    n_conditions = len(roi_labels) * len(direction_labels)
    if len(symbols) < n_conditions:
        raise ValueError(f"Need at least {n_conditions} condition symbols, got {len(symbols)}")
    if len(set(symbols)) < n_conditions:
        raise ValueError("Condition symbols must be unique for blinded PCA feedback.")
    shuffled_symbols = symbols[:]
    random.Random(symbol_seed).shuffle(shuffled_symbols)
    conditions: list[PCACondition] = []
    symbol_iter = iter(shuffled_symbols)
    for roi in roi_labels:
        for direction in direction_labels:
            symbol = next(symbol_iter)
            conditions.append(
                PCACondition(
                    condition_id=f"condition_{symbol}",
                    symbol=symbol,
                    roi=roi,
                    pc=target_pc,
                    direction=direction,
                )
            )
    return conditions


def load_or_create_condition_schedule(
    *,
    private_path: Path,
    public_path: Path,
    roi_labels: list[str],
    direction_labels: list[str],
    symbols: list[str],
    target_pc: str = "PC01",
    condition_seed: Optional[int] = None,
    symbol_seed: Optional[int] = None,
) -> dict:
    schema_version = 2
    if private_path.exists():
        with private_path.open("r", encoding="utf-8") as f:
            schedule = json.load(f)
        expected = {
            "schema_version": schema_version,
            "roi_labels": roi_labels,
            "direction_labels": direction_labels,
            "target_pc": target_pc,
        }
        mismatches = [k for k, v in expected.items() if schedule.get(k) != v]
        if not mismatches:
            if not public_path.exists():
                public_schedule = {
                    "schema_version": schema_version,
                    "created_at": schedule.get("created_at"),
                    "condition_seed": schedule.get("condition_seed"),
                    "symbol_seed": schedule.get("symbol_seed"),
                    "order": schedule.get("order", []),
                    "conditions": [
                        {
                            "condition_id": c["condition_id"],
                            "symbol": c["symbol"],
                        }
                        for c in schedule.get("conditions", [])
                    ],
                }
                public_path.parent.mkdir(parents=True, exist_ok=True)
                public_path.write_text(
                    json.dumps(public_schedule, indent=2) + "\n",
                    encoding="utf-8",
                )
            return schedule
        raise ValueError(
            f"Existing PCA condition key does not match requested setup: {private_path}. "
            f"Mismatched fields: {mismatches}. Move/delete it to create a new blinded mapping."
        )

    conditions = _make_conditions(
        roi_labels,
        direction_labels,
        symbols,
        target_pc,
        symbol_seed,
    )
    order = [c.condition_id for c in conditions]
    random.Random(condition_seed).shuffle(order)
    private_schedule = {
        "schema_version": schema_version,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "roi_labels": roi_labels,
        "direction_labels": direction_labels,
        "target_pc": target_pc,
        "condition_seed": condition_seed,
        "symbol_seed": symbol_seed,
        "order": order,
        "conditions": [asdict(c) for c in conditions],
    }
    public_schedule = {
        "schema_version": schema_version,
        "created_at": private_schedule["created_at"],
        "condition_seed": condition_seed,
        "symbol_seed": symbol_seed,
        "order": order,
        "conditions": [
            {"condition_id": c.condition_id, "symbol": c.symbol}
            for c in conditions
        ],
    }
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(json.dumps(private_schedule, indent=2) + "\n", encoding="utf-8")
    public_path.write_text(json.dumps(public_schedule, indent=2) + "\n", encoding="utf-8")
    return private_schedule


def condition_for_index(schedule: dict, index: int) -> PCACondition:
    order = schedule["order"]
    if not order:
        raise ValueError("Condition schedule has an empty order")
    idx = (max(1, int(index)) - 1) % len(order)
    condition_id = order[idx]
    lookup = {c["condition_id"]: c for c in schedule["conditions"]}
    cond = lookup[condition_id]
    return PCACondition(
        condition_id=cond["condition_id"],
        symbol=cond["symbol"],
        roi=cond["roi"],
        pc=cond["pc"],
        direction=cond["direction"],
    )


def volume_path_for_kind(run_dir: Path, volume_idx: int, volume_kind: str) -> Path:
    if volume_kind == "reg":
        return run_dir / "reg" / f"vol_{volume_idx:05d}_reg.nii"
    if volume_kind == "mc":
        return run_dir / "mc" / f"vol_{volume_idx:05d}_mc.nii"
    if volume_kind == "unwarped":
        return run_dir / "unwarped" / f"vol_{volume_idx:05d}_mc_uw.nii"
    if volume_kind == "t1":
        return run_dir / "t1" / f"vol_{volume_idx:05d}_t1.nii"
    if volume_kind == "mni":
        return run_dir / "mni" / f"vol_{volume_idx:05d}_mni.nii"
    raise ValueError(f"Unknown PCA volume kind: {volume_kind}")


def append_pca_score(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    fieldnames = [
        "volume_idx",
        "timestamp",
        "condition_id",
        "symbol",
        "raw_component_score",
        "directed_score",
        "score_z",
        "feedback_score",
    ]
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def run_realtime_pca_scorer(
    *,
    run_dir: Path,
    pca_root: Path,
    condition: PCACondition,
    score_queue: Queue,
    stop_event,
    reference_stats_path: Optional[Path] = None,
    volume_kind: str = "reg",
    normalization: str = "zscore",
    score_metric: str = "projection",
    max_trs: Optional[int] = None,
    poll_interval: float = 0.05,
) -> None:
    import nibabel as nib

    decoder = load_decoder_artifacts(pca_root / condition.roi)
    reference_stats = load_reference_stats(reference_stats_path)
    pc_idx = int(condition.pc.upper().replace("PC", "")) - 1
    if pc_idx < 0 or pc_idx >= decoder["weights"].shape[0]:
        raise ValueError(
            f"{condition.roi} has {decoder['weights'].shape[0]} PCs, "
            f"cannot use {condition.pc}"
        )

    out_csv = run_dir / "pca_realtime_scores.csv"
    processed: set[int] = set()
    volume_idx = 1
    while True:
        if max_trs is not None and volume_idx > max_trs:
            break
        if stop_event.is_set() and volume_idx in processed:
            break
        if volume_idx in processed:
            volume_idx += 1
            continue

        vol_path = volume_path_for_kind(run_dir, volume_idx, volume_kind)
        if not vol_path.exists():
            if stop_event.is_set():
                break
            time.sleep(poll_interval)
            continue

        try:
            img = nib.load(str(vol_path))
            scores = score_pca_volume(
                np.asanyarray(img.dataobj),
                decoder,
                normalization=normalization,
                score_metric=score_metric,
            )
            raw_score = float(scores[pc_idx])
            feedback = feedback_from_score(raw_score, condition, reference_stats)
            row = {
                "volume_idx": volume_idx,
                "timestamp": time.time(),
                "condition_id": condition.condition_id,
                "symbol": condition.symbol,
                "raw_component_score": raw_score,
                **feedback,
            }
            append_pca_score(out_csv, row)
            try:
                score_queue.put_nowait(
                    {
                        **row,
                        "score_raw": row["feedback_score"],
                        "reg_ready": True,
                        "blind_condition": True,
                    }
                )
            except Full:
                pass
            processed.add(volume_idx)
            volume_idx += 1
        except Exception as exc:
            try:
                score_queue.put_nowait(
                    {
                        "volume_idx": volume_idx,
                        "timestamp": time.time(),
                        "reg_ready": False,
                        "error": str(exc),
                    }
                )
            except Full:
                pass
            processed.add(volume_idx)
            volume_idx += 1
