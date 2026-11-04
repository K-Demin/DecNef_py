from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from fmri_rt_preproc.config import (
    normalize_registration_config, optional_registration_path,
    validate_registration_direction, validate_registration_settings,
    validate_epi_reference_n4,
    validate_registration_threads, validate_bridge_initialization,
)


REGISTRATION_KEYS = ("registration_initialization", "registration_syn",
                     "registration_direction", "offline_assets", "registration_qc_policy", "epi_reference_n4",
                     "registration_threads", "registration_bridge_initialization")
VOXEL_NORMALIZATION_KEYS = ("voxel_norm_mode", "voxel_norm_reference", "voxel_norm_ref_volumes")
DECODER_QC_KEYS = ("enable_decoder_correlation_qc", "decoder_qc_reference", "decoder_qc_roi_mask")


def validate_voxel_normalization_settings(mode: str, reference: str, ref_volumes: int) -> tuple[str, str, int]:
    """Validate a complete normalization choice before updating live settings."""
    mode = str(mode).lower()
    reference = str(reference).lower()
    if mode not in {"zscore", "mean"}:
        raise ValueError("voxel_norm_mode must be 'zscore' or 'mean'.")
    if reference not in {"fixed", "running"}:
        raise ValueError("voxel_norm_reference must be 'fixed' or 'running'.")
    try:
        count = int(ref_volumes)
        integral = not isinstance(ref_volumes, (bool, np.bool_)) and float(ref_volumes) == count
    except (TypeError, ValueError, OverflowError):
        integral = False
        count = 0
    minimum = 2 if mode == "zscore" else 1
    if not integral or count < minimum:
        raise ValueError(f"voxel_norm_ref_volumes must be an integer >= {minimum} for {mode} mode.")
    return mode, reference, count


@dataclass
class RegressorSettings:
    enable_motion_regression: bool = True
    mot_reg: str = "mot6"
    max_poly_order: float = np.inf
    TR: float = 1.0
    analysis_space: str = "mni"  # "epi", "t1", or "mni" (default)
    smoothing_fwhm_mm: float = 0.0  # 0: no filtering; cortex/custom masking still applies
    analysis_mask: str = "whole_brain"  # whole_brain, cortical_gm, custom
    analysis_mask_file: Optional[str] = None  # binary NIfTI for custom mode
    analysis_mask_space: str = "final"  # custom mask coordinates: final, epi, t1, mni
    use_gpu_resampler: bool = False
    gpu_resampler_device: str = "cuda"
    gpu_resampler_interpolation: str = "linear"  # linear or lanczos (CUDA + Triton)
    use_gs: bool = False
    use_wm: bool = True
    use_vent: bool = True

    # Spatial transforms -> smoothing -> original-intensity GLM -> normalization.
    # zscore starts from the first fit's residual history; mean uses pre-GLM data.
    # N also sets the GLM startup minimum. Running references use prior data only.
    voxel_norm_mode: str = "zscore"  # zscore (sample SD), mean (100 * residual / raw mean)
    voxel_norm_reference: str = "fixed"  # fixed first N, or running prior observations
    voxel_norm_ref_volumes: int = 30
    enable_fd_censor_reg: bool = True
    fd_thr: float = 0.3

    enable_dvars_censor_reg: bool = True
    dvars_thr_robust_z: float = 3.0
    censor_plus1: bool = True
    dvars_warmup: int = 20
    dvars_mask_source: str = "ref_mask"

    enable_biopac_physio: bool = True
    biopac_phys_reg: str = "RICOR8"
    biopac_host: str = "0.0.0.0"
    biopac_port: int = 15000
    biopac_timeout: float = 0.3
    biopac_handshake: bool = True
    biopac_start_online_only: bool = False
    biopac_mode: str = "tcp"
    biopac_file: Optional[Path] = None
    biopac_poll_interval: float = 0.05
    biopac_timelag: bool = False
    max_workers: int = 6
    rt_max_scan_length: int = 2000
    max_retries: int = 3
    pipeline_engine: str = "parallel_ordered"  # "parallel_ordered" or "legacy"
    commit_wait_timeout_s: float = 1.0
    skip_first_trs: int = 0
    truncate_t1_to_epi_fov: bool = True
    truncate_t1_padding_vox: int = 2
    registration_initialization: str = "center_of_mass"
    registration_syn: bool = True
    registration_direction: str = "epi_to_t1"
    offline_assets: Optional[Path] = None
    registration_qc_policy: Optional[str] = None
    epi_reference_n4: bool = True  # N4 reference copy for fitting; original MC target is retained
    fieldmap_method: str = "pyhysco"  # "pyhysco" or "ants"
    epi_phase_encoding: str = "PA"  # "AP" or "PA"
    use_preloaded_pyhysco: bool = True
    save_intermediate_unwarped: bool = True  # pre-MC unwarped intermediate only
    save_decoder_contributions: bool = True  # ready linear-decoder voxel terms, bias excluded
    enable_volume_streamer: bool = False
    volume_stream_kind: str = "unwarped"  # "unwarped" aliases fully corrected "mc"
    volume_stream_every_n: int = 1
    volume_stream_max_queue: int = 2
    volume_stream_window_title: str = "RT volume QC"
    volume_stream_window_x: int = 50
    volume_stream_window_y: int = 50
    pyhysco_device: str = "cuda"
    pyhysco_backend: str = "grid_sample"  # "grid_sample" or "gather1d"
    registration_threads: int = 1  # Anatomical ANTs ITK threads, including SyN; OMP policy is independent
    registration_bridge_initialization: str = "identity"  # Separate EPI-to-EPI initialization
    enable_decoder_correlation_qc: bool = True
    decoder_qc_reference: Optional[str] = None
    decoder_qc_roi_mask: Optional[str] = None

    def __post_init__(self) -> None:
        self.update({})

    def update(self, overrides: dict[str, Any]) -> None:
        overrides = normalize_registration_config(overrides)
        if not isinstance(overrides.get("save_decoder_contributions", self.save_decoder_contributions), bool):
            raise ValueError("save_decoder_contributions must be a boolean.")
        registration_values = {key: overrides.get(key, getattr(self, key)) for key in REGISTRATION_KEYS}
        registration_values["registration_initialization"], registration_values["registration_syn"] = validate_registration_settings(
            registration_values["registration_initialization"], registration_values["registration_syn"]
        )
        registration_values["registration_direction"] = validate_registration_direction(registration_values["registration_direction"])
        registration_values["registration_threads"] = validate_registration_threads(registration_values["registration_threads"])
        registration_values["registration_bridge_initialization"] = validate_bridge_initialization(registration_values["registration_bridge_initialization"])
        registration_values["epi_reference_n4"] = validate_epi_reference_n4(registration_values["epi_reference_n4"])
        registration_values["offline_assets"] = optional_registration_path(registration_values["offline_assets"], "offline_assets")
        policy = optional_registration_path(registration_values["registration_qc_policy"], "registration_qc_policy")
        registration_values["registration_qc_policy"] = str(policy) if policy is not None else None
        decoder_qc_values = {key: overrides.get(key, getattr(self, key)) for key in DECODER_QC_KEYS}
        if not isinstance(decoder_qc_values["enable_decoder_correlation_qc"], bool):
            raise ValueError("enable_decoder_correlation_qc must be a boolean.")
        for key in ("decoder_qc_reference", "decoder_qc_roi_mask"):
            value = decoder_qc_values[key]
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{key} must be a nonempty string path or null.")
                decoder_qc_values[key] = str(Path(value).expanduser())
        normalization_values = validate_voxel_normalization_settings(*(
            overrides.get(key, getattr(self, key)) for key in VOXEL_NORMALIZATION_KEYS
        ))
        interpolation = str(overrides.get(
            "gpu_resampler_interpolation", self.gpu_resampler_interpolation)).lower()
        if interpolation not in {"linear", "lanczos"}:
            raise ValueError("gpu_resampler_interpolation must be 'linear' or 'lanczos'.")
        if interpolation == "lanczos":
            enabled = overrides.get("use_gpu_resampler", self.use_gpu_resampler)
            device = str(overrides.get("gpu_resampler_device", self.gpu_resampler_device))
            if not enabled or not device.startswith("cuda"):
                raise ValueError("Lanczos requires use_gpu_resampler=true and a CUDA device.")
        known = {f.name: f for f in fields(self)}
        for key, value in overrides.items():
            if (key not in known or key in REGISTRATION_KEYS or key in VOXEL_NORMALIZATION_KEYS or key in DECODER_QC_KEYS
                    or key == "gpu_resampler_interpolation"):
                continue
            if key == "biopac_file" and value is not None:
                setattr(self, key, Path(value))
            elif key in {"analysis_mask", "analysis_mask_space"}:
                value = str(value).lower()
                allowed = {"whole_brain", "cortical_gm", "custom"} if key == "analysis_mask" else {"final", "epi", "t1", "mni"}
                if value not in allowed:
                    raise ValueError(f"Invalid {key}: {value}")
                setattr(self, key, value)
            elif key == "analysis_mask_file":
                self.analysis_mask_file = str(value) if value else None
            elif key == "smoothing_fwhm_mm":
                fwhm = float(value)
                if not np.isfinite(fwhm) or fwhm < 0:
                    raise ValueError("smoothing_fwhm_mm must be finite and >= 0.")
                self.smoothing_fwhm_mm = fwhm
            elif key == "skip_first_trs":
                setattr(self, key, max(0, int(value)))
            elif key == "rt_max_scan_length":
                setattr(self, key, max(1, int(value)))
            elif key == "truncate_t1_padding_vox":
                setattr(self, key, max(0, int(value)))
            elif key == "fieldmap_method":
                setattr(self, key, str(value).lower())
            elif key == "epi_phase_encoding":
                setattr(self, key, str(value).upper())
            elif key == "pyhysco_backend":
                setattr(self, key, str(value).lower())
            elif key == "volume_stream_kind":
                setattr(self, key, str(value).lower())
            elif key == "volume_stream_every_n":
                setattr(self, key, max(1, int(value)))
            elif key == "volume_stream_max_queue":
                setattr(self, key, max(1, int(value)))
            else:
                setattr(self, key, value)
        for key, value in registration_values.items():
            setattr(self, key, value)
        for key, value in decoder_qc_values.items():
            setattr(self, key, value)
        for key, value in zip(VOXEL_NORMALIZATION_KEYS, normalization_values):
            setattr(self, key, value)
        self.gpu_resampler_interpolation = interpolation

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("biopac_file", "offline_assets"):
            if payload[key] is not None:
                payload[key] = str(payload[key])
        if np.isinf(payload["max_poly_order"]):
            payload["max_poly_order"] = "inf"
        return payload


def _coerce_special_values(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    if out.get("max_poly_order") == "inf":
        out["max_poly_order"] = np.inf
    return out


def load_regressor_settings(path: str | Path | None = None) -> RegressorSettings:
    settings = RegressorSettings()
    if path is None:
        return settings

    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    settings.update(_coerce_special_values(data))
    return settings


def save_regressor_settings(path: str | Path, settings: RegressorSettings | None = None) -> None:
    payload = (settings or RegressorSettings()).to_json_dict()
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def add_registration_arguments(parser: argparse.ArgumentParser) -> None:
    """Use identical, opt-in CLI overrides in every preparation launcher."""
    parser.add_argument(
        "--registration-direction", choices=["epi_to_t1", "t1_to_epi"], default=None,
        help="Native anatomical fit direction (default: epi_to_t1). "
             "Both directions preserve the same canonical EPI-to-T1 output names.",
    )
    parser.add_argument(
        "--registration-initialization", choices=["identity", "center_of_mass", "auto"],
        default=None,
        help="Anatomical registration initialization: center_of_mass is the default; "
             "explicit identity and auto use headers.",
    )
    parser.add_argument(
        "--registration-threads", type=int, default=None,
        help="ITK threads for anatomical Rigid/Affine/SyN (default: 1); independent OMP policy is unchanged.",
    )
    parser.add_argument(
        "--registration-bridge-initialization", choices=["identity", "center_of_mass", "auto"], default=None,
        help="Separate EPI-to-EPI bridge initialization (default: identity); auto aliases identity.",
    )
    parser.add_argument(
        "--registration-syn", action=argparse.BooleanOptionalAction, default=None,
        help="Enable SyN refinement after Rigid + Affine "
             "(default: enabled; overrides the settings file only when supplied).",
    )
    parser.add_argument(
        "--epi-reference-n4", action=argparse.BooleanOptionalAction, default=None,
        help="Bias-correct a reference copy before spatial registration (default: enabled); "
             "does not N4-correct incoming volumes or replace the motion-correction target.",
    )
    parser.add_argument(
        "--offline-assets", default=None, type=Path,
        help="Offline anatomical/EPI asset manifest to reuse for session preparation.",
    )
    parser.add_argument(
        "--registration-qc-policy", default=None,
        help="Registration QC policy JSON path; omission requires manual review.",
    )


def apply_registration_arguments(settings: RegressorSettings, args: argparse.Namespace) -> None:
    """Apply explicit CLI choices after loading a profile, preserving omitted values."""
    settings.update({
        key: getattr(args, key)
        for key in REGISTRATION_KEYS
        if getattr(args, key, None) is not None
    })


def registration_cli_arguments(settings: RegressorSettings) -> list[str]:
    """Forward resolved choices explicitly, independently of child defaults."""
    arguments = [
        "--registration-direction", settings.registration_direction,
        "--registration-initialization", settings.registration_initialization,
        "--registration-threads", str(settings.registration_threads),
        "--registration-bridge-initialization", settings.registration_bridge_initialization,
        "--registration-syn" if settings.registration_syn else "--no-registration-syn",
        "--epi-reference-n4" if settings.epi_reference_n4 else "--no-epi-reference-n4",
    ]
    for key, flag in (("offline_assets", "--offline-assets"),
                      ("registration_qc_policy", "--registration-qc-policy")):
        value = getattr(settings, key)
        if value is not None:
            arguments += [flag, str(value)]
    return arguments
