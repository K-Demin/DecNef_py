from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class RegressorSettings:
    enable_motion_regression: bool = True
    mot_reg: str = "mot6"
    max_poly_order: float = np.inf
    TR: float = 1.0
    analysis_space: str = "mni"  # "mni" (default) or "subject"
    use_gs: bool = False
    use_wm: bool = True
    use_vent: bool = True

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
    max_retries: int = 3

    def update(self, overrides: dict[str, Any]) -> None:
        known = {f.name: f for f in fields(self)}
        for key, value in overrides.items():
            if key not in known:
                continue
            if key == "biopac_file" and value is not None:
                setattr(self, key, Path(value))
            else:
                setattr(self, key, value)

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["biopac_file"] is not None:
            payload["biopac_file"] = str(payload["biopac_file"])
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
