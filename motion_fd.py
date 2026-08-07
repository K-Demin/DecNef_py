"""Framewise displacement helpers for RTPSpy motion parameters."""

import numpy as np


def fd_from_rtpspy_delta(delta: np.ndarray, radius_mm: float = 50.0) -> np.ndarray:
    """Compute Power-style FD from RTPSpy ``[rotations, translations]`` deltas.

    RTPSpy exposes motion as ``[roll, pitch, yaw, dS, dL, dP]``: rotations
    are in degrees and translations are in millimetres.
    """
    delta = np.asarray(delta, dtype=float)
    if delta.shape[-1:] != (6,):
        raise ValueError(f"Expected motion vectors with 6 values, got shape {delta.shape}")

    rotation_mm = np.deg2rad(delta[..., :3]) * float(radius_mm)
    translation_mm = delta[..., 3:]
    return np.sum(np.abs(rotation_mm), axis=-1) + np.sum(
        np.abs(translation_mm), axis=-1
    )


def fd_from_rtpspy_motion(motion: np.ndarray, radius_mm: float = 50.0) -> np.ndarray:
    """Compute FD for an RTPSpy motion time series, with zero for its first TR."""
    motion = np.asarray(motion, dtype=float)
    if motion.ndim == 1:
        motion = motion[None, :]
    if motion.ndim != 2 or motion.shape[1] != 6:
        raise ValueError(f"Expected a T-by-6 motion array, got shape {motion.shape}")

    delta = np.zeros_like(motion)
    delta[1:] = np.diff(motion, axis=0)
    return fd_from_rtpspy_delta(delta, radius_mm=radius_mm)
