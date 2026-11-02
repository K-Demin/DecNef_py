"""Pure ANTs command plans with explicit image-resampling directions.

The anatomical recipe retains the Affine and SyN schedules in June commit
30e6b5d and September commit 0335a2e, with the Rigid stage added in 71381f0.
Dense MI evaluation replaces the sampling-sensitive 20% anatomical metric.
Fixed/moving mask and initialization order are corrected in either direction.
Building a plan does not run ANTs or establish anatomical alignment quality.
"""
from dataclasses import dataclass
from pathlib import Path


ANATOMICAL_RECIPE = "rigid_affine_syn_dense_v1"
BRIDGE_RECIPE = "session_reference_rigid_v1"
OUTPUT_SUFFIXES = ("Composite.h5", "InverseComposite.h5", "Warped.nii", "InverseWarped.nii")

# transform, convergence, shrink factors, smoothing sigmas, metric sampling
_RIGID = ("Rigid[0.1]", "[1000x500x250x100,1e-7,15]", "8x4x2x1", "3x2x1x0vox", "None,1")
_AFFINE = ("Affine[0.1]", "[1000x500x250x100,1e-7,15]", "8x4x2x1", "3x2x1x0vox", "None,1")
_SYN = ("SyN[0.1,1,0]", "[100x70x50x20,1e-7,15]", "6x4x2x1", "3x2x1x0vox", "None,1")
_BRIDGE_RIGID = ("Rigid[0.1]", "[500x250x50,1e-6,10]", "2x2x1", "2x1x0vox", "Regular,0.2")


@dataclass(frozen=True)
class RegistrationPlan:
    args: list[str]
    fixed: Path
    moving: Path
    fixed_mask: Path | None
    moving_mask: Path | None
    # Native ANTs order: moving-to-fixed, fixed-to-moving, moving on fixed,
    # fixed on moving. Direction names refer to IMAGE resampling, not points.
    native_outputs: tuple[Path, ...]
    semantic_outputs: dict[str, Path]
    output_references: dict[Path, Path]
    provenance: dict


def _build_plan(fixed, moving, fixed_mask, moving_mask, prefix, *,
                initialization, stages, semantic_prefix, invert, recipe, direction):
    initialization = str(initialization).lower()
    if initialization == "auto":
        initialization = "identity"
    if initialization not in {"identity", "center_of_mass"}:
        raise ValueError("registration_initialization must be identity, center_of_mass or auto.")
    fixed, moving, prefix = Path(fixed), Path(moving), Path(prefix)
    fixed_mask = Path(fixed_mask) if fixed_mask is not None else None
    moving_mask = Path(moving_mask) if moving_mask is not None else None
    native = tuple(Path(str(prefix) + suffix) for suffix in OUTPUT_SUFFIXES)
    initial = "Identity" if initialization == "identity" else f"[{fixed},{moving},1]"
    masks = f"[{fixed_mask if fixed_mask is not None else 'NULL'},{moving_mask if moving_mask is not None else 'NULL'}]"
    args = [
        "antsRegistration", "--dimensionality", "3", "--float", "1", "--verbose", "1",
        "--output", f"[{prefix},{native[2]},{native[3]}]", "--interpolation", "Linear",
        "--winsorize-image-intensities", "[0.005,0.995]", "--use-histogram-matching", "0",
        "--write-composite-transform", "1", "--initial-moving-transform", initial,
    ]
    stage_records = []
    for transform, convergence, shrink, sigma, sampling in stages:
        metric = f"MI[{fixed},{moving},1,32,{sampling}]"
        args += ["--transform", transform, "--metric", metric,
                 "--convergence", convergence, "--shrink-factors", shrink,
                 "--smoothing-sigmas", sigma, "--masks", masks]
        stage_records.append({"transform": transform, "metric": "MI", "weight": 1,
                              "bins": 32, "sampling": sampling, "convergence": convergence,
                              "shrink_factors": shrink, "smoothing_sigmas": sigma})
    semantic = (native[1], native[0], native[3], native[2]) if invert else native
    return RegistrationPlan(
        args=args, fixed=fixed, moving=moving, fixed_mask=fixed_mask, moving_mask=moving_mask,
        native_outputs=native,
        semantic_outputs={semantic_prefix + suffix: source
                          for suffix, source in zip(OUTPUT_SUFFIXES, semantic)},
        output_references={native[2]: fixed, native[3]: moving},
        provenance={"recipe": recipe, "direction": direction, "initialization": initialization,
                    "stages": stage_records, "interpolation": "Linear", "float": True,
                    "winsorize": [0.005, 0.995], "histogram_matching": False,
                    "fixed": str(fixed), "moving": str(moving),
                    "fixed_mask": str(fixed_mask) if fixed_mask is not None else None,
                    "moving_mask": str(moving_mask) if moving_mask is not None else None},
    )


def build_epi_t1_registration(epi, t1, epi_mask, t1_mask, prefix, *,
                              direction="epi_to_t1", initialization="identity", syn=True):
    """Plan the full anatomical fit, preserving canonical ``epi2t1_*`` meanings.

    ``t1_mask`` may be None. Input masks must already occupy their own image
    grids; callers validate input and output grids and publish the mapped files.
    Changing native direction swaps both images and masks. The recipe is otherwise
    identical; disabling SyN retains both Rigid and Affine stages.
    """
    direction = str(direction).lower()
    if direction not in {"epi_to_t1", "t1_to_epi"}:
        raise ValueError("registration_direction must be epi_to_t1 or t1_to_epi.")
    if not isinstance(syn, bool):
        raise ValueError("registration_syn must be a boolean.")
    invert = direction == "t1_to_epi"
    fixed, moving = (epi, t1) if invert else (t1, epi)
    fixed_mask, moving_mask = (epi_mask, t1_mask) if invert else (t1_mask, epi_mask)
    stages = (_RIGID, _AFFINE, _SYN) if syn else (_RIGID, _AFFINE)
    return _build_plan(fixed, moving, fixed_mask, moving_mask, prefix,
                       initialization=initialization, stages=stages,
                       semantic_prefix="epi2t1_", invert=invert,
                       recipe=ANATOMICAL_RECIPE, direction=direction)


def build_epi_bridge_registration(session_epi, offline_epi, session_mask, offline_mask,
                                  prefix, *, initialization="identity"):
    """Fit an offline EPI to the fixed session slab, exposing S-to-O outputs.

    The fixed session mask limits the metric to acquired anatomy. The forward
    native transform maps the offline reference to the session; its inverse is
    published semantically as ``session_to_reference_Composite.h5``. Its Warped
    diagnostic is therefore the session EPI on the offline reference grid.
    This rigid-only bridge has a fixed fine-grid schedule and no SyN stage.
    """
    return _build_plan(session_epi, offline_epi, session_mask, offline_mask, prefix,
                       initialization=initialization, stages=(_BRIDGE_RIGID,),
                       semantic_prefix="session_to_reference_", invert=True,
                       recipe=BRIDGE_RECIPE, direction="reference_to_session")
