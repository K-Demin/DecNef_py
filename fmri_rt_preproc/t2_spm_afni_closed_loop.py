import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

from fmri_rt_preproc.utils import ensure_dir

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("t2_spm_afni_closed_loop")

DICOM_LIKE_SUFFIXES = {".dcm", ".ima"}
DEFAULT_DG_LABELS = [17, 53]  # fallback proxy: L/R hippocampus in aparc+aseg-style segmentations


def _run(cmd: list[str]) -> None:
    log.info("RUN: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _is_dicom_like(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in DICOM_LIKE_SUFFIXES


def _convert_dicoms_if_needed(scan_dir: Path, out_prefix: str) -> list[Path]:
    nii_files = sorted(scan_dir.glob(f"{out_prefix}*.nii*"))
    if nii_files:
        return nii_files

    dicoms = sorted([p for p in scan_dir.iterdir() if _is_dicom_like(p)])
    if not dicoms:
        return []

    _run([
        "dcm2niix",
        "-z",
        "y",
        "-b",
        "n",
        "-f",
        out_prefix,
        "-o",
        str(scan_dir),
        str(scan_dir),
    ])
    return sorted(scan_dir.glob(f"{out_prefix}*.nii*"))


def find_or_create_t2(anat_dir: Path) -> Path:
    ensure_dir(anat_dir)
    candidates = sorted(anat_dir.glob("T2*.nii*"))
    if candidates:
        return candidates[0]

    converted = _convert_dicoms_if_needed(anat_dir, "T2")
    if converted:
        return converted[0]

    raise FileNotFoundError(
        f"No T2 file found in {anat_dir}. Provide T2*.nii* or DICOM files."
    )


def run_spm_segmentation(t2_path: Path, out_dir: Path, spm_dir: Path) -> dict[str, Path]:
    ensure_dir(out_dir)
    batch_file = out_dir / "spm_segment_t2_batch.m"
    tpm_path = (spm_dir / "tpm" / "TPM.nii").as_posix()
    script = f"""
spm('defaults','fmri');
spm_jobman('initcfg');
matlabbatch{{1}}.spm.spatial.preproc.channel.vols = {{'{t2_path.as_posix()},1'}};
matlabbatch{{1}}.spm.spatial.preproc.channel.biasreg = 0.001;
matlabbatch{{1}}.spm.spatial.preproc.channel.biasfwhm = 60;
matlabbatch{{1}}.spm.spatial.preproc.channel.write = [1 1];
for k=1:6
    matlabbatch{{1}}.spm.spatial.preproc.tissue(k).tpm = {{sprintf('{tpm_path},%d', k)}};
    matlabbatch{{1}}.spm.spatial.preproc.tissue(k).ngaus = 1;
    matlabbatch{{1}}.spm.spatial.preproc.tissue(k).native = [1 0];
    matlabbatch{{1}}.spm.spatial.preproc.tissue(k).warped = [0 0];
end
matlabbatch{{1}}.spm.spatial.preproc.warp.mrf = 1;
matlabbatch{{1}}.spm.spatial.preproc.warp.cleanup = 1;
matlabbatch{{1}}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
matlabbatch{{1}}.spm.spatial.preproc.warp.affreg = 'mni';
matlabbatch{{1}}.spm.spatial.preproc.warp.fwhm = 0;
matlabbatch{{1}}.spm.spatial.preproc.warp.samp = 3;
matlabbatch{{1}}.spm.spatial.preproc.warp.write = [1 1];
spm_jobman('run',matlabbatch);
exit;
"""
    batch_file.write_text(script)

    matlab = shutil.which("matlab")
    if matlab is None:
        raise RuntimeError("MATLAB not found; required for SPM segmentation.")

    _run([matlab, "-batch", f"run('{batch_file}')"])

    return {
        "gm": t2_path.with_name(f"c1{t2_path.name}"),
        "wm": t2_path.with_name(f"c2{t2_path.name}"),
        "csf": t2_path.with_name(f"c3{t2_path.name}"),
        "bias_corrected": t2_path.with_name(f"m{t2_path.name}"),
        "forward_deform": t2_path.with_name(f"y_{t2_path.name.split('.nii')[0]}.nii"),
    }


def run_afni_t2_cleanup(t2_path: Path, out_dir: Path, mni_template: Path | None = None) -> dict[str, Path]:
    ensure_dir(out_dir)
    unifized = out_dir / "T2_unifize.nii.gz"
    skull = out_dir / "T2_skullstrip.nii.gz"

    if not unifized.exists():
        _run(["3dUnifize", "-input", str(t2_path), "-prefix", str(unifized)])
    if not skull.exists():
        _run(["3dSkullStrip", "-input", str(unifized), "-prefix", str(skull)])

    out = {"unifized": unifized, "skullstrip": skull}

    if mni_template:
        tlrc = out_dir / "T2_in_MNI.nii.gz"
        if not tlrc.exists():
            _run([
                "3dAllineate",
                "-base",
                str(mni_template),
                "-input",
                str(skull),
                "-prefix",
                str(tlrc),
                "-1Dmatrix_save",
                str(out_dir / "T2_to_MNI.aff12.1D"),
                "-final",
                "wsinc5",
                "-cost",
                "lpa+ZZ",
                "-twopass",
            ])
        out["mni"] = tlrc

    return out


def extract_dg_mask(seg_path: Path, out_path: Path, dg_labels: list[int]) -> Path:
    img = nib.load(str(seg_path))
    data = np.asanyarray(img.dataobj)
    mask = np.isin(data, np.array(dg_labels, dtype=data.dtype)).astype(np.uint8)

    if np.count_nonzero(mask) == 0:
        raise RuntimeError(
            f"DG mask is empty from labels {dg_labels}. Provide correct --dg-labels for your segmentation atlas."
        )

    nib.save(nib.Nifti1Image(mask, img.affine, img.header), str(out_path))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop T2 pipeline using SPM + AFNI + optional DG extraction.")
    parser.add_argument("--subject-root", type=Path, required=True, help="Path to sub-XXXX folder.")
    parser.add_argument("--day", required=True, help="Session/day label, e.g. day-01 or 1.")
    parser.add_argument("--spm-dir", type=Path, required=True, help="SPM12 installation directory.")
    parser.add_argument("--mni-template", type=Path, help="Optional MNI template for AFNI alignment.")
    parser.add_argument("--segmentation", type=Path, help="Optional segmentation NIfTI to extract DG from.")
    parser.add_argument("--dg-labels", nargs="+", type=int, default=DEFAULT_DG_LABELS, help="Label IDs for DG extraction.")
    args = parser.parse_args()

    subject_root = args.subject_root
    day_root = subject_root / str(args.day)
    anat_dir = subject_root / "anat"
    out_dir = day_root / "t2_pipeline"
    ensure_dir(out_dir)

    t2_path = find_or_create_t2(anat_dir)
    log.info("Using T2: %s", t2_path)

    spm_outputs = run_spm_segmentation(t2_path, out_dir, args.spm_dir)
    afni_outputs = run_afni_t2_cleanup(
        spm_outputs["bias_corrected"] if spm_outputs["bias_corrected"].exists() else t2_path,
        out_dir,
        args.mni_template,
    )

    dg_mask = None
    if args.segmentation:
        dg_mask = extract_dg_mask(args.segmentation, out_dir / "DG_mask.nii.gz", args.dg_labels)
        log.info("DG mask saved: %s", dg_mask)

    summary = {
        "t2": str(t2_path),
        "spm": {k: str(v) for k, v in spm_outputs.items()},
        "afni": {k: str(v) for k, v in afni_outputs.items()},
        "dg_mask": str(dg_mask) if dg_mask else None,
        "dg_labels": args.dg_labels,
    }
    (out_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info("Pipeline complete. Summary written to %s", out_dir / "pipeline_summary.json")


if __name__ == "__main__":
    main()
