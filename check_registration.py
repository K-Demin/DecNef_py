#!/usr/bin/env python
"""Check registration QC reports, or record explicit review of their images.

Optional decoder-intensity coverage retains a separate coverage guard. It never
approves registration. Exit 0 means all checks pass; exit 1 means blocked.
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from fmri_rt_preproc.registration_qc import (
    approve_registration_qc, assert_qc_gate, load_registration_qc_report,
)


def decoder_signal_coverage(trans, weights):
    """Legacy signal-presence summary, not geometric registration accuracy."""
    warped = nib.load(str(Path(trans) / "epi2t1_Warped.nii"))
    decoder = nib.load(str(weights))
    values = np.asarray(decoder.dataobj)
    if values.ndim != 3:
        raise ValueError("Decoder coverage requires a 3D weight image")
    ijk = np.argwhere(np.isfinite(values) & (values != 0))
    xyz = nib.affines.apply_affine(decoder.affine, ijk)
    vox = np.rint(nib.affines.apply_affine(np.linalg.inv(warped.affine), xyz)).astype(int)
    epi = np.asarray(warped.dataobj, dtype=np.float32)
    if epi.ndim == 4:
        epi = epi[..., 0]
    if epi.ndim != 3:
        raise ValueError("Decoder coverage requires a 3D or 4D registered EPI")
    inside = np.all((vox >= 0) & (vox < np.array(epi.shape)), axis=1)
    samples = np.zeros(len(vox), np.float32)
    samples[inside] = epi[tuple(vox[inside].T)]
    positive = epi[np.isfinite(epi) & (epi > 0)]
    threshold = .10 * float(np.median(positive)) if positive.size else 0.
    count = int(np.sum(np.isfinite(samples) & (samples > threshold)))
    return {"signal_voxels": count, "decoder_voxels": len(vox),
            "fraction": count / len(vox) if len(vox) else None,
            "intensity_threshold": threshold}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trans", required=True, type=Path,
                        help="Prepared transform folder containing registration_qc*.json and review PNGs")
    parser.add_argument("--approve", action="store_true",
                        help="Record that you visually reviewed all current report images; cannot waive failures")
    parser.add_argument("--reviewer", help="Name of the person who inspected alignment and coverage (required with --approve)")
    parser.add_argument("--note", default="", help="Optional review note saved with approval")
    parser.add_argument("--weights", type=Path, help="Optional legacy decoder signal-coverage guard (T1-space weights)")
    parser.add_argument("--min-frac", type=float, default=.90,
                        help="Minimum decoder signal-coverage fraction when --weights is supplied; not alignment accuracy (default: 0.90)")
    args = parser.parse_args(argv)
    if args.approve and not (args.reviewer and args.reviewer.strip()):
        parser.error("--approve requires --reviewer naming the person who reviewed the images")
    if not 0 <= args.min_frac <= 1:
        parser.error("--min-frac must be between 0 and 1")
    try:
        if args.weights:
            coverage = decoder_signal_coverage(args.trans, args.weights)
            fraction = coverage["fraction"]
            text = f"{100 * fraction:.1f}%" if fraction is not None else "no nonzero decoder voxels"
            print(f"Descriptive decoder signal coverage: {coverage['signal_voxels']}/{coverage['decoder_voxels']} ({text})")
            if fraction is None or fraction < args.min_frac:
                raise ValueError(f"Decoder signal-coverage guard failed (required {100 * args.min_frac:.1f}%). "
                                 "Signal coverage does not establish anatomical alignment.")
        paths = sorted(path for path in args.trans.glob("registration_qc*.json")
                       if not path.name.endswith(".approval.json"))
        if not paths:
            raise ValueError("No registration QC reports. Generate the registration reports before approval or neurofeedback.")
        reports = [load_registration_qc_report(path) for path in paths]
        for path, report in zip(paths, reports):
            print(f"{path.name}: {report['gate_status']} ({report['comparison']})")
            print("  Metrics: " + json.dumps(report["metrics"], sort_keys=True))
            if report.get("review_png"):
                print("  Review image: " + report["review_png"])
        if args.approve:
            if any(report["gate_status"] == "fail" for report in reports):
                raise ValueError("Failed registration QC cannot be approved; correct the registration or inputs first.")
            for path in paths:
                approve_registration_qc(path, reviewer=args.reviewer, note=args.note)
        assert_qc_gate(paths)
        print("Registration QC reports approved. The NF launcher also checks the session's required report set and assets.")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"REGISTRATION QC BLOCKED: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
