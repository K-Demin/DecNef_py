#!/usr/bin/env python
"""Pre-NF check: are the decoder voxels inside today's registered EPI?

Usage:
  python check_registration.py --trans /SSD2/DecNef_py/data/sub-00307/1/func/trans \
                               --weights /SSD2/DecNef_py/decoders/olga/<..._weights_PFC.nii.gz>
Exit code 0 = OK, 1 = coverage below --min-frac (do NOT start NF; re-run the preparation).
"""
import argparse, sys
from pathlib import Path
import numpy as np
import nibabel as nib

ap = argparse.ArgumentParser()
ap.add_argument("--trans", required=True, type=Path, help="<day>/func/trans folder")
ap.add_argument("--weights", required=True, type=Path, help="decoder weights NIfTI (T1 space)")
ap.add_argument("--min-frac", type=float, default=0.90)
a = ap.parse_args()

warped = nib.load(str(a.trans / "epi2t1_Warped.nii"))
w = nib.load(str(a.weights))
wd = np.asanyarray(w.dataobj)
ijk = np.argwhere(np.isfinite(wd) & (wd != 0))
# decoder voxel centres -> world -> voxel indices of the registered EPI (nearest neighbour)
xyz = nib.affines.apply_affine(w.affine, ijk)
vox = np.rint(nib.affines.apply_affine(np.linalg.inv(warped.affine), xyz)).astype(int)
ed = np.asanyarray(warped.dataobj).astype(np.float32)
if ed.ndim > 3:
    ed = ed[..., 0]
inside = np.all((vox >= 0) & (vox < np.array(ed.shape[:3])), axis=1)
vals = np.zeros(len(vox), np.float32)
vals[inside] = ed[tuple(vox[inside].T)]
thr = 0.10 * np.percentile(ed[ed > 0], 50) if (ed > 0).any() else 0.0
n_in = int((vals > thr).sum()); n = len(vals); frac = n_in / max(n, 1)
print(f"decoder voxels inside registered EPI: {n_in}/{n} ({100*frac:.1f} %)")
if frac < a.min_frac:
    print("REGISTRATION CHECK FAILED - do not start NF. Rename func/trans and re-run run_preproc.py.")
    sys.exit(1)
print("Registration check OK")
