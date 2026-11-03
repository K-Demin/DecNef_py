## PyHySCO
This is a package for Echo-Planar MRI susceptibility artifact correction implemented in PyTorch.

## DecNef local modifications to PyHySCO

DecNef commit `3255b60` modified the bundled estimator relative to DecNef commit
`b30a116`. These changes affect field validity checks and optimizer stopping.

**Field validity.** Let `Db` be the displacement derivative along the phase-encoding
axis. The two intensity multipliers are `1 + Db` and `1 - Db`. Both are positive
only when `abs(Db) < 1`. Within this interval, the original barrier
`Db⁴ / (1 - Db²)` and its derivatives apply. Previously, invalid values could
receive zero penalty when `nan_to_num` converted `inf / inf` to zero. The local
repair evaluates `D(field)` directly to avoid rounding through `1 + Db - 1`,
assigns infinite cost to invalid trials, and rejects invalid initial fields.
Before export, it checks both polarities and generates the corrected AP/PA
images from the accepted field.

**Optimizer stopping.** DecNef allows up to 50 outer Gauss–Newton updates per
calibration pair. After a nonfinite trial triggers backtracking, a small change
in the field or objective can reflect a shortened step. From that point, the
solver uses its gradient criterion to declare convergence and raises the default
inner PCG limit from 10 to 50 at the same residual tolerance. A fit that exhausts
the outer budget can still produce a field with positive Jacobians. Its log
records `maximum_iterations`.

**Preparing and applying the field.** The DecNef wrapper records the estimator
recipe with the calibration and dependent references. It checks the saved
field in float64 using the AP prototype's voxel spacing and caches this check
by file identity. The runtime applier checks the derivative in float32 when it
builds the sampling grid. Incoming BOLD volumes use that fixed field without
another optimization.

On November 3, 2026, DecNef refit 72 calibration pairs with `3255b60`. All 72
had positive Jacobians for both polarities. The optimizer met the gradient
criterion for 57 and reached the outer cap for 15. All 106 previously saved
field arrays compared in that test matched the corresponding fresh fits exactly.

## Associated Publication
PyHySCO: GPU-Enabled Susceptibility Artifact Distortion Correction in Seconds

https://arxiv.org/abs/2403.10706

Please cite as
```
@article{julian2024pyhysco,
      title={PyHySCO: GPU-Enabled Susceptibility Artifact Distortion Correction in Seconds}, 
      author={Abigail Julian and Lars Ruthotto},
      year={2024},
      journal = {arXiv preprint arXiv:2403.10706}
}
```

## Installation
From PyPI:
```bash
pip install PyHySCO
```

Python package dependencies (automatically installed by pip) are listed in `requirements.txt`.
It is suggested to run the python file `tests/test_all.py` to ensure all tests are passing and the code is setup properly.

## Usage

### Command Line Correction
The program can be run directly from a terminal or command line by using the ```python``` command to run the file ```pyhysco.py``` or if installed using pip the command ```pyhysco```.
Supplying the following required parameters:
* file_1: file path of first image (stored as nii.gz) with phase encoding direction opposite of file_2
* file_2: file path of second image (stored as nii.gz) with phase encoding direction opposite of file_1
* ped: phase-encoding dimension (1, 2, or 3)

Use the help flag (--help) to see optional parameters available.

Minimal Usage:
```bash
pyhysco --file_1 <image1> --file_2 <image2> --ped <phase encoding direction>
```
Example:
```bash
pyhysco --file_1 image1.nii.gz --file_2 image2.nii.gz --ped 1 --output_dir results/ --max_iter 25
```

### Write a Correction Script
A user-written script can be used to call the methods of the program. 

Example:
```python
from EPI_MRI.EPIMRIDistortionCorrection import *
from optimization.GaussNewton import *
import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# load the image and domain information
# change this function call to be the filepath for your data
data = DataObject('../data/156334_v.nii.gz', '../data/156334_-v.nii.gz', 1, device=device,dtype=torch.float32)

loss_func = EPIMRIDistortionCorrection(data, 300, 1e-4, regularizer=myLaplacian3D, PC = JacobiCG)
# initialize the field map
B0 = loss_func.initialize(blur_result=True)
# set-up the optimizer
# change path to be where you want logfile and corrected images to be stored
opt = GaussNewton(loss_func, max_iter=500, verbose=True, path='results/gnpcg-Jac/')
# optimize!
opt.run_correction(B0)
# save field map and corrected images
opt.apply_correction()
# see plot of corrected images
opt.visualize()
```

### Examples and Further Documentation
There are a set of examples in the `examples` directory. Full API documentation is in the `docs` directory. See also `Instructions.md` for an overview of the correction process.

## Acknowledgements
This material is partly based upon work supported by the US National Science Foundation Graduate Research Fellowship under Grant No. 1937971 and NSF Grants DMS-1751636 and DMS-2038118. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the funding agencies.
