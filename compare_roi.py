import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn.image import resample_to_img
from scipy.ndimage import zoom  # needs scipy installed

# -----------------------------
# PUT YOUR ROI FILES HERE
# -----------------------------
roi_A_path = "/SSD2/DecNef_py/data/sub-00086/3/func/trans/ROI_DECODER.txt"
roi_B_path = "/home/sin/Downloads/ROI_NPS.txt"
template_nifti_path = "/SSD2/DecNef_py/decoders/rweights_NSF_grouppred_cvpcrTMP_nonzeros.nii"

# ----------------------
# PATHS (edit only this)
# ----------------------
A_path = "/SSD2/DecNef_py/data/sub-00086/3/func/1/epi_in_MNI.nii.gz"
B_path = "/home/sin/DecNef_pain_Dec23/reference/00086/3/Functional/NIFTIs/wmeanufKostya-0004-00009-000009-01.nii"
T_path = template_nifti_path
#
# # ----------------------
# # LOAD FILES
# # ----------------------
# A = nib.load(A_path)
# B = nib.load(B_path)
# T = nib.load(T_path)
#
# # ----------------------
# # RESAMPLE ALL → Template space
# # ----------------------
# A_r = resample_to_img(A, T, interpolation="continuous")
# B_r = resample_to_img(B, T, interpolation="continuous")
#
# A_data = A_r.get_fdata()
# B_data = B_r.get_fdata()
# T_data = T.get_fdata()
#
# # ----------------------
# # Use center slice safely
# # ----------------------
# cx = T_data.shape[0] // 2
# cy = T_data.shape[1] // 2
# cz = T_data.shape[2] // 2
#
# # ----------------------
# # PLOT
# # ----------------------
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#
# # A
# axes[0,0].imshow(A_data[:, :, cz], cmap="viridis")
# axes[0,0].set_title("A_resampled")
# axes[0,0].axis("off")
#
# # B
# axes[0,1].imshow(B_data[:, :, cz], cmap="viridis")
# axes[0,1].set_title("B_resampled")
# axes[0,1].axis("off")
#
# # Template
# axes[0,2].imshow(T_data[:, :, cz], cmap="viridis")
# axes[0,2].set_title("Template")
# axes[0,2].axis("off")
#
# # A - Template
# axes[1,0].imshow(A_data[:, :, cz] - T_data[:, :, cz], cmap="bwr")
# axes[1,0].set_title("A - Template")
# axes[1,0].axis("off")
#
# # B - Template
# axes[1,1].imshow(B_data[:, :, cz] - T_data[:, :, cz], cmap="bwr")
# axes[1,1].set_title("B - Template")
# axes[1,1].axis("off")
#
# # A - B
# axes[1,2].imshow(A_data[:, :, cz] - B_data[:, :, cz], cmap="bwr")
# axes[1,2].set_title("A - B")
# axes[1,2].axis("off")
#
# plt.tight_layout()
# plt.show()

# ----------------------------
# LOAD ROI TXT
# ----------------------------
def load_roi_txt(path):
    with open(path, "r") as f:
        nx, ny, nz = map(int, f.readline().split())
        n_vox = int(f.readline().strip())

        vol = np.zeros((nx, ny, nz), dtype=np.float32)

        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            x, y, z = map(int, parts[:3])
            w = float(parts[3])

            if x == 0 and y == 0 and z == 0:
                continue

            vol[x - 1, y - 1, z - 1] = w

    return vol


# ----------------------------
# LOAD ALL THREE
# ----------------------------
A = load_roi_txt(roi_A_path)
B = load_roi_txt(roi_B_path)

template_img = nib.load(template_nifti_path)
T = np.asanyarray(template_img.get_fdata())

print("A:", A.shape, "non-zero:", np.count_nonzero(A))
print("B:", B.shape, "non-zero:", np.count_nonzero(B))
print("T:", T.shape, "non-zero:", np.count_nonzero(T))


# ----------------------------
# RESAMPLE A AND B → TEMPLATE GRID
# ----------------------------
def resample_to_template(vol, template):
    zoom_factors = np.array(template.shape) / np.array(vol.shape, dtype=float)
    print("Zoom factors:", zoom_factors)
    return zoom(vol, zoom_factors, order=0)  # nearest neighbor


A_res = resample_to_template(A, T)
B_res = resample_to_template(B, T)

print("A_res:", A_res.shape)
print("B_res:", B_res.shape)


# ----------------------------
# DIFFERENCE MAPS
# ----------------------------
diff_A_T = A_res - T
diff_B_T = B_res - T
diff_A_B = A_res - B_res

# pick mid-slice
z = T.shape[2] // 2


# ----------------------------
# VISUALIZATION
# ----------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

def show(ax, data, title, cmap="viridis", vmin=None, vmax=None):
    im = ax.imshow(data.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    return im

vmin = min(A_res[:, :, z].min(), B_res[:, :, z].min(), T[:, :, z].min())
vmax = max(A_res[:, :, z].max(), B_res[:, :, z].max(), T[:, :, z].max())

# Row 1: raw ROIs
show(axes[0, 0], A_res[:, :, z], "A_resampled")
show(axes[0, 1], B_res[:, :, z], "B_resampled")
show(axes[0, 2], T[:, :, z], "Template decoder")

# Row 2: difference maps
show(axes[1, 0], diff_A_T[:, :, z], "A - Template", cmap="bwr")
show(axes[1, 1], diff_B_T[:, :, z], "B - Template", cmap="bwr")
show(axes[1, 2], diff_A_B[:, :, z], "A - B", cmap="bwr")

plt.tight_layout()
plt.show()