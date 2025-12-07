###
#
#
# Add denoising
# Rearrange files a bit - most of anat in a subj folder
#
#
#
# from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fmri_rt_preproc_pipeline")

from fmri_rt_preproc.config import SubjectDayConfig
from fmri_rt_preproc.utils import run, ensure_dir, run_in_conda
from fmri_rt_preproc.RTPSpy_tools.rtp_volreg import RtpVolreg
from pathlib import Path
from nilearn import plotting, image
import nibabel as nib
import numpy as np
import gzip
import shutil

def gunzip_python(gz_path):
    gz_path = Path(gz_path)
    out_path = gz_path.with_suffix("")  # removes .gz → gives .nii

    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return out_path

class FMRIRealtimePreprocessor:
    def __init__(self, cfg: SubjectDayConfig,
                 fastsurfer_env: str = "fastsurfer",
                 ants_env: str = "ants_env"):
        self.cfg = cfg
        self.anat_dir = cfg.root / "anat"
        self.fmap_dir = cfg.root / "fmap"
        self.func_dir = cfg.root / "func"
        self.decoder_template = (cfg.root / cfg.decoder_template).resolve()
        self.fastsurfer_env = fastsurfer_env
        self.ants_env = ants_env
        self.trans_dir = self.func_dir / "trans"

        # --- NEW: global EPI reference for RT (per day) ---
        # This will be set from the FIRST run we preprocess.
        self.rt_ref_epi = self.func_dir / "rt_ref_epi.nii"
        self.rt_ref_mask = self.func_dir / "rt_ref_epi_mask.nii"

    # ---------- Top-level entry points ----------

    def run_all(self):
        ensure_dir(self.anat_dir)
        ensure_dir(self.func_dir)
        ensure_dir(self.trans_dir)
        self._prepare_anat()
        self._prepare_fieldmap()
        for run_cfg in self.cfg.runs:
            self._prepare_run(run_cfg)

    # ---------- T1 / anatomical ----------

    def _prepare_anat(self):
        ensure_dir(self.anat_dir)

        # ---------- 1. N4 BIAS CORRECTION ----------
        t1 = self.cfg.t1_file
        t1_n4 = self.anat_dir / "T1_N4.nii"

        if t1_n4.exists():
            print("✓ N4: already exists — skipping")
        else:
            print("→ Running N4")
            run(["N4BiasFieldCorrection", "-d", "3", "-i", str(t1), "-o", str(t1_n4)])

        # ---------- 2. FASTSURFER ----------
        fastsurfer_dir = self.anat_dir / "fastsurfer"
        ensure_dir(fastsurfer_dir)

        sid = f"{self.cfg.subject_id}_day-{self.cfg.day_id}"
        aparc_aseg = fastsurfer_dir / sid / "mri" / "aseg.auto_noCCseg.mgz"

        if aparc_aseg.exists():
            print("✓ FastSurfer: already exists — skipping")
        else:
            print("→ Running FastSurfer")
            self._run_fastsurfer(t1, fastsurfer_dir)

        # ---------- 3. BRAIN MASKS ----------
        t1_combined = self.anat_dir / "T1_combined_mask.nii"

        if t1_combined.exists():
            print("✓ Brain masks: already exist — skipping")
        else:
            print("→ Creating brain masks")
            self._make_brain_masks(t1_n4, fastsurfer_dir)

        # ---------- 4. SYNTHMORPH ----------
        warp = self.anat_dir / "warp_T1_to_MNI_synth.nii"
        warped = self.anat_dir / "T1_warped_to_MNI_synth.nii"

        if warp.exists() and warped.exists():
            print("✓ SynthMorph: already exists — skipping")
        else:
            print("→ Running SynthMorph")
            self._run_synthmorph(t1_n4)

    def _run_fastsurfer(self, t1_n4: Path, fs_base_dir: Path):
        """
        Run FastSurfer in a separate conda env, once per subject/day.
        """
        sid = f"{self.cfg.subject_id}_day-{self.cfg.day_id}"
        cmd_str = (
            f"run_fastsurfer.sh "
            f"--t1 {t1_n4} "
            f"--sid {sid} "
            f"--sd {fs_base_dir} "
            "--threads max --seg_only --3T --device cuda"
        )
        run_in_conda("fastsurfer", cmd_str)

    def _make_brain_masks(self, t1_n4: Path, fastsurfer_dir: Path):
        """
        - Takes aparc+aseg.mgz from FastSurfer
        - Creates brainmask_noCSF_filled.nii
        - Skullstrips T1_N4 via SynthStrip
        - Resamples brainmask_noCSF_filled to T1 brain mask space
        - Multiplies to get T1_combined_mask.nii
        """
        sid = f"{self.cfg.subject_id}_day-{self.cfg.day_id}"
        fs_mri_dir = fastsurfer_dir / sid / "mri"
        aparc_aseg = fs_mri_dir / "aseg.auto_noCCseg.mgz"

        brainmask_noCSF = self.anat_dir / "brainmask_noCSF.nii"
        filled = self.anat_dir / "brainmask_noCSF_filled.nii"
        t1_brain = self.anat_dir / "T1_brain.nii"
        t1_mask_skull = self.anat_dir / "T1_mask_skull.nii"
        t1_combined = self.anat_dir / "T1_combined_mask.nii"

        # 1) binarize aparc+aseg into cortical/GM mask without CSF
        if not brainmask_noCSF.exists():
            run([
                "mri_binarize", "--i", str(aparc_aseg),
                "--match",
                # 2 3 7 8 10 11 12 13 16 17 18 26 28
                "2", "3", "7", "8", "10", "11", "12", "13",
                "16", "17", "18", "26", "28",
                # 41 42 46 47 49 50 51 52 53 54 58 60
                "41", "42", "46", "47", "49", "50", "51", "52",
                "53", "54", "58", "60",
                # 77
                "77",
                # 1002 1003 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015
                "1002", "1003", "1005", "1006", "1007", "1008", "1009",
                "1010", "1011", "1012", "1013", "1014", "1015",
                # 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028
                "1016", "1017", "1018", "1019", "1020", "1021", "1022",
                "1023", "1024", "1025", "1026", "1027", "1028",
                # 1029 1030 1031 1034 1035
                "1029", "1030", "1031", "1034", "1035",
                # 2002 2003 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015
                "2002", "2003", "2005", "2006", "2007", "2008", "2009",
                "2010", "2011", "2012", "2013", "2014", "2015",
                # 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 2027 2028
                "2016", "2017", "2018", "2019", "2020", "2021", "2022",
                "2023", "2024", "2025", "2026", "2027", "2028",
                # 2029 2030 2031 2034 2035
                "2029", "2030", "2031", "2034", "2035",
                "--o", str(brainmask_noCSF)
            ])
            run(["fslmaths", str(brainmask_noCSF), "-fillh", str(filled)])
        # 2) SynthStrip on T1_N4
        if not t1_brain.exists() or not t1_mask_skull.exists():
            run([
                "mri_synthstrip",
                "-i", str(t1_n4),
                "-o", str(t1_brain),
                "-m", str(t1_mask_skull)
            ])

        # 3) Resample brainmask_noCSF_filled to T1 skullstrip mask space,
        #    then intersect to get combined mask.
        if not t1_combined.exists():
            brainmask_resampled = self.anat_dir / "brainmask_resampled_to_T1.nii"
            run([
                "flirt",
                "-in", str(filled),
                "-ref", str(t1_mask_skull),
                "-applyxfm", "-usesqform",
                "-out", str(brainmask_resampled)
            ])
            run([
                "fslmaths", str(brainmask_resampled),
                "-mul", str(t1_mask_skull),
                str(t1_combined)
            ])

    def _run_synthmorph(self, t1_n4: Path):
        warp = self.anat_dir / "warp_T1_to_MNI_synth.nii"
        warped = self.anat_dir / "T1_warped_to_MNI_synth.nii"
        if warp.exists() and warped.exists():
            return
        run([
            "mri_synthmorph", "-g",
            "-t", str(warp),
            "-o", str(warped),
            str(t1_n4),
            str(self.cfg.mni_template)
        ])

    # ---------- Fieldmap / AP-PA ----------

    def _prepare_fieldmap(self):
        ensure_dir(self.fmap_dir)
        ap = self.cfg.ap_file
        pa = self.cfg.pa_file
        ap_mean_gz = self.fmap_dir / "AP_mean.nii.gz"
        pa_mean_gz = self.fmap_dir / "PA_mean.nii.gz"

        if not ap_mean_gz.exists():
            run(["fslmaths", str(ap), "-Tmean", str(ap_mean_gz)])
        if not pa_mean_gz.exists():
            run(["fslmaths", str(pa), "-Tmean", str(pa_mean_gz)])

        gunzip_python(ap_mean_gz)
        gunzip_python(pa_mean_gz)

        ap_mean = self.fmap_dir / "AP_mean.nii"
        pa_mean = self.fmap_dir / "PA_mean.nii"

        ap2pa_warp = self.fmap_dir / "AP2PA_Warped.nii"
        if not ap2pa_warp.exists():
            self._run_ap_pa_ants(ap_mean, pa_mean)

    def _run_ap_pa_ants(self, ap_mean: Path, pa_mean: Path):
        out_prefix = self.fmap_dir / "AP2PA_"
        cmd = [
            "bash", "-lc",
            # wrap in bash so we can export vars easily
            f"""
            export ANTS_USE_GPU=1
            export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$(nproc)
            export OMP_NUM_THREADS=$(nproc)
            antsRegistration \
              --dimensionality 3 \
              --float 1 \
              --verbose 1 \
              --output [{out_prefix}, {self.fmap_dir / 'AP2PA_Warped.nii'}, {self.fmap_dir / 'AP2PA_InverseWarped.nii'}] \
              --interpolation Linear \
              --winsorize-image-intensities [0.005,0.995] \
              --use-histogram-matching 1 \
              --initial-moving-transform [{pa_mean}, {ap_mean}, 1] \
              \
              --transform Affine[0.1] \
              --metric MI[{pa_mean}, {ap_mean}, 1, 64, Regular, 0.2] \
              --convergence [3000x2000x1000x500x250,1e-9,25] \
              --shrink-factors 16x12x8x4x2 \
              --smoothing-sigmas 5x4x3x2x1vox \
              \
              --transform SyN[0.1,3,0] \
              --metric MI[{pa_mean}, {ap_mean}, 1, 64, Regular, 0.2] \
              --convergence [200x150x100x50x20,1e-9,20] \
              --shrink-factors 6x5x4x2x1 \
              --smoothing-sigmas 3x2.5x2x1x0vox \
              \
              --transform SyN[0.05,1,0] \
              --metric CC[{pa_mean}, {ap_mean}, 1, 4] \
              --convergence [100x100x50x10,1e-9,10] \
              --shrink-factors 4x3x2x1 \
              --smoothing-sigmas 2x1.5x1x0vox
            """
        ]
        run(cmd)
        # unzip inverse warp
        inverse_warp_path = self.fmap_dir / "AP2PA_1InverseWarp.nii.gz"
        gunzip_python(inverse_warp_path)


    # ---------- Per-run ----------

    def _prepare_run(self, run_cfg):
        run_dir = run_cfg.epi_file.parent
        ensure_dir(run_dir)

        # 0) Input: 4D epi, already combined outside this script
        epi_4d = run_cfg.epi_file  # e.g. epi_4d.nii

        # 1) Apply AP/PA warp -> epi_unwarped (4D)
        epi_unwarped = run_dir / "epi_unwarped.nii"
        self._unwarp_epi(epi_4d, epi_unwarped)

        # 2) Skullstrip all EPI -> epi_brain (4D) + epi_mask (3D or 4D depending on SynthStrip)
        epi_brain = run_dir / "epi_brain.nii"
        epi_mask = run_dir / "epi_mask.nii"
        self._skullstrip_epi(epi_unwarped, epi_brain, epi_mask)

        # 3–4) Motion correction + means
        epi_mean = run_dir / "epi_unwarped_mean.nii"
        epi_mask_mean = run_dir / "epi_mask_mean.nii"
        self._motion_correct_and_mean(
            epi_unwarped=epi_unwarped,
            epi_mask=epi_mask,
            epi_mean=epi_mean,
            epi_mask_mean=epi_mask_mean,
            n_vols_for_mean=20,  # or None to use all volumes
        )

        # --- NEW: set global RT reference if not yet set ---
        self._maybe_set_rt_reference(epi_mean, epi_mask_mean)

        # 5) EPI->T1 registration using EPI mean + masks
        self._register_epi_to_t1(run_dir, epi_mean, epi_mask_mean)

        # 6) EPI mean -> MNI
        self._warp_epi_mean_to_mni(run_dir, epi_mean)

        # ----- Build ROI on *decoder/template* grid -----
        epi_mni_path = run_dir / "epi_in_MNI.nii"
        epi_mni = nib.load(str(epi_mni_path))

        # This is your template decoder (79x95x79 etc.)

        if self.decoder_template.exists():
            dec_img = nib.load(str(self.decoder_template))

            # resample epi to decoder/template grid
            epi_on_dec = image.resample_to_img(
                epi_mni,
                dec_img,
                interpolation="continuous",
                force_resample=True,
                copy_header=True,
            )

            # ROI will now be in decoder/template space
            roi_txt = run_dir / "ROI_DECODER.txt"
            self.export_decoder_roi_txt(
                stat_img=dec_img,      # decoder on its native/template grid
                reference_img=epi_on_dec,  # mean EPI resampled onto that grid
                out_txt=roi_txt,
                bias_value=0.0,
            )

    def _unwarp_epi(self, epi_4d: Path, out: Path):
        """
        Apply AP->PA warp to a 4D EPI (fieldmap-style unwarping).

        Uses the AP2PA_1Warp.nii + AP2PA_0GenericAffine.mat transforms
        generated by _run_ap_pa_ants.
        """
        if out.exists():
            return

        warp = self.fmap_dir / "AP2PA_1InverseWarp.nii"
        affine = self.fmap_dir / "AP2PA_0GenericAffine.mat"
        PA_mean = self.fmap_dir / "PA_mean.nii"

        if not warp.exists() or not affine.exists():
            raise FileNotFoundError(
                f"Expected fieldmap transforms not found:\n"
                f"  {warp}\n"
                f"  {affine}\n"
                "Make sure _run_ap_pa_ants completed successfully."
            )

        cmd = [
            "bash", "-lc",
            f"""
            export ANTS_USE_GPU=1
            export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$(nproc)
            export OMP_NUM_THREADS=$(nproc)
            antsApplyTransforms -d 3 \
              -e 3 \
              -i {epi_4d} \
              -o {out} \
              -r {PA_mean} \
              -t {warp} \
              -t {affine} --float 1
            """

        ]
        run(cmd)

    def _skullstrip_epi(self, epi_unwarped: Path, brain: Path, mask: Path):
        if brain.exists() and mask.exists():
            return
        run([
            "mri_synthstrip",
            "-i", str(epi_unwarped),
            "-o", str(brain),
            "-m", str(mask)
        ])

    def _motion_correct_and_mean(
            self,
            epi_unwarped: Path,
            epi_mask: Path,
            epi_mean: Path,
            epi_mask_mean: Path,
            n_vols_for_mean: int | None = 20,
    ):
        """
        Motion correction using RTPSpy's RtpVolreg + temporal means.

        Inputs:
          epi_unwarped : 4D EPI after AP/PA unwarping
          epi_mask     : EPI skullstrip mask from mri_synthstrip
        Outputs:
          epi_mc.nii      : motion-corrected EPI (subset of volumes)
          motion.1D           : AFNI-style motion params
          epi_mean            : mean over time of epi_mc
          epi_mask_mean       : mean over time of epi_mask
        """

        # If already computed, skip
        if epi_mean.exists() and epi_mask_mean.exists():
            return

        # Output paths (KEEP EXACT OLD NAMING)
        mc_epi = epi_unwarped.with_name("epi_mc.nii")
        motion_1d = epi_unwarped.with_name("motion.1D")

        # ------------------------------------------------------------------
        #                LOAD INPUT EPI (4D)
        # ------------------------------------------------------------------
        img_4d = nib.load(str(epi_unwarped))
        data_4d = np.asanyarray(img_4d.dataobj)
        affine = img_4d.affine
        header = img_4d.header
        n_total_vols = data_4d.shape[-1]

        # Subset number of volumes (same behavior as AFNI [0..N])
        if n_vols_for_mean is not None:
            n_use = min(n_vols_for_mean, n_total_vols)
        else:
            n_use = n_total_vols

        # Prepare output arrays
        mc_data = np.zeros((*data_4d.shape[:3], n_use), dtype=np.float32)
        motion = np.zeros((n_use, 6), dtype=np.float32)

        # ------------------------------------------------------------------
        #                INITIALIZE RTP VOLREG
        # ------------------------------------------------------------------
        vr = RtpVolreg(regmode='heptic')  # AFNI default interpolation
        vr.ignore_init = 0
        vr.save_proc = False

        # Reference is the first volume
        vr.set_ref_vol(f"{epi_unwarped}[0]")

        # ------------------------------------------------------------------
        #                RUN MOTION CORRECTION VOLUME-BY-VOLUME
        # ------------------------------------------------------------------
        for t in range(n_use):
            vol = data_4d[..., t]
            fmri_img = nib.Nifti1Image(vol, affine)
            fmri_img.set_filename(f"vol_{t:04d}.nii")  # required by RTPSpy internals

            vr.do_proc(fmri_img, vol_idx=t)

            # Extract corrected volume
            mc_vol = np.asanyarray(fmri_img.dataobj)
            mc_data[..., t] = mc_vol.astype(np.float32)

            # Extract motion params (already re-ordered inside RtpVolreg)
            motion[t] = vr._motion[t]

        # ------------------------------------------------------------------
        #                SAVE RESULTS (EXACT SAME NAMES AS OLD CODE)
        # ------------------------------------------------------------------
        nib.save(nib.Nifti1Image(mc_data, affine, header), str(mc_epi))
        np.savetxt(str(motion_1d), motion)

        # ------------------------------------------------------------------
        #                COMPUTE MEANS (EPI + MASK)
        # ------------------------------------------------------------------
        if not epi_mean.exists():
            mean_vol = mc_data.mean(axis=-1)
            nib.save(nib.Nifti1Image(mean_vol, affine, header), str(epi_mean))

        if epi_mask.exists() and not epi_mask_mean.exists():
            mask_img = nib.load(str(epi_mask))
            mask_data = np.asanyarray(mask_img.dataobj)

            # If mask is 3D → no-op copy
            if mask_data.ndim == 3:
                nib.save(mask_img, str(epi_mask_mean))
            else:
                mask_mean = mask_data[..., :n_use].mean(axis=-1)
                nib.save(nib.Nifti1Image(mask_mean, mask_img.affine, mask_img.header),
                         str(epi_mask_mean))

    def _maybe_set_rt_reference(self, epi_mean: Path, epi_mask_mean: Path):
        """
        Set the global real-time EPI reference if it does not exist yet.

        We use the first run's unwarped+MC'd mean EPI as the reference
        for:
          - RT motion correction (RtpVolreg)
          - epi→T1 registration anchor
        """
        if not self.rt_ref_epi.exists():
            print(f"→ Setting RT reference EPI to {epi_mean.name}")
            # Use Python copy so we don't assume 'cp' exists
            import shutil
            shutil.copyfile(epi_mean, self.rt_ref_epi)

            if epi_mask_mean.exists():
                shutil.copyfile(epi_mask_mean, self.rt_ref_mask)


    def _register_epi_to_t1(self, run_dir: Path, epi_mean: Path, epi_mask_mean: Path):
        t1_n4 = self.anat_dir / "T1_N4.nii"
        t1_combined = self.anat_dir / "T1_combined_mask.nii"

        out_prefix = self.trans_dir / "epi2t1_"
        warped = self.trans_dir / "epi2t1_Warped.nii"
        inv_warped = self.trans_dir / "epi2t1_InverseWarped.nii"
        composite = self.trans_dir / "epi2t1_Composite.h5"

        # If transforms already exist → reuse, do nothing
        if warped.exists() and inv_warped.exists() and composite.exists():
            print("✓ epi→T1 transform already exists — skipping antsRegistration")
            return

        # Otherwise, run registration ONCE using this run’s epi_mean/mask
        ensure_dir(self.trans_dir)
        print("→ Running epi→T1 antsRegistration (global)")

        cmd = [
            "bash", "-lc",
            f"""
            export ANTS_USE_GPU=1
            export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$(nproc)
            export OMP_NUM_THREADS=$(nproc)
            antsRegistration \
              --dimensionality 3 \
              --float 1 \
              --verbose 1 \
              --output [{out_prefix}, {warped}, {inv_warped}] \
              --interpolation Linear \
              --winsorize-image-intensities [0.005,0.995] \
              --use-histogram-matching 0 \
              --write-composite-transform 1 \
              --initial-moving-transform [{epi_mean}, {t1_n4}, 1] \
              --masks [{epi_mask_mean}, {t1_combined}] \
              \
              --transform Affine[0.1] \
              --metric MI[{t1_n4}, {epi_mean}, 1, 32, Regular, 0.2] \
              --convergence [1000x500x250x100,1e-7,15] \
              --shrink-factors 8x4x2x1 \
              --smoothing-sigmas 3x2x1x0vox \
              \
              --transform SyN[0.1,1,0] \
              --metric MI[{t1_n4}, {epi_mean}, 1, 32, Regular, 0.2] \
              --convergence [100x70x50x20,1e-7,15] \
              --shrink-factors 6x4x2x1 \
              --smoothing-sigmas 3x2x1x0vox
            """
        ]
        run(cmd)
        # Copy geometry from T1 (helps keep everything aligned)
        run(["fslcpgeom", str(t1_n4), str(warped)])
        run(["fslcpgeom", str(t1_n4), str(inv_warped)])

    def _warp_epi_mean_to_mni(self, run_dir: Path, epi_mean: Path):
        mni = self.cfg.mni_template
        warp_t1_mni = self.anat_dir / "warp_T1_to_MNI_synth.nii"
        composite = self.trans_dir / "epi2t1_Composite.h5"
        out_epi_mni = self.trans_dir / "epi_in_MNI.nii"

        if out_epi_mni.exists():
            print(f"✓ epi_in_MNI already exists for {self.trans_dir.name} — skipping warp")
            return

        if not self.decoder_template.exists():
            raise FileNotFoundError(
                f"Decoder template not found at {self.decoder_template} "
                f"(expected relative to {self.cfg.root})"
            )

        if not composite.exists():
            raise FileNotFoundError(
                f"EPI→T1 composite transform not found at {composite}. "
                f"Make sure _register_epi_to_t1 ran at least once."
            )

        cmd = [
            "bash", "-lc",
            f"""
            export ANTS_USE_GPU=1
            export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$(nproc)
            export OMP_NUM_THREADS=$(nproc)
            antsApplyTransforms \
              -d 3 \
              -i {epi_mean} \
              -r {self.decoder_template} \
              -o {out_epi_mni} \
              -t {warp_t1_mni} \
              -t {composite} \
              -n Linear --float 1
            """
        ]
        #-r {mni} \
        run(cmd)

    def qc_plot_epi_in_mni(
            self,
            run_dir: Path,
            decoder_path: Path | None = None,
            out_png: Path | None = None,
            dpi: int = 300,
    ):
        """High-quality QC plot."""

        epi_mni = self.trans_dir / "epi_in_MNI.nii"
        if not epi_mni.exists():
            raise FileNotFoundError(f"epi_in_MNI not found at {epi_mni}")

        if out_png is None:
            out_png = self.trans_dir / "qc_epi_in_MNI.png"

        try:
            import nibabel as nib
            import matplotlib.pyplot as plt
            from nilearn import plotting, image
        except ImportError:
            log.warning("Nilearn not installed; skipping QC plot.")
            return

        epi_img = nib.load(str(epi_mni))
        mni_template = nib.load(str(self.cfg.mni_template))

        # Default background
        bg_img = mni_template
        stat_img = epi_img

        # Overlay decoder if provided
        if decoder_path is not None and Path(decoder_path).exists():
            dec_img = nib.load(str(decoder_path))
            dec_resampled = image.resample_to_img(
                dec_img, epi_img,
                interpolation="continuous",
                force_resample=True,
                copy_header=True
            )
            bg_img = epi_img
            stat_img = dec_resampled

        # Auto cut coordinates
        cut_coords = plotting.find_xyz_cut_coords(stat_img)

        # High-quality figure
        fig = plt.figure(figsize=(9, 9), dpi=dpi)
        display = plotting.plot_stat_map(
            stat_img,
            bg_img=bg_img,
            display_mode="ortho",
            cut_coords=cut_coords,
            threshold=None,
            annotate=True,
            axes=fig.add_subplot(111),
            title="EPI in MNI (and decoder)" if decoder_path else "EPI in MNI",
        )

        display.savefig(str(out_png), dpi=dpi)
        plt.close(fig)

    def export_decoder_roi_txt(
            self,
            stat_img,
            reference_img,
            out_txt: Path,
            bias_value: float | None = None,
    ):
        """
        Export ROI in the same format as the MATLAB ROI_NPS.txt script.

        stat_img      = nibabel image (decoder in MNI or EPI space)
                        (equivalent to `decoder` in MATLAB)
        reference_img = nibabel image (usually epi_mean / wmeanVol)
                        (equivalent to `wmeanVol` in MATLAB)
        out_txt       = output txt path (e.g. ROI_DECODER.txt)
        bias_value    = final bias term; if None -> 0.0

        File format (same as MATLAB):
          line 1: nx ny nz
          line 2: N_voxels
          next N lines: x y z decoder(x,y,z) wmeanVol(x,y,z)
          last line: 0 0 0 bias 0
        """

        import numpy as np

        dec = np.asanyarray(stat_img.get_fdata())
        ref = np.asanyarray(reference_img.get_fdata())

        # ---- voxel selection: same logic as MATLAB ----
        # MATLAB: find(decoder ~= 0 & ~isnan(decoder))
        # Here we treat tiny values as non-zero as well,
        # to reproduce lines that *print* as 0.000000 but are actually ~1e-7.
        eps = 1e-12
        mask_nonzero = np.logical_and(np.abs(dec) > eps, ~np.isnan(dec))
        coords = np.array(np.where(mask_nonzero)).T  # 0-based indices (i,j,k)

        nx, ny, nz = dec.shape

        with open(out_txt, "w") as f:
            # Header: dimensions
            f.write(f"{nx} {ny} {nz}\n")

            # Number of voxels
            f.write(f"{coords.shape[0]}\n")

            # Each voxel: x y z decoder(x,y,z) wmeanVol(x,y,z)
            # IMPORTANT: +1 to go from 0-based (Python) to 1-based (MATLAB / ROI_NPS.txt)
            for (i, j, k) in coords:
                x = i + 1
                y = j + 1
                z = k + 1
                f.write(
                    f"{x} {y} {z} "
                    f"{dec[i, j, k]:.6f} {ref[i, j, k]:.6f}\n"
                )

            # Final bias line (MATLAB: if exist(bias) ... else ... end)
            if bias_value is None:
                bias_value = 0.0
            f.write(f"0 0 0 {bias_value:.6f} 0\n")



