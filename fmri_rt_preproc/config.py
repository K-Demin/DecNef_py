from dataclasses import dataclass, asdict
from pathlib import Path
import json
import os

@dataclass
class RunConfig:
    id: str
    epi_file: Path

@dataclass
class SubjectDayConfig:
    subject_id: str
    day_id: str
    root: Path
    subject_root: Path
    ap_file: Path
    pa_file: Path
    t1_file: Path
    mni_template: Path
    runs: list[RunConfig]
    project_root = Path(__file__).resolve().parents[1]
    decoder_template = project_root / "decoders" / "rweights_NSF_grouppred_cvpcrTMP_nonzeros.nii"

    @classmethod
    def from_json(cls, path: str | Path) -> "SubjectDayConfig":
        cfg = json.loads(Path(path).read_text())
        root = Path(cfg["root"])
        subject_root = Path(cfg.get("subject_root", root.parent))
        runs = [
            RunConfig(id=r["id"], epi_file=root / r["epi_file"])
            for r in cfg["runs"]
        ]
        return cls(
            subject_id=cfg["subject_id"],
            day_id=cfg["day_id"],
            root=root,
            subject_root=subject_root,
            ap_file=root / "fmap" / "AP.nii",
            pa_file=root / "fmap" / "PA.nii",
            t1_file=subject_root / "anat" / "T1.nii",
            mni_template=Path(cfg["templates"]["mni_t1"]),
            runs=runs,
        )

    @classmethod
    def auto_from_layout(
        cls,
        subject_id: str,
        day_id: str,
        root: Path,
        epi_basename: str = "epi.nii",
    ) -> "SubjectDayConfig":
        """
        Build config from a standard layout:

        subject_root/
          anat/T1.nii.gz
        root/
          fmap/AP.nii.gz, PA.nii.gz
          func/run-XX/epi_basename
        """
        # Infer MNI template from FREESURFER_HOME
        fs_home = os.environ.get("FREESURFER_HOME")
        if fs_home is None:
            raise RuntimeError(
                "FREESURFER_HOME is not set. Can't infer MNI template path."
            )
        mni_template = (
            Path(fs_home)
            / "average"
            / "mni_icbm152_nlin_asym_09c"
            / "mni_icbm152_t1_tal_nlin_asym_09c.nii.gz"
        )

        func_root = root / "func"
        if not func_root.exists():
            raise FileNotFoundError(f"func directory not found: {func_root}")

        runs: list[RunConfig] = []
        for run_dir in sorted(func_root.glob("run-*")):
            epi_path = run_dir / epi_basename
            if not epi_path.exists():
                raise FileNotFoundError(
                    f"EPI file not found: {epi_path} "
                    f"(expected basename: {epi_basename})"
                )
            runs.append(RunConfig(id=run_dir.name, epi_file=epi_path))

        return cls(
            subject_id=subject_id,
            day_id=day_id,
            root=root,
            subject_root=root.parent,
            ap_file=root / "fmap" / "AP.nii.gz",
            pa_file=root / "fmap" / "PA.nii.gz",
            t1_file=root.parent / "anat" / "T1.nii.gz",
            mni_template=mni_template,
            runs=runs,
        )

    @classmethod
    def for_single_run(
        cls,
        subject_id: str,
        day_id: str,
        root: Path,
        run_id: str,
        epi_file: Path,
    ) -> "SubjectDayConfig":
        """
        Build config for a SINGLE run, where epi_file is already a merged 4D EPI.

        Layout:
            subject_root/
              anat/T1.nii
            root/
              fmap/AP.nii, PA.nii
              func/<run_id>/epi_4d.nii.gz
        """
        import os

        # Infer MNI template from FREESURFER_HOME (same as in auto_from_layout)
        fs_home = os.environ.get("FREESURFER_HOME")
        if fs_home is None:
            raise RuntimeError(
                "FREESURFER_HOME is not set. Can't infer MNI template path."
            )
        mni_template = (
            Path(fs_home)
            / "average"
            / "mni_icbm152_nlin_asym_09c"
            / "mni_icbm152_t1_tal_nlin_asym_09c.nii.gz"
        )

        runs = [RunConfig(id=run_id, epi_file=epi_file)]

        return cls(
            subject_id=subject_id,
            day_id=day_id,
            root=root,
            subject_root=root.parent,
            ap_file=root / "fmap" / "AP.nii",
            pa_file=root / "fmap" / "PA.nii",
            t1_file=root.parent / "anat" / "T1.nii",
            mni_template=mni_template,
            runs=runs,
        )


    def to_json(self, path: str | Path):
        """
        Save a simple JSON config so we can re-use it or edit manually.

        We store relative epi_file paths and template path.
        """
        root = self.root
        cfg = {
            "subject_id": self.subject_id,
            "day_id": self.day_id,
            "root": str(root),
            "subject_root": str(self.subject_root),
            "templates": {
                "mni_t1": str(self.mni_template),
            },
            "runs": [
                {
                    "id": r.id,
                    "epi_file": str(r.epi_file.relative_to(root))
                }
                for r in self.runs
            ],
        }
        path = Path(path)
        path.write_text(json.dumps(cfg, indent=2) + "\n")
