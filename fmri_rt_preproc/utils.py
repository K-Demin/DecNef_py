# utils.py
import subprocess
from pathlib import Path
import logging

log = logging.getLogger("fmri_rt_preproc")

def run(cmd: list[str], env=None, cwd=None):
    """
    Run a command list directly (no conda activation).
    """
    cmd_str = " ".join(cmd)
    log.info(f"Running: {cmd_str}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=cwd,
    )
    log.info(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd_str}\n{result.stdout}")
    return result


def run_in_conda(env_name: str, cmd_str: str):
    """
    Run a shell command string inside a conda env via bash.
    This version explicitly sources conda.sh so that the correct
    Anaconda installation and its environments are visible.
    """
    conda_sh = "/home/sin/anaconda3/etc/profile.d/conda.sh"

    full_cmd = [
        "bash", "-lc",
        f"source {conda_sh} && conda activate {env_name} && {cmd_str}"
    ]

    log.info(f"[conda:{env_name}] {cmd_str}")
    result = subprocess.run(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log.info(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"Conda cmd failed in env '{env_name}': {cmd_str}\n{result.stdout}"
        )
    return result



def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
