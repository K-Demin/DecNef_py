import copy
from importlib import resources
import yaml
from pathlib import Path
import logging


def load_defaults_params():
    """Load default experiment parameters from config file.

    Returns:
        Dictionary of default parameters.
    """
    default_file = resources.files(f"{__package__}.config") / "defaults_delay.yaml"

    with open(default_file, "r") as file:
        params = yaml.safe_load(file)
    return params


def get_game_params(params: dict) -> dict:
    """Extract game-specific parameters by removing experiment- and block-level
    settings.

    Args:
        params: Full parameter dictionary.

    Returns:
        Dictionary with experiment-level parameters removed.
    """
    game_params = copy.deepcopy(params)

    var_list = [
        "win_size",
        "fullscr",
        "init_level",
        "init_points",
        "performance_criteria",
        "use_aggressive_leveling",
        "block_pre_delay",
    ]
    [game_params.pop(i) for i in var_list]

    return game_params

def load_defaults_levels():
    """Load default experiment leveling system from config file.

    Returns:
        Dictionary of level parameters.
    """
    default_file = resources.files(f"{__package__}.config") / "levels.yaml"

    with open(default_file, "r") as file:
        level_input = yaml.safe_load(file)
    return level_input

def setup_logging(log_file: str | Path | None = None):
    """Configure logging for the experiment.

    Args:
        log_file: Path to log file. If None, only logs to console.

    Returns:
        Configured logger instance.
    """
    handlers = [logging.StreamHandler()]

    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, mode="w"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=handlers,
        force=True,
    )

    logger = logging.getLogger(__name__)
    return logger


def get_window_info(win, logger):
    """Get window configuration and frame rate.

    Args:
        win: PsychoPy window instance.
        logger: Logger instance for output.

    Returns:
        Tuple of (configured_size, actual_size, fps).
    """
    configured = win.winHandle.get_size()
    actual = win.size

    is_hidef = actual[0] != configured[0] or actual[1] != configured[1]
    scale_factor = actual[0] / configured[0] if configured[0] > 0 else 1.0
    scaling_label = "retina/hidpi" if is_hidef else "standard"

    logger.info(f"configured window size: {configured[0]}x{configured[1]}")
    logger.info(f"actual window size: {actual[0]}x{actual[1]}")
    logger.info(f"screen scaling: {scale_factor:.2f}x ({scaling_label})")

    fps = win.getActualFrameRate()
    if fps is None or fps <= 0:
        fps = 60.0
        logger.info(f"could not measure frame rate, using default: {fps} Hz")
    else:
        logger.info(f"using measured frame rate: {fps:.1f} Hz")

    return configured, actual, fps
