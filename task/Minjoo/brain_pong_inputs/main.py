from __future__ import annotations

import os
import yaml
from pathlib import Path
import argparse

import pandas as pd

from .utils import load_defaults_params, get_game_params, load_defaults_levels

WIN_SIZE = [1200, 800]

def handle_game_config(config: str | Path | None):
    params = load_defaults_params()
    if config is None:
        return params

    with open(config, "r") as f:
        custom = yaml.safe_load(f)

    params.update(custom)
    return params


def handle_levels(levels: str | Path | None):
    level_input = load_defaults_levels()
    if not levels:
        return level_input

    with open(levels, "r") as f:
        level_configs = yaml.safe_load(f)
    
    level_input.update(level_configs)
    return level_input


def get_next_level(
    current_level: int,
    player_won,
    max_level: int,
) -> int:
    """1-up/1-down. player_won: True=up, False=down, None=hold."""
    if player_won is True:
        return min(current_level + 1, max_level)
    if player_won is False:
        return max(current_level - 1, 0)
    return current_level


def load_resume_state(
    resume_from: str | Path,
    max_level: int,
) -> int:
    """Load starting level from the last row of a block summary."""
    resume_path = Path(resume_from)
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume file not found: {resume_path}")

    summary = pd.read_csv(resume_path, sep="\t")
    if summary.empty:
        raise ValueError(f"Resume file has no trial rows: {resume_path}")

    last_trial = summary.iloc[-1]
    current_level = int(last_trial["diff_level"])
    if current_level > max_level:
        current_level = max_level

    return current_level


def run_experiment(
    save_dir: str | Path,
    input_method: str,
    paddle_response_mode: str | None,
    n_blocks: int,
    n_trials: int,
    levels: str | Path | None = None,
    config: str | Path | None = None,
    start_block: int = 1,
    resume_from: str | Path | None = None,
    overwrite: bool = False,
    keyboard_delay_sec: float | None = None,
    window_backend: str | None = None,
):
    """Top-level function that runs the pong task in its entirety

    Args:
        save_dir: Output directory for experiment data.
        n_blocks: Number of trial blocks to run.
        n_trials: Number of trials per block.
        input_method: input method for response
        paddle_response_mode: paddle response mode for press/wheel input.
        levels: Path to YAML file with level configurations.
        config: Path to YAML file with runtime parameter overrides.
        start_block: First block number to run. Block numbering is 1-based.
        resume_from: Previous block summary TSV to continue level from.
        overwrite: Allow overwriting existing block output files.
    """
    if start_block < 1:
        raise ValueError("start_block must be 1 or greater")

    blocks = range(start_block, start_block + n_blocks)

    print("[BrainPong] Loading configuration...", flush=True)
    params = handle_game_config(config)
    levels = handle_levels(levels)

    os.makedirs(save_dir, exist_ok=True)

    current_level = params["init_level"]
    total_points = params["init_points"]
    if resume_from is not None:
        current_level = load_resume_state(
            resume_from,
            max(levels.keys()),
        )

    if not overwrite:
        for b in blocks:
            block_file = Path(save_dir, f"block{b:02d}.tsv")
            if block_file.exists():
                raise FileExistsError(
                    f"{block_file} already exists. Use --overwrite to replace it."
                )

    game_params = get_game_params(params)
    game_params["input_method"] = input_method
    if keyboard_delay_sec is not None:
        game_params["keyboard_delay_sec"] = keyboard_delay_sec
    if paddle_response_mode is not None:
        game_params["paddle_response_mode"] = paddle_response_mode

    backend = window_backend or params.get("win_type", "pyglet")
    refresh_rate = params.get("refresh_rate")

    # Import the GUI modules here so the terminal can identify an import-time
    # stall separately from a window-creation stall on macOS.
    print("[BrainPong] Importing PsychoPy visual modules...", flush=True)
    from psychopy import visual, core
    from .block import TrialBlock

    print(
        f"[BrainPong] Creating window "
        f"(backend={backend}, fullscr={bool(params.get('fullscr', True))})...",
        flush=True,
    )
    try:
        win = visual.Window(
            size=params["win_size"],
            fullscr=bool(params.get("fullscr", True)),
            winType=backend,
            color="black",
            units="pix",
            allowGUI=True,
            checkTiming=False,
        )
    except Exception as exc:
        print(
            f"[BrainPong] Window creation failed with {backend}: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        raise

    print("[BrainPong] Window created.", flush=True)
    handle = getattr(win, "winHandle", None)
    activate = getattr(handle, "activate", None)
    if callable(activate):
        activate()

    try:
        for b in blocks:
            print(f"[BrainPong] Preparing block {b}...", flush=True)
            trial_block = TrialBlock(
                b,
                n_trials,
                save_dir,
                levels,
                current_level,
                total_points,
                params["block_pre_delay"],
                params["performance_criteria"],
                win=win,
                game_params=game_params,
                use_aggressive_leveling=params["use_aggressive_leveling"],
                refresh_rate=refresh_rate,
            )
            completed = trial_block.run()

            if not completed:
                print("[BrainPong] Experiment ended early by user.", flush=True)
                break

            total_points = trial_block.total_points
            current_level = trial_block.current_level
    finally:
        win.close()

    print("[BrainPong] DONE!", flush=True)
    core.quit()


def handler(args: argparse.Namespace) -> int:

    params = load_defaults_params()
    game_params = get_game_params(params)

    game_params["input_method"] = args.input_method

    run_experiment(
        save_dir=args.save_dir,
        input_method=args.input_method,
        paddle_response_mode=args.paddle_response_mode,
        n_blocks=args.blocks,
        n_trials=args.trials,
        levels=args.levels,
        config=args.config,
        start_block=args.start_block,
        resume_from=args.resume_from,
        overwrite=args.overwrite,
        keyboard_delay_sec=args.keyboard_delay_sec,
        window_backend=args.window_backend,
    )
    return 0


def create_parser(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:

    #parser = argparse.ArgumentParser(...)
    #subparsers = parser.add_subparsers()
    
    parser = parser or argparse.ArgumentParser()
    #positional = _parser.add_argument_group("positional arguments")
    #required = _parser.add_argument_group("required parameters")
    #optional = _parser.add_argument_group("optional parameters")

    #run = subparsers.add_parser("run", help="run experiment")
   
    parser.add_argument(
        "save_dir", type=Path, metavar="OUT_DIR", help="Output directory"
    )

    parser.add_argument(
        "--input_method", choices=["keyboard", "mouse", "press", "wheel", "brain"],
        default="keyboard", help="Default: keyboard (1=TOP, 2=BOTTOM, delayed by 6 seconds)"
    )
    parser.add_argument("--keyboard-delay-sec", type=float, default=None,
                        help="Fixed behavioral key delay in seconds (default: 6); NF is unaffected")
    parser.add_argument(
        "--window-backend",
        choices=["pyglet", "glfw"],
        default=None,
        help="PsychoPy window backend (overrides win_type in the YAML config)",
    )

    parser.add_argument(
        "--paddle_response_mode",
        type=str,
        choices=["hrf", "hold"],
        default=None,
        help=(
            "Legacy compatibility option. Keyboard uses --keyboard-delay-sec; "
            "NF receives no additional delay or HRF convolution."
        ),
    )

    parser.add_argument(
        "--blocks", "-b", type=int, required=True, help="Set the number of trial blocks"
    )

    parser.add_argument(
        "--trials",
        "-t",
        type=int,
        required=True,
        help="Set the number of trials per block",
    )

    parser.add_argument(
        "--start_block",
        type=int,
        default=1,
        help="First block number to run. Block and trial numbering starts at 1.",
    )

    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help=(
            "Path to a previous block summary TSV. The next run starts with "
            "that block's last saved level; points reset to init_points."
        ),
    )

    parser.add_argument(
        "--levels",
        "-l",
        type=str,
        help="Path to a YAML file containing level configurations. If not "
        "provided, uses the default binary-pilot level table. "
        "Only opponent miss probability changes with level.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to a YAML file containing runtime parameter configurations "
        "Any key in the file overwrites default configuration. If not provided, "
        "then default configuration is used",
    )
    parser.add_argument(
        "--overwrite", "-o", action="store_true", help="Allow overwriting block output files"
    )

    parser.set_defaults(handler=handler)
    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    if hasattr(args, "handler"):
        return args.handler(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    main()
