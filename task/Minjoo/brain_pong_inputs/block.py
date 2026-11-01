from __future__ import annotations
import os
from pathlib import Path
from typing import Callable
import pandas as pd
from psychopy import visual

from .utils import setup_logging, get_window_info
from .pong import Pong
from .binary import BalancedTargets


WIN_SIZE = [1200, 800]


def save_block_data(data, fname):
    """Save block summary data to TSV file.

    Args:
        data (list): List of trial data dictionaries.
        fname (str | Path): Output file path.
    """
    df = pd.DataFrame(data)
    df = df.astype(
        {
            "block_number": "uint16",
            "trial_number": "uint16",
            "diff_level": "uint16",
            "paddle_size": "float32",
            "rallies": "uint16",
            "start_points": "uint16",
            "end_points": "uint16",
            "agent_points": "uint16",
            "total_hits": "uint16",
            "total_possible": "uint16",
            "performance": "float32",
            "agent_miss_prob": "float32",
            "n_rally_wins": "uint16",
            "n_rally_total": "uint16",
        }
    )
    df.to_csv(fname, index=False, sep="\t")


class TrialBlock:
    """Run block of of pong trials

    Args:
        block_num (int): Block number.
        n_trials (int): Number of trials in block.
        data_dir (str | Path): Directory to save data.
        level_dict (dict, optional): Difficulty level configurations.
        init_level (int): Starting difficulty level.
        init_points (int): Starting point value.
        block_pre_delay (int, optional): Extended pre-delay for first trial.
        performance_criteria (float): Performance threshold for level advancement.
        win (visual.Window, optional): PsychoPy window.
        game_params (dict, optional): Game configuration parameters.
        block_instruction (str, optional): Instruction text for block.
        use_aggressive_leveling (bool): Whether to decrease level on poor performance.

    Attributes:
        block_num (int): Block number.
        total_points (int): Current total points across trials.
        current_level (int): Current difficulty level.
    """

    def __init__(
        self,
        block_num: int,
        n_trials: int,
        data_dir: str | Path,
        level_dict: dict | None = None,
        init_level: int = 0,
        init_points: int = 0,
        block_pre_delay: int | None = None,
        performance_criteria: float = 0.6,
        win: visual.Window | None = None,
        game_params: dict | None = None,
        block_instruction: str | None = None,
        use_aggressive_leveling: bool = False,
        game_runner: Callable | None = None,
        show_instructions: bool = True,
        refresh_rate: float | None = None,
    ):
        # params
        self.block_num = block_num
        self.n_trials = n_trials
        self.data_dir = data_dir
        self.level_dict = level_dict
        self.init_level = init_level
        self.init_points = init_points
        self.block_pre_delay = block_pre_delay
        self.performance_criteria = performance_criteria
        self.use_aggressive_leveling = use_aggressive_leveling
        self.game_runner = game_runner
        self.show_instructions = show_instructions

        os.makedirs(data_dir, exist_ok=True)

        # log
        self.logger = setup_logging(
            log_file=Path(data_dir, f"block{block_num:02d}.log")
        )

        # configured params
        self._apply_levels = self.level_dict is not None
        self._max_level = None
        if self._apply_levels:
            self._max_level = max(self.level_dict.keys())

        if win:
            self.win = win
        else:
            self.win = visual.Window(
                size=WIN_SIZE,
                fullscr=False,
                winType="pyglet",
                color="black",
                units="pix",
            )

        # config block instructions
        self.block_instruction = None
        if block_instruction:
            self.block_instruction = block_instruction
        self.block_instruction = f"\nBLOCK: {block_num} POINTS: {init_points}\n"

        self.game_params = game_params
        if game_params and game_params.get("input_method") == "keyboard":
            delay = float(game_params.get("keyboard_delay_sec", 6.0))
            self.block_instruction += (
                f"\n1: TOP    2: BOTTOM\nResponses appear {delay:g} seconds after each key press.\n"
                "Press SPACE to start.\n"
            )

        _, _, measured_refresh_rate = get_window_info(
            self.win, self.logger, refresh_rate=refresh_rate
        )
        self.refresh_rate = measured_refresh_rate

        self.logger.info("=== PARAMS: ===")
        self.logger.info(f"block_num: {self.block_num}")
        self.logger.info(f"n_trials: {self.n_trials}")
        self.logger.info(f"data_dir: {self.data_dir}")
        self.logger.info(f"apply_levels: {self._apply_levels}")
        self.logger.info(f"init_level: {self.init_level}")
        self.logger.info(f"init_points: {self.init_points}")
        self.logger.info(f"performance_criteria: {self.performance_criteria}")
        self.logger.info(f"use_aggressive_leveling: {self.use_aggressive_leveling}")
        self.logger.info(f"refresh_rate: {self.refresh_rate}")

    def _setup_levels(self):
        self._apply_levels = self.level_dict is not None

        if self._apply_levels:
            self.block_instruction += f"LEVEL: {self.init_level}\n"

    def _adjust_difficulty(self, player_won):
        if not self._apply_levels:
            return

        if player_won is True:
            self.current_level = min(self.current_level + 1, self._max_level)
        elif player_won is False:
            self.current_level = max(self.current_level - 1, 0)

    def run(self):
        """Run all trials in the block.

        Returns:
            bool: True if block completed, False if user escaped.
        """
        self._setup_levels()

        self.total_points = self.init_points
        self.current_level = self.init_level
        block_data = []
        target_schedule = self.game_params.get("target_schedule") or BalancedTargets()

        trial_pfx = f"block{self.block_num:02d}_trial"

        for trial in range(1, self.n_trials + 1):
            self.logger.info(f"=== NEW TRIAL: {trial}/{self.n_trials} ===")
            game_params = self.game_params.copy()
            game_params["target_schedule"] = target_schedule
            game_params["run_id"] = game_params.get("run_id", str(self.block_num))
            game_params["trial_id"] = trial
            trial_start_points = 0

            if self._apply_levels:
                game_params["level"] = self.current_level
                game_params["agent_miss_prob"] = self.level_dict[self.current_level][
                    "agent_miss_prob"
                ]

            if self.block_pre_delay and (trial == 1):
                game_params["pre_delay"] = self.block_pre_delay

            game = Pong(
                self.win,
                self.logger,
                save_file=Path(self.data_dir, f"{trial_pfx}{trial:02d}.tsv"),
                init_points=trial_start_points,
                **game_params,
            )
            if self.game_runner is None:
                completed = game.run(
                    instruction_text=self.block_instruction if trial == 1 else None,
                    show_instructions=self.show_instructions and trial == 1,
                    refresh_rate=self.refresh_rate,
                )
            else:
                completed = self.game_runner(
                    game,
                    self.block_num,
                    trial,
                    self.refresh_rate,
                )

            if game.player_points > game.agent_points:
                player_won = True
            elif game.player_points < game.agent_points:
                player_won = False
            else:
                player_won = None

            start_level = game_params["level"] if self._apply_levels else self.current_level
            agent_miss_prob = (
                self.level_dict[start_level]["agent_miss_prob"]
                if self._apply_levels
                else None
            )
            self._adjust_difficulty(player_won)

            data = {
                "block_number": self.block_num,
                "trial_number": trial,
                "diff_level": self.current_level,
                "paddle_size": game_params["paddle_size"],
                "start_time": game.start_time,
                "rallies": game.rally,
                "start_points": trial_start_points,
                "end_points": game.player_points,
                "agent_points": game.agent_points,
                "player_won": player_won,
                "agent_miss_prob": agent_miss_prob,
                "n_rally_wins": sum(1 for w in game.rally_outcomes if w),
                "n_rally_total": len(game.rally_outcomes),
                "total_possible": game.total_possible,
                "total_hits": game.total_hits,
                "performance": game.performance,
                "participant_hits": game.participant_hits,
                "participant_misses": game.participant_misses,
                "participant_hit_rate": game.performance,
                "opponent_hits": game.agent_hits,
                "opponent_misses": game.agent_misses,
                "participant_score": game.player_points,
                "opponent_score": game.agent_points,
                "round_result": "WIN" if player_won is True else ("LOSS" if player_won is False else "DRAW"),
                "level_before": start_level,
                "level_after": self.current_level,
            }
            block_data.append(data)

            # Kept for compatibility with callers, but each new trial starts at 0.
            self.total_points = game.player_points

            if not completed:
                self.logger.info("esc - ending experiment")
                break

        save_block_data(
            block_data, Path(self.data_dir, f"block{self.block_num:02d}.tsv")
        )
        self.block_data = block_data

        self.logger.info("Block complete")

        return completed
