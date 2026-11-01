from __future__ import annotations
from pathlib import Path
import logging
import math
import random
from typing import Callable

from psychopy import visual, core, event

import numpy as np
import pandas as pd
from .binary import BalancedTargets, DelayedBinaryKeys, decoder_state, target_position


LOGGER = None



class Pong:
    """Pong game for experimental trials.

    Args:
        win (visual.Window): PsychoPy window.
        logger (logging.Logger): Logger instance.
        input_method (str): Input method ('mouse' or 'brain' or 'key').
        init_points (int): Starting point value.
        point_increment (int): Points awarded per hit.
        level (int): Difficulty level.
        paddle_size (float): Paddle height in normalized units.
        reset_timer (float): Timer for reset delay.
        reset_delay (float): Delay in seconds after miss before reset.
        ball_vel (tuple): Initial ball velocity (x, y).
        ball_radius (float): Ball radius in normalized units.
        save_file (str | Path, optional): Path to save event data.
        time_limit (float): Trial duration in seconds.
        pre_delay (float): Delay before trial starts in seconds.
        send_speed (float, optional): Ball speed when player hits.
        receive_speed (float, optional): Ball speed when opponent hits.
        paddle_lag (int): Legacy option; binary input bypasses all lag.
        opponent_handicap (float): Opponent speed reduction (0-1).

    Attributes:
        points (int): Current point total.
        rally (int): Current rally number.
        performance (float): Proportion of successful hits.
        total_hits (int): Number of successful paddle hits.
        total_possible (int): Total possible hits.
    """

 

    def __init__(
        self,
        win: visual.Window,
        logger: logging.Logger,
        init_points: int = 0,
        point_increment: int = 1,
        level: int = 0,
        paddle_size: float = 0.3,
        reset_timer: float = 0,
        reset_delay: float = 2.0,
        ball_vel: tuple = (0.008, 0.006),
        ball_radius: float = 0.03,
        save_file: str | Path | None = None,
        time_limit: float = 60.0,
        pre_delay: float = 0.0,
        send_speed: float | None = None,
        receive_speed: float | None = None,
        paddle_lag: int = 360,
        opponent_handicap: float = 0,
        agent_miss_prob: float = 0.2,
        agent_miss_mode: str = "random",
        agent_miss_offset: float = 0.15,
        agent_tracking_noise: float = 0.0,
        step_size: float = 1.0, # in percent?
        intended_y: float = 0.0,
        intended_shift: float = 0.0,
        input_method: str = "wheel",
        paddle_response_mode: str = "hrf",
        brain_controller: Callable[[float], float] | None = None,
        frame_callback: Callable | None = None,
        idle_callback: Callable | None = None,
        rally_callback: Callable | None = None,
        contact_callback: Callable | None = None,
        target_schedule: BalancedTargets | None = None,
        run_id: str = "",
        trial_id: int = 0,
        keyboard_delay_sec: float = 6.0,
        rng: random.Random | None = None,
    ):
        # params
        self.logger = logger
        self.win = win
        self.rng = rng if rng is not None else random

        # Validate input method
        if input_method not in ["mouse", "brain", "press", "wheel", "keyboard"]:
            raise ValueError(
                f"input_method must be mouse/brain/press/wheel/keyboard, got '{input_method}'"
            )

        if paddle_response_mode not in ["hrf", "hold"]:
            raise ValueError(
                "paddle_response_mode must be 'hrf' or 'hold', "
                f"got '{paddle_response_mode}'"
            )

        self.input_method = input_method
        self.paddle_response_mode = paddle_response_mode
        self.keyboard_controller = DelayedBinaryKeys(keyboard_delay_sec)
        self.brain_controller = brain_controller
        self.frame_callback = frame_callback
        self.idle_callback = idle_callback
        self.brain_position = 0.0
        self.z_score = 0.0
        self.decoder_state = decoder_state(self.z_score)
        self.target_schedule = target_schedule or BalancedTargets()
        self.contact_callback = contact_callback
        self.run_id = run_id
        self.trial_id = trial_id
        self.contacts = []
        self.participant_hits = 0
        self.participant_misses = 0
        self.approach_id = 0
        self.refresh_rate = 60.0
        if self.input_method not in ("brain", "keyboard"):
            self.mouse = event.Mouse(win=win)

        self.reset_timer = reset_timer
        self.reset_delay = reset_delay
        self.point_increment = point_increment
        self.time_limit = time_limit
        self.pre_delay = pre_delay
        self.paddle_lag = paddle_lag
        self.level = level
        self.opponent_handicap = 1 - opponent_handicap
        self.agent_miss_prob = float(agent_miss_prob)
        if agent_miss_mode not in ["random", "adaptive"]:
            raise ValueError(
                "agent_miss_mode must be 'random' or 'adaptive', "
                f"got '{agent_miss_mode}'"
            )
        self.agent_miss_mode = agent_miss_mode
        self.agent_miss_offset = float(agent_miss_offset)
        self.agent_tracking_noise = float(agent_tracking_noise)
        self.rally_callback = rally_callback
        
        # object properties
        self.init_vel = np.array(ball_vel)
        self.ball_radius = ball_radius

        # speed parameters (magnitude of velocity vector)
        default_speed = np.linalg.norm(ball_vel)
        self.send_speed = send_speed if send_speed is not None else default_speed
        self.receive_speed = (
            receive_speed if receive_speed is not None else default_speed
        )

        # state variables
        self.player_points = init_points
        self.agent_points = 0
        self.in_play = True
        self.is_left_miss = False
        self.is_right_miss = False
        self.show_miss_message = False
        self._agent_will_miss = False
        self._agent_miss_dir = 1.0
        self.agent_hits = 0
        self.agent_misses = 0
        self.rally_outcomes = []
        self.last_rally_won = None
        self.rally = 1  # 0 = delay

        # event tracking - pre-allocate array w/ buffer
        estimated_frames = int((time_limit + pre_delay) * 120 * 2.0)
        self.events = np.zeros((estimated_frames, 13), dtype= np.float32) #np.float32)
        self.event_idx = 0
        self.events_buffer_warned = False
        self.save_file = save_file
        self.start_time = None
        self.last_hit = None

        self.intended_y = target_position(self.decoder_state)
        self.intended_shift = 0.0

        # display setup
        self.win_width, self.win_height = win.winHandle.get_size()
        self._setup_game_objects(paddle_size, ball_radius=self.ball_radius)

        # key setup
        self.step_size = self.win_height * step_size / 100
        if self.input_method == 'press':
            self.step_size = self.step_size / 10
        self.click = 0
        self.wheel = 0

       # performance
        self.total_hits = None
        self.total_rallies = None
        self.performance = None

        # log
        self.logger.info("trial params:")
        self.logger.info(f"  input method: {input_method}")
        self.logger.info(f"  paddle response mode: {paddle_response_mode}")
        self.logger.info(f"  reset timer: {reset_timer}")
        self.logger.info(f"  reset delay {reset_delay}")
        self.logger.info(f"  points increment: {point_increment}")
        self.logger.info(f"  time limit: {time_limit}")
        self.logger.info(f"  pre delay: {pre_delay}")
        self.logger.info("  binary controller: z > 0 TOP, otherwise BOTTOM")
        if self.input_method == "keyboard":
            self.logger.info(f"  behavioral input: 1=TOP, 2=BOTTOM; fixed delay={keyboard_delay_sec}s")
        else:
            self.logger.info("  no additional input delay")
        self.logger.info(f"  init velo: {ball_vel}")
        self.logger.info(f"  default speed: {default_speed}")
        self.logger.info(f"  send speed: {self.send_speed}")
        self.logger.info(f"  receive speed: {self.receive_speed}")
        self.logger.info(f"  agent miss prob: {self.agent_miss_prob}")
        self.logger.info(f"  agent miss mode: {self.agent_miss_mode}")
        self.logger.info(f"  agent miss offset: {self.agent_miss_offset}")
        self.logger.info(f"  agent tracking noise: {self.agent_tracking_noise}")

        self._reset_game()

    @property
    def points(self):
        return self.player_points

    @points.setter
    def points(self, value):
        self.player_points = value

    def _setup_game_objects(
        self,
        paddle_height: float = 0.2,
        paddle_width: float = 0.03,
        paddle_pos: float = 0.875,
        ball_radius: float = 0.02,
        fix_size: float = 0.05,
    ):
        self.paddle_height = paddle_height
        self.paddle_width = paddle_width
        self.paddle_pos = paddle_pos
        self.ball_radius = ball_radius

        self.left_paddle = visual.Rect(
            win=self.win,
            units="norm",
            width=self.paddle_width,
            height=self.paddle_height,
            fillColor="white",
            lineColor="white",
            pos=[-paddle_pos, 0],
        )

        self.right_paddle = visual.Rect(
            win=self.win,
            units="norm",
            width=self.paddle_width,
            height=self.paddle_height,
            fillColor="white",
            lineColor="white",
            pos=[paddle_pos, 0],
        )

        # aspect ratio (ensure ball circular)
        aspect_ratio = self.win_height / self.win_width

        self.ball = visual.Circle(
            win=self.win,
            units="norm",
            radius=self.ball_radius,
            fillColor="red",
            lineColor="red",
            pos=[0, 0],
            size=(aspect_ratio, 1.0),
        )

        self.score_text = visual.TextStim(
            win=self.win,
            units="norm",
            text="0 : 0",
            height=0.08,
            pos=[0, 0.85],
            color="white",
        )
        self.message_text = visual.TextStim(
            win=self.win,
            units="norm",
            text="",
            height=0.06,
            pos=[0, 0.72],
            color="white",
        )

        self.level_text = visual.TextStim(
            win=self.win,
            units="norm",
            text=f"Level: {self.level}",
            height=0.05,
            pos=[0, 0.8],
        )

        self.fixation = visual.ShapeStim(
            win=self.win,
            units="norm",
            vertices="cross",
            fillColor="white",
            lineColor="white",
            pos=[0, 0],
            size=(aspect_ratio * fix_size, fix_size),
        )

    def delay(self, change_color_time: float = 1.0):
        """Display fixation cross during pre-trial delay period.

        Args:
            change_color_time (float): Time before delay ends to change fixation color to red.

        Returns:
            bool: True if user escaped, False otherwise.
        """
        delay_start = core.getTime()
        self.logger.info("delay start")
        while core.getTime() - delay_start < self.pre_delay:
            keys = event.getKeys(keyList=["escape"])
            if "escape" in keys:
                self.logger.info("quitting...")
                self.export()
                self.compute_performance()
                return True

            elapsed = core.getTime() - delay_start
            time_remaining = self.pre_delay - elapsed

            if self.idle_callback is not None:
                self.idle_callback(self, elapsed, time_remaining)
            elif self.input_method == "keyboard":
                self._update_player_paddle()

            if time_remaining <= change_color_time:
                self.fixation.fillColor = "red"
                self.fixation.lineColor = "red"
            else:
                self.fixation.fillColor = "white"
                self.fixation.lineColor = "white"

            timestamp = core.getTime() - self.start_time
            if self.event_idx < len(self.events):
                self.events[self.event_idx] = [timestamp] + [0] * 12 # 11
                self.event_idx += 1

            self.fixation.draw()
            self.level_text.draw()
            self.win.flip()

        # reset after
        self.fixation.fillColor = "white"
        self.fixation.lineColor = "white"
        return False

    def _reset_game(self):
        # Agent serves each rally from the right side toward the player.
        self.ball.pos = [self.paddle_pos - self.paddle_width, 0]
        self._start_participant_approach()
        self._agent_will_miss = False
        self._agent_miss_dir = self.rng.choice([-1.0, 1.0])

        self.logger.info("ball reset")
        self.left_paddle.pos = [-self.paddle_pos, target_position(self.decoder_state)]
        self.right_paddle.pos = [self.paddle_pos, 0]
        self.in_play = True
        self.is_left_miss = False
        self.is_right_miss = False
        self.reset_timer = 0.0
        self.show_miss_message = False

        self.intended_y = target_position(self.decoder_state)
        self.intended_shift = 0.0
        self.wheel = 0

    def _start_participant_approach(self):
        self.target_side = self.target_schedule.next()
        self.approach_id += 1
        self._participant_contact_done = False
        endpoint = np.array([-self.paddle_pos, target_position(self.target_side)])
        direction = endpoint - np.asarray(self.ball.pos, dtype=float)
        self.ball_vel = direction * (self.receive_speed / np.linalg.norm(direction))

    def time_to_arrival(self) -> float | None:
        if not self.in_play or self.ball_vel[0] >= 0 or self._participant_contact_done:
            return None
        frames = (-self.paddle_pos - self.ball.pos[0]) / self.ball_vel[0]
        return max(0.0, float(frames) / self.refresh_rate)

    def _record_participant_contact(self) -> bool:
        hit = self.decoder_state == self.target_side
        self.participant_hits += int(hit)
        self.participant_misses += int(not hit)
        record = {
            "timestamp": core.getTime() - (self.start_time or 0.0),
            "run_id": self.run_id,
            "trial_id": self.trial_id,
            "rally": self.rally,
            "approach_id": self.approach_id,
            "target_side": self.target_side,
            "decoder_state_at_contact": self.decoder_state,
            "z_score_at_contact": self.z_score,
            "participant_hit": hit,
            "target_aligned_z": self.z_score if self.target_side == "TOP" else -self.z_score,
            "level": self.level,
            "clock_timestamp": core.getTime(),
            "trial_start_clock": self.start_time,
            "input_method": self.input_method,
            "keyboard_delay_sec": self.keyboard_controller.delay_sec if self.input_method == "keyboard" else 0.0,
        }
        self.contacts.append(record)
        if self.contact_callback is not None:
            self.contact_callback(self, record)
        return hit

    def _choose_agent_miss(self) -> bool:
        if self.agent_miss_mode == "random":
            return self.rng.random() < self.agent_miss_prob

        attempts_after_next = self.agent_hits + self.agent_misses + 1
        target_misses = math.floor(self.agent_miss_prob * attempts_after_next + 0.5)
        return self.agent_misses < target_misses


    def _update_player_paddle(self, dt: float = 0.0):
        """Brain input is a normalized z-score; manual inputs simulate its sign."""
        z = self.z_score
        self.click = 0
        if self.input_method == "keyboard":
            keys = event.getKeys(keyList=["1", "2", "num_1", "num_2"])
            now = core.getTime() - (self.start_time or 0.0)
            side = self.keyboard_controller.update(now, keys)
            self.click = (1 if keys[-1] in ("1", "num_1") else -1) if keys else 0
            z = 1.0 if side == "TOP" else -1.0
        elif self.input_method == "brain":
            z = float(self.brain_controller(dt)) if self.brain_controller else self.brain_position
        elif self.input_method == "mouse":
            z = self.mouse.getPos()[1]
        elif self.input_method == "press":
            left, _, right = self.mouse.getPressed()
            self.click = 1 if left else (-1 if right else 0)
            if self.click:
                z = float(self.click)
        elif self.input_method == "wheel":
            wheel = self.mouse.getWheelRel()[1]
            self.click = -1 if wheel > 0 else (1 if wheel < 0 else 0)
            if self.click:
                z = float(self.click)
        self.set_brain_position(z)

    def set_brain_position(self, position: float) -> None:
        """Compatibility setter: the supplied value is now z, never a y coordinate."""
        self.decoder_state = decoder_state(position)
        self.z_score = float(position)
        self.brain_position = self.z_score
        self.intended_y = target_position(self.decoder_state)
        self.left_paddle.pos = [-self.paddle_pos, self.intended_y]

    def _update_opponent_paddle(self):
        target_y = self.ball.pos[1]
        if self.agent_tracking_noise > 0:
            target_y += self.rng.gauss(0, self.agent_tracking_noise)
        if (
            self._agent_will_miss
            and self.ball.pos[0] > 0.5
            and self.ball_vel[0] > 0
        ):
            miss_offset = max(
                self.agent_miss_offset,
                self.paddle_height / 2 + self.ball_radius + 0.02,
            )
            miss_dir = self._agent_miss_dir
            candidate_y = float(np.clip(target_y + miss_dir * miss_offset, -1.0, 1.0))
            miss_distance = self.paddle_height / 2 + self.ball_radius
            if abs(candidate_y - self.ball.pos[1]) <= miss_distance:
                miss_dir *= -1.0
                candidate_y = float(
                    np.clip(target_y + miss_dir * miss_offset, -1.0, 1.0)
                )
            target_y = candidate_y
        self.right_paddle.pos = [
            self.paddle_pos,
            float(np.clip(target_y, -1.0, 1.0)),
        ]

    def _check_hits(self):
        """Resolve one decision at the fixed paddle x plane, independent of size."""
        wall, left, right = 0, 0, 0
        if abs(self.ball.pos[1]) > 0.95:
            self.ball_vel[1] *= -1
            wall = 1

        if self.ball_vel[0] < 0:
            if not self._participant_contact_done and self.ball.pos[0] <= -self.paddle_pos:
                self._participant_contact_done = True
                # Snap the small frame overshoot to the categorical visual endpoint.
                self.ball.pos = [-self.paddle_pos, target_position(self.target_side)]
                if self._record_participant_contact():
                    self.ball_vel[0] *= -1
                    self.ball_vel *= self.send_speed / np.linalg.norm(self.ball_vel)
                    left = 1
                    self._agent_will_miss = self._choose_agent_miss()
                    self._agent_miss_dir = self.rng.choice([-1.0, 1.0])
                    self.logger.info("left binary hit")
        elif self.ball.pos[0] >= self.paddle_pos and not self._agent_will_miss:
            # The planned opponent miss probability is authoritative as well.
            self.ball.pos = [self.paddle_pos, float(self.ball.pos[1])]
            self.agent_hits += 1
            right = 1
            self._start_participant_approach()
            self.logger.info("right hit")
        return wall, left, right

    def _update_miss(self, player_won: bool):
        self.in_play = False
        self.reset_timer = 0.0
        self.rally += 1
        self.last_rally_won = player_won
        self.rally_outcomes.append(player_won)
        if self.rally_callback is not None:
            self.rally_callback(self, player_won)

    def _check_miss(self):
        if self.ball.pos[0] < -1.0:
            self.logger.info("miss left (player)")
            self.show_miss_message = True
            self.is_left_miss = True
            self.agent_points += 1
            self._update_miss(player_won=False)

        elif self.ball.pos[0] > 1.0:
            self.logger.info("miss right (agent)")
            self.is_right_miss = True
            self.agent_misses += 1
            self.player_points += 1
            self._update_miss(player_won=True)

    def update(self, dt):
        """Update game state for one frame.

        Args:
            dt (float): Time delta since last frame.
        """
        timestamp = core.getTime() - self.start_time

        # Continue consuming TRs during the short post-miss animation too.
        self._update_player_paddle(dt)
        if not self.in_play:
            self.reset_timer += dt
            if self.reset_timer >= self.reset_delay:
                self._reset_game()
            return

        self._update_opponent_paddle()

        # update ball position
        ball_pos = self.ball.pos
        ball_pos[0] += self.ball_vel[0]
        ball_pos[1] += self.ball_vel[1]
        self.ball.pos = ball_pos

        hits = self._check_hits()
        self._check_miss()
        ball_pos = self.ball.pos

        if self.event_idx < len(self.events):
            self.events[self.event_idx, 0] = timestamp
            self.events[self.event_idx, 1] = self.rally
            self.events[self.event_idx, 2] = self.points
            self.events[self.event_idx, 3] = ball_pos[0]
            self.events[self.event_idx, 4] = ball_pos[1]
            self.events[self.event_idx, 5] = self.click
            self.events[self.event_idx, 6] = self.left_paddle.pos[1]
            self.events[self.event_idx, 7] = self.right_paddle.pos[1]
            self.events[self.event_idx, 8] = hits[0]
            self.events[self.event_idx, 9] = hits[1]
            self.events[self.event_idx, 10] = hits[2]
            self.events[self.event_idx, 11] = self.is_left_miss
            self.events[self.event_idx, 12] = self.is_right_miss
            self.event_idx += 1

        if self.frame_callback is not None:
            self.frame_callback(self, dt, timestamp)

    def draw(self):
        """Draw all game objects to the window."""
        self.left_paddle.draw()
        self.right_paddle.draw()
        self.ball.draw()

        self.score_text.text = f"{self.player_points} : {self.agent_points}"
        self.score_text.draw()
        if self.show_miss_message:
            self.message_text.text = "You scored!" if self.is_right_miss else "Missed!"
            self.message_text.draw()

    def export(self):
        """Save event data to file.

        Returns:
            pd.DataFrame: Event data, or None if no events recorded.
        """
        if self.save_file is not None and self.input_method == "keyboard":
            key_file = Path(self.save_file).with_name(Path(self.save_file).stem + "_keys.tsv")
            pd.DataFrame(self.keyboard_controller.records, columns=[
                "key", "key_timestamp", "requested_state", "scheduled_timestamp", "applied_timestamp",
            ]).to_csv(key_file, sep="\t", index=False)
        if self.event_idx == 0 or self.save_file is None:
            self.logger.info("no events to save")
            return None

        # Trim array to actual number of events recorded
        events_trimmed = self.events[: self.event_idx]

        df = pd.DataFrame(
            events_trimmed,
            columns=[
                "timestamp",
                "rally",
                "points",
                "ball_x",
                "ball_y",
                "click",
                "left_paddle_y",
                "right_paddle_y",
                "wall_hit",
                "left_paddle_hit",
                "right_paddle_hit",
                "left_miss",
                "right_miss",
            ],
        )

        df = df.astype(
            {
                "timestamp": "float32",
                "rally": "uint16",
                "points": "uint16",
                "ball_x": "float32",
                "ball_y": "float32",
                "click": "int8",
                "left_paddle_y": "float32",
                "right_paddle_y": "float32",
                "wall_hit": "uint8",
                "left_paddle_hit": "uint8",
                "right_paddle_hit": "uint8",
                "left_miss": "uint8",
                "right_miss": "uint8",
            }
        )
        df.to_csv(self.save_file, sep="\t", index=False)
        contact_file = Path(self.save_file).with_name(Path(self.save_file).stem + "_contacts.tsv")
        columns = ["timestamp", "run_id", "trial_id", "rally", "approach_id", "target_side",
                   "decoder_state_at_contact", "z_score_at_contact", "participant_hit",
                   "target_aligned_z", "level", "clock_timestamp", "trial_start_clock",
                   "input_method", "keyboard_delay_sec"]
        pd.DataFrame(self.contacts, columns=columns).to_csv(contact_file, sep="\t", index=False)
        self.logger.info(f"saved {len(df)} events to {self.save_file}")
        return df

    def compute_performance(self):
        """Calculate performance metrics from event data."""
        self.total_hits = self.participant_hits
        misses = self.participant_misses
        self.total_possible = self.total_hits + misses
        self.performance = (
            self.total_hits / self.total_possible
            if self.total_possible > 0
            else 0.0
        )
        self.logger.info(
            f"performance: {self.total_hits}/{self.total_possible} = {self.performance}"
        )

    def run(
        self,
        instruction_text: str | None = None,
        show_instructions: bool = True,
        refresh_rate: float | None = None,
    ):
        """Run the game loop.

        Args:
            instruction_text (str, optional): Text to display on instruction screen.
            show_instructions (bool): Whether to show instruction screen.
            refresh_rate (float, optional): Display refresh rate in Hz.

        Returns:
            bool: True if completed normally, False if escaped.
        """
        if refresh_rate is None:
            refresh = self.win.getActualFrameRate() or 60.0
        else:
            refresh = refresh_rate

        self.refresh_rate = refresh
        if show_instructions and instruction_text is not None:
            instructions = visual.TextStim(
                self.win,
                text=instruction_text,
                height=20,
            )
            instructions.draw()
            self.win.flip()
            print(
                "[BrainPong] Instruction screen is ready. "
                "Select the PsychoPy window and press SPACE.",
                flush=True,
            )
            event.waitKeys(keyList=["space"])
            print("[BrainPong] SPACE received; starting trial.", flush=True)

        self.start_time = core.getTime()
        self.logger.info("trial start")

        if self.pre_delay > 0:
            escaped = self.delay()
            if escaped:
                return False

        self.logger.info("game start")
        game_start_time = core.getTime()
        running = True
        while running:
            dt = 1 / refresh

            elapsed_time = core.getTime() - game_start_time
            if elapsed_time >= self.time_limit:
                running = False
                self.logger.info("time limit reached, ending...")
                break

            keys = event.getKeys(keyList=["escape"])
            if "escape" in keys:
                running = False
                self.export()
                self.compute_performance()
                return False

            self.update(dt)
            self.draw()
            self.win.flip()

        self.export()
        self.compute_performance()

        return True
