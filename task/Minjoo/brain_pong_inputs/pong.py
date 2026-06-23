from __future__ import annotations
from pathlib import Path
import logging
import random
from collections import deque
from typing import Callable

from psychopy import visual, core, event

import numpy as np
import pandas as pd
from scipy.stats import gamma


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
        paddle_lag (int): Input lag in frames.
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
        step_size: float = 1.0, # in percent?
        intended_y: float = 0.0,
        intended_shift: float = 0.0,
        input_method: str = "wheel",
        paddle_response_mode: str = "hrf",
        brain_controller: Callable[[float], float] | None = None,
        frame_callback: Callable | None = None,
        idle_callback: Callable | None = None,
    ):
        # params
        self.logger = logger
        self.win = win

        # Validate input method
        if input_method not in ["mouse", "brain", "press", "wheel"]:
            raise ValueError(
                f"input_method must be 'mouse' or 'brain' or 'press' or 'wheel', got '{input_method}'"
            )

        if paddle_response_mode not in ["hrf", "hold"]:
            raise ValueError(
                "paddle_response_mode must be 'hrf' or 'hold', "
                f"got '{paddle_response_mode}'"
            )

        self.input_method = input_method
        self.paddle_response_mode = paddle_response_mode
        self.brain_controller = brain_controller
        self.frame_callback = frame_callback
        self.idle_callback = idle_callback
        self.brain_position = 0.0
        if self.input_method != "brain":
            self.mouse = event.Mouse(win=win)

        self.reset_timer = reset_timer
        self.reset_delay = reset_delay
        self.point_increment = point_increment
        self.time_limit = time_limit
        self.pre_delay = pre_delay
        self.paddle_lag = paddle_lag
        self.level = level
        self.opponent_handicap = 1 - opponent_handicap
        
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
        self.points = init_points
        self.in_play = True
        self.is_left_miss = False
        self.is_right_miss = False
        self.show_miss_message = False
        self.rally = 1  # 0 = delay

        # event tracking - pre-allocate array w/ buffer
        estimated_frames = int((time_limit + pre_delay) * 120 * 2.0)
        self.events = np.zeros((estimated_frames, 13), dtype= np.float32) #np.float32)
        self.event_idx = 0
        self.events_buffer_warned = False
        self.save_file = save_file
        self.start_time = None
        self.last_hit = None

        # paddle lag buffer - stores input position history
        self.input_history =[] # np.zeros(paddle_lag+1) #deque(maxlen=max(paddle_lag + 1, 1))
        self.intended_y = 0.0
        self.intended_shift = 0.0

        # Do HRF convolution
        self.window_input = []
        self.input_move = []
        self.convolved = []
        self.hrf_input = []


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
        self.logger.info(f"  paddle lag: {paddle_lag} frames")
        self.logger.info(f"  init velo: {ball_vel}")
        self.logger.info(f"  default speed: {default_speed}")
        self.logger.info(f"  send speed: {self.send_speed}")
        self.logger.info(f"  receive speed: {self.receive_speed}")

        self._reset_game()

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

        self.points_text = visual.TextStim(
            win=self.win, units="norm", text="Points: 0", height=0.05, pos=[0, 0.8]
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
        self.ball.pos = [0, 0]

        # always go toward opponent at random angle with send speed
        ymin, ymax = (0.3, 0.8)
        angle_y = random.uniform(*random.choice([(-ymax, -ymin), (ymin, ymax)]))
        direction = np.array([1.0, angle_y])
        self.ball_vel = direction * (self.send_speed / np.linalg.norm(direction))

        self.logger.info("ball reset")
        self.left_paddle.pos = [-self.paddle_pos, 0]
        self.right_paddle.pos = [self.paddle_pos, 0]
        self.in_play = True
        self.is_left_miss = False
        self.is_right_miss = False
        self.reset_timer = 0.0
        self.show_miss_message = False

        # reset intended position and input history
        self.intended_y = 0.0
        self.intended_shift = 0.0
        self.wheel = 0
        #self.input_history.clear()
        self.input_history = list(np.zeros(self.paddle_lag+1))
        self.clickevent = []
        self.hrf_kernel = self._make_hrf_kernel()
        self.peak_hold_kernel = self._make_peak_hold_kernel()

        ## pre-fill buffer with initial position
        #for _ in range(self.paddle_lag + 1):
        #    self.input_history.append(0.0)


    def _hrf_(self, times):
        """ Return values for HRF at given times """
        # Gamma pdf for the peak
        peak_values = gamma.pdf(times, 6)
        # Gamma pdf for the undershoot
        undershoot_values = gamma.pdf(times, 12)
        # Combine them
        values = peak_values - 0.35 * undershoot_values
        # Scale max to 0.6
        return values / np.max(values)  

    def _make_hrf_kernel(self):
        """paddle_lag+1 길이의 HRF 커널 (프레임 기반)"""
        n = self.paddle_lag + 1
        hrf_dur = 20.0                         # HRF 20초 동안 자르기 (원하면 조절)
        dt = hrf_dur / (n - 1)                 # HRF 샘플 간격
        t = np.arange(0, n) * dt               # 0, dt, 2dt, ...
        return self._hrf_(t)                   # shape: (n,)

    def _make_peak_hold_kernel(self):
        """HRF rising limb that holds at peak instead of returning to baseline."""
        hrf_kernel = self._make_hrf_kernel()
        peak_idx = int(np.argmax(hrf_kernel))
        peak_val = hrf_kernel[peak_idx]
        hold_kernel = hrf_kernel.copy()
        hold_kernel[peak_idx:] = peak_val
        return hold_kernel / peak_val

    def _get_click_response(self):
        n = min(len(self.clickevent), len(self.hrf_kernel))
        ev_use = self.clickevent[-n:]

        if self.paddle_response_mode == "hold":
            k = self.peak_hold_kernel[:n]
            older_click_sum = sum(self.clickevent[:-n])
            return older_click_sum + float(np.dot(ev_use[::-1], k))

        k = self.hrf_kernel[:n]
        return float(np.dot(ev_use[::-1], k))


    def _update_player_paddle(self, dt: float = 0.0):

        if self.input_method == "brain":
            if self.brain_controller is not None:
                self.brain_position = float(self.brain_controller(dt))
            self.intended_y = float(np.clip(self.brain_position, -1.0, 1.0))
            self.left_paddle.pos = [-self.paddle_pos, self.intended_y]
            self.click = 0
            return

        if self.input_method == 'mouse':
            mouse_pos = self.mouse.getPos()
            self.intended_y = mouse_pos[1] / (self.win_height / 2)
            self.input_history.append(self.intended_y)

            lagged_y = self.input_history[-self.paddle_lag]
            self.left_paddle.pos = [-self.paddle_pos, lagged_y] 

        else:
            if self.input_method == 'press':

                left, middle, right = self.mouse.getPressed()

                if left:
                    self.click = 1
                elif right:
                    self.click = -1
                else:
                    self.click = 0

            elif self.input_method == 'wheel':

                wheel = self.mouse.getWheelRel()

                if wheel[1] > 0:
                    self.click = -1
                    #wheel = [0, 0]
                elif wheel[1] < 0:
                    self.click = 1
                    #wheel = [0, 0]
                else:
                    self.click = 0


            self.clickevent.append(self.click)

            self.intended_y = (
                self._get_click_response() * self.step_size
            ) / (self.win_height / 2)
            self.input_history.append(np.clip(self.intended_y, -1, 1)) 

            if len(self.input_history) > self.paddle_lag:
                delayed_val = self.input_history[-(self.paddle_lag+1)]
            else:
                delayed_val = 0.0

            self.left_paddle.pos = [-self.paddle_pos, delayed_val]


        # apply most/oldest lagged position to actual paddle
        
        # When you need direct n-s delay, uncomment these lines.

    def set_brain_position(self, position: float) -> None:
        """Set paddle position supplied by an external neural controller."""
        self.brain_position = float(np.clip(position, -1.0, 1.0))

    def _update_opponent_paddle(self):
        self.right_paddle.pos = [self.paddle_pos, self.ball.pos[1]]

    def _check_paddle_hit(self, paddle):
        """Check if ball hits a paddle"""
        ball_x, ball_y = self.ball.pos
        paddle_x, paddle_y = paddle.pos

        within_x = abs(ball_x - paddle_x) < (self.ball_radius + self.paddle_width / 2)
        within_y = abs(ball_y - paddle_y) < (self.paddle_height / 2 + self.ball_radius)

        return within_x and within_y

    def _check_hits(self):
        """Detect hit with paddle Reverse velocity i"""
        wall, left, right = 0, 0, 0
        ball_vel = self.ball_vel
        ball_pos = self.ball.pos

        if abs(ball_pos[1]) > 0.95:
            ball_vel[1] *= -1
            wall = 1

        if ball_vel[0] < 0:
            # ball moving left - check left paddle
            if (
                self._check_paddle_hit(self.left_paddle)
                and ball_pos[0] > self.left_paddle.pos[0]
            ):
                ball_vel[0] = -ball_vel[0]
                speed = np.sqrt(ball_vel[0] ** 2 + ball_vel[1] ** 2)
                scale = self.send_speed / speed
                ball_vel[0] *= scale
                ball_vel[1] *= scale
                self.points += self.point_increment
                left = 1
                self.logger.info("left hit")
        else:
            # ball moving right - check right paddle
            if (
                self._check_paddle_hit(self.right_paddle)
                and ball_pos[0] < self.right_paddle.pos[0]
            ):
                ball_vel[0] = -ball_vel[0]
                speed = np.sqrt(ball_vel[0] ** 2 + ball_vel[1] ** 2)
                scale = self.receive_speed / speed
                ball_vel[0] *= scale
                ball_vel[1] *= scale
                right = 1
                self.logger.info("right hit")

        return wall, left, right

    def _update_miss(self):
        self.in_play = False
        self.reset_timer = 0.0
        self.rally += 1

    def _check_miss(self):
        if self.ball.pos[0] < -1.0:
            self.logger.info("miss left")
            self.show_miss_message = True
            self.is_left_miss = True
            self._update_miss()

        elif self.ball.pos[0] > 1.0:
            self.logger.info("miss right")
            self.is_right_miss = True
            self._update_miss()

    def update(self, dt):
        """Update game state for one frame.

        Args:
            dt (float): Time delta since last frame.
        """
        timestamp = core.getTime() - self.start_time

        if not self.in_play:
            self.reset_timer += dt
            if self.reset_timer >= self.reset_delay:
                self._reset_game()
            return

        self._update_player_paddle(dt)
        self._update_opponent_paddle()

        # update ball position
        ball_pos = self.ball.pos
        ball_pos[0] += self.ball_vel[0]
        ball_pos[1] += self.ball_vel[1]
        self.ball.pos = ball_pos

        hits = self._check_hits()
        self._check_miss()

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

        if self.show_miss_message:
            self.points_text.text = "Miss!"
        else:
            self.points_text.text = f"Points: {self.points}"
        self.points_text.draw()

    def export(self):
        """Save event data to file.

        Returns:
            pd.DataFrame: Event data, or None if no events recorded.
        """
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
        self.logger.info(f"saved {len(df)} events to {self.save_file}")
        return df

    def compute_performance(self):
        """Calculate performance metrics from event data."""
        events = self.events[: self.event_idx]
        self.total_hits = int(np.sum(events[:, -4]))
        misses = int(np.sum(events[:, -2]))
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
            refresh = self.get_fps()
        else:
            refresh = refresh_rate

        if show_instructions and instruction_text is not None:
            instructions = visual.TextStim(
                self.win,
                text=instruction_text,
                height=20,
            )
            instructions.draw()
            self.win.flip()
            event.waitKeys(keyList=["space"])

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
