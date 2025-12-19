#!/usr/bin/env python
import argparse
from collections import deque
from multiprocessing import Process, Queue
from queue import Empty
from pathlib import Path

from rt_pipeline import RTSessionConfig, run_rt_pipeline


def run_psychopy_presentation(score_queue: Queue, max_points: int) -> None:
    from psychopy import core, event, visual

    win = visual.Window(size=(1000, 700), color="black", units="pix")
    title = visual.TextStim(win, text="Real-time Scores", pos=(0, 300), color="white")

    margins = {"left": 80, "right": 40, "bottom": 80, "top": 80}
    plot_width = win.size[0] - margins["left"] - margins["right"]
    plot_height = win.size[1] - margins["bottom"] - margins["top"]
    origin_x = -win.size[0] / 2 + margins["left"]
    origin_y = -win.size[1] / 2 + margins["bottom"]

    x_axis = visual.Line(
        win,
        start=(origin_x, origin_y),
        end=(origin_x + plot_width, origin_y),
        lineColor="white",
    )
    y_axis = visual.Line(
        win,
        start=(origin_x, origin_y),
        end=(origin_x, origin_y + plot_height),
        lineColor="white",
    )
    score_line = visual.ShapeStim(win, vertices=[], closeShape=False, lineColor="cyan")
    last_score_text = visual.TextStim(win, text="", pos=(0, -300), color="white")

    scores = deque(maxlen=max_points)
    needs_redraw = True

    def update_plot() -> None:
        nonlocal needs_redraw
        if not scores:
            score_line.vertices = []
            last_score_text.text = "Waiting for scores…"
            needs_redraw = True
            return

        min_score = min(scores)
        max_score = max(scores)
        if min_score == max_score:
            min_score -= 0.5
            max_score += 0.5

        span = max_score - min_score
        x_step = plot_width / max(1, max_points - 1)

        vertices = []
        for idx, score in enumerate(scores):
            x = origin_x + idx * x_step
            y_norm = (score - min_score) / span
            y = origin_y + y_norm * plot_height
            vertices.append((x, y))

        score_line.vertices = vertices
        last_score_text.text = f"Last score: {scores[-1]:.4f}"
        needs_redraw = True

    while True:
        updated = False
        try:
            while True:
                message = score_queue.get_nowait()
                scores.append(float(message["score_raw"]))
                updated = True
        except Empty:
            pass

        if updated or needs_redraw:
            update_plot()
            win.clearBuffer()
            title.draw()
            x_axis.draw()
            y_axis.draw()
            score_line.draw()
            last_score_text.draw()
            win.flip()
            needs_redraw = False

        if "escape" in event.getKeys():
            break

        core.wait(0.02)

    win.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run rt_pipeline in parallel with a PsychoPy visualization of the last 20 scores."
        )
    )
    parser.add_argument("--sub", required=True, help="Subject ID, e.g. 00086")
    parser.add_argument("--day", required=True, help="Day/session, e.g. 3")
    parser.add_argument("--run", required=True, help="Run number, e.g. 4")
    parser.add_argument(
        "--incoming-root",
        required=False,
        default="/home/sin/DecNef_pain_Dec23/realtime/incoming/pain7T/20251105.20251105_00085.Kostya",
        help="Folder where scanner writes DICOMs in real-time.",
    )
    parser.add_argument(
        "--base-data",
        required=False,
        default="/SSD2/DecNef_py/data",
        help="Base preproc data folder (same as offline pipeline).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=20,
        help="Number of most recent scores to plot.",
    )
    parser.add_argument(
        "--decoder-template",
        required=False,
        help="Optional decoder template path to override the default.",
    )
    args = parser.parse_args()

    cfg = RTSessionConfig(
        subject=args.sub,
        day=args.day,
        run=args.run,
        incoming_root=Path(args.incoming_root),
        base_data=Path(args.base_data),
        decoder_template=Path(args.decoder_template) if args.decoder_template else None,
    )

    score_queue: Queue = Queue(maxsize=100)
    pipeline_process = Process(target=run_rt_pipeline, args=(cfg, score_queue))
    pipeline_process.start()

    try:
        run_psychopy_presentation(score_queue, args.max_points)
    finally:
        if pipeline_process.is_alive():
            pipeline_process.terminate()
        pipeline_process.join(timeout=5)


if __name__ == "__main__":
    main()
