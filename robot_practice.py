"""
Continuous neurofeedback DELAY-training task - GAMIFIED (two levels).

TOP    : big white ECG-style line (0-100). Scroll the wheel UP to raise it, but
         the effect appears with an EXACT 8 s delay. By default it jitters around
         the middle, like the decoder output.
BOTTOM : your robot vs an opponent, fighting on their own.

TWO LEVELS:
  OUR LEVEL   - keeping the line HIGH fills a POWER meter (fills faster the higher
                the line). Full meter -> our robot LEVELS UP: bigger and stronger.
  ENEMY LEVEL - every time you WIN 3 FIGHTS IN A ROW, the enemy LEVELS UP too.

Both start at level 1 (an even, ~50/50 fight).

Window is built exactly like affectivestroop.py (size=[1920,1080], units='height',
monitor='testMonitor', fullscr set after construction) so text/geometry stay
centered on the scanner display instead of bunching to the left.

Saves CSVs per run in ./data
PsychoPy 2023+ / Python 3. Press ESCAPE to quit.
"""

from psychopy import visual, core, event, data
from collections import deque
import numpy as np
import random, csv, os, math

# ---------------------------------------------------------------------------
# CONFIG  (tune these)
# ---------------------------------------------------------------------------
DELAY            = 8.0    # feedback delay, seconds (fixed, matches real task)
SESSION_DURATION = 180    # 3 minutes

# --- signal ---
SCROLL_GAIN      = 6.0    # points added per wheel notch
EFFORT_HALFLIFE  = 6.0    # s for the scrolled-up boost to halve (must keep working)
NOISE_SD         = 7.0    # size of the ongoing jitter
NOISE_TAU        = 2.0    # s, how slow the jitter is (bigger = smoother)
DIP_MIN, DIP_MAX = 1.5, 4.0     # random seconds between downward dips
P_RETURN         = 0.8    # chance a dip pulls back to centre (else a small drop)
RETURN_FRAC      = (0.5, 1.0)   # how far to centre a "return" dip pulls
DROP_AMOUNT      = (6.0, 12.0)  # size of a "small drop" dip

# --- OUR level (from the meter) ---
OUR_LEVEL_COST   = 3500.0   # meter units for the first level-up
COST_GROWTH      = 0.20     # each level costs this much more (0 = constant)

# --- ENEMY level (from win streak) ---
WIN_STREAK       = 3        # consecutive wins that level the enemy up

# --- fight strength (both sides) ---
STR_BASE         = 50.0     # strength at level 1
STR_STEP         = 12.0     # strength gained per level
GAME_HP          = 100.0
DMG_K            = 0.22     # damage per hit = DMG_K * strength
ATTACK_INTERVAL  = 0.5      # seconds between hits
DMG_JITTER       = 0.30     # +/- randomness per hit (keeps equal levels ~chance)

MAX_LEVEL        = 12       # safety cap for both sides

WINDOW_SEC       = 14.0     # seconds of signal across the screen
N_TRACE          = 240
INVERT_WHEEL     = False

SAMPLE_DT = WINDOW_SEC / N_TRACE

def our_cost(lvl):
    return OUR_LEVEL_COST * (1 + COST_GROWTH * (lvl - 1))

def strength(lvl):
    return STR_BASE + STR_STEP * (lvl - 1)

def scale_for(lvl, base):
    return base * min(0.72 + 0.12 * (lvl - 1), 2.2)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# PARTICIPANT INFO  +  display options
# ---------------------------------------------------------------------------
import argparse
_ap = argparse.ArgumentParser(
    description="Robot practice task (mouse-driven). Opens on YOUR monitor by default.")
_ap.add_argument("--screen", type=int, default=0,
                 help="Monitor index to open on (default 0 = your main monitor). "
                      "Practice should stay on your monitor even if the scanner is connected; "
                      "if it opens on the wrong display, try --screen 1.")
_ap.add_argument("--windowed", action="store_true",
                 help="Open in a window instead of fullscreen (useful to keep it on your laptop "
                      "while the scanner display is attached).")
_ap.add_argument("--win-size", type=int, nargs=2, default=[1280, 720], metavar=("W", "H"),
                 help="Window size in pixels when using --windowed (default 1280 720).")
_ap.add_argument("--pid", default=None, help="Participant ID (skips the prompt if given).")
_ap.add_argument("--ses", default=None, help="Session label (skips the prompt if given).")
_args, _ = _ap.parse_known_args()

pid = _args.pid if _args.pid is not None else (input("Participant ID: ").strip() or "test")
ses = _args.ses if _args.ses is not None else (input("Session [001]: ").strip() or "001")
os.makedirs('data', exist_ok=True)
base = os.path.join('data', "%s_%s_robot_%s" % (pid, ses, data.getDateStr()))
log_file = open(base + "_timeseries.csv", 'w', newline='')
lw = csv.writer(log_file)
lw.writerow(['t', 'control', 'displayed', 'meter', 'our_level', 'enemy_level',
             'streak', 'our_hp', 'opp_hp', 'wins', 'losses'])

# ---------------------------------------------------------------------------
# WINDOW  (affectivestroop.py structure) but forced onto YOUR monitor.
# Practice must open on the experimenter's screen (default screen 0), NOT the
# scanner display, even when the scanner is connected. Use --screen to override
# and --windowed to avoid fullscreen entirely.
# ---------------------------------------------------------------------------
_fullscreen = not _args.windowed
_win_size = [1920, 1080] if _fullscreen else list(_args.win_size)
win = visual.Window(size=_win_size, units='height', color=(-0.85, -0.85, -0.8),
                    colorSpace='rgb', pos=(0, 0), allowGUI=True,
                    checkTiming=False, screen=_args.screen, monitor='testMonitor')  # size in pix, units in height
win.fullscr = _fullscreen
if _fullscreen:
    try:
        win.getActualFrameRate()
    except Exception:
        pass
win.mouseVisible = False

w, h = win.size
RIGHT = (w / float(h)) / 2.0
LEFT = -RIGHT
mouse = event.Mouse(visible=False, win=win)

# --- BIG plot on top ---
PLOT_BOT, PLOT_TOP = -0.14, 0.46
GROUND = -0.46
def sy(score):
    return PLOT_BOT + (score / 100.0) * (PLOT_TOP - PLOT_BOT)

frame_top = visual.Line(win, start=(LEFT + 0.02, sy(100)), end=(RIGHT - 0.02, sy(100)), lineColor=(-0.4, -0.4, -0.3), colorSpace='rgb')
frame_bot = visual.Line(win, start=(LEFT + 0.02, sy(0)),   end=(RIGHT - 0.02, sy(0)),   lineColor=(-0.4, -0.4, -0.3), colorSpace='rgb')
mid_line  = visual.Line(win, start=(LEFT + 0.02, sy(50)),  end=(RIGHT - 0.02, sy(50)),  lineColor=(-0.6, -0.6, -0.5), colorSpace='rgb')
xs = np.linspace(LEFT + 0.02, RIGHT - 0.02, N_TRACE)
trace = visual.ShapeStim(win, vertices=np.column_stack([xs, np.full(N_TRACE, sy(50))]),
                         closeShape=False, fillColor=None, lineColor='white', lineWidth=3, colorSpace='rgb')

# --- POWER meter (fills OUR level) ---
BAR_X, BAR_W = LEFT + 0.07, 0.05
BAR_BOT, BAR_TOP = GROUND + 0.02, GROUND + 0.24
BAR_H = BAR_TOP - BAR_BOT
bar_out = visual.Rect(win, width=BAR_W, height=BAR_H, pos=(BAR_X, (BAR_BOT + BAR_TOP) / 2),
                      fillColor=(-0.75, -0.75, -0.65), lineColor=(0.1, 0.3, 0.45), lineWidth=2, colorSpace='rgb')
bar_fill = visual.Rect(win, width=BAR_W - 0.010, height=0.0, fillColor=(0.0, 0.75, 0.85), lineColor=None, colorSpace='rgb')
pow_lbl = visual.TextStim(win, text='POWER', pos=(BAR_X, BAR_TOP + 0.03), height=0.022, color=(0.3, 0.6, 0.7), colorSpace='rgb')
our_lvl_lbl = visual.TextStim(win, text='LV 1', pos=(BAR_X, BAR_BOT - 0.045), height=0.030, color=(0.3, 0.85, 1.0), bold=True, colorSpace='rgb')

# --- win / loss counter ---
wl_lbl = visual.TextStim(win, text='W 0   L 0', pos=(RIGHT - 0.14, 0.485), height=0.030,
                         color=(0.75, 0.75, 0.6), bold=True, colorSpace='rgb')

# ---------------------------------------------------------------------------
# ROBOTS  (drawn from polygons, no image files)
# ---------------------------------------------------------------------------
def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

PARTS = [
    ('leg_l',  rect(-0.22, 0.00, -0.06, 0.28), 'dark'),
    ('leg_r',  rect( 0.06, 0.00,  0.22, 0.28), 'dark'),
    ('arm_b',  rect(-0.46, 0.34, -0.30, 0.68), 'dark'),
    ('body',   rect(-0.30, 0.26,  0.30, 0.74), 'main'),
    ('core',   rect(-0.10, 0.42,  0.10, 0.60), 'glow'),
    ('arm_f',  rect( 0.30, 0.40,  0.54, 0.58), 'main'),
    ('fist',   rect( 0.50, 0.36,  0.68, 0.62), 'glow'),
    ('neck',   rect(-0.06, 0.74,  0.06, 0.80), 'dark'),
    ('head',   rect(-0.21, 0.79,  0.21, 1.06), 'main'),
    ('eye_l',  rect(-0.14, 0.88, -0.04, 0.97), 'glow'),
    ('eye_r',  rect( 0.04, 0.88,  0.14, 0.97), 'glow'),
    ('ant',    rect(-0.02, 1.06,  0.02, 1.18), 'dark'),
    ('ant_t',  rect(-0.05, 1.17,  0.05, 1.25), 'glow'),
]

def make_robot(main, dark, glow):
    cols = {'main': main, 'dark': dark, 'glow': glow}
    return [visual.ShapeStim(win, vertices=v, fillColor=cols[c], lineColor=None,
                             closeShape=True, autoLog=False, colorSpace='rgb') for _, v, c in PARTS]

def place(parts, x, y, scale, flip=1):
    for p in parts:
        p.pos = (x, y)
        p.size = (scale * flip, scale)

ours = make_robot((0.05, 0.55, 0.95), (-0.35, 0.05, 0.45), (0.35, 0.95, 1.0))
opp  = make_robot((0.85, 0.15, 0.10), (0.35, -0.35, -0.4), (1.0, 0.6, 0.0))

OUR_X, OPP_X = -0.14, 0.30
BASE_SCALE = 0.16

ground_line = visual.Line(win, start=(LEFT + 0.02, GROUND), end=(RIGHT - 0.02, GROUND), lineColor=(-0.5, -0.5, -0.4), colorSpace='rgb')

HP_Y, HP_W = -0.20, 0.22
def hp_bar(x, col):
    back = visual.Rect(win, width=HP_W, height=0.020, pos=(x, HP_Y),
                       fillColor=(-0.7, -0.7, -0.6), lineColor=(-0.3, -0.3, -0.2), colorSpace='rgb')
    fill = visual.Rect(win, width=HP_W, height=0.015, pos=(x, HP_Y), fillColor=col, lineColor=None, colorSpace='rgb')
    return back, fill

our_hp_bg, our_hp_fg = hp_bar(OUR_X, (0.2, 0.85, 0.3))
opp_hp_bg, opp_hp_fg = hp_bar(OPP_X, (0.9, 0.35, 0.2))

flash = visual.TextStim(win, text='', pos=(0.05, -0.30), height=0.055, color=(1.0, 0.9, 0.2), bold=True, colorSpace='rgb')
you_lbl = visual.TextStim(win, text='YOU  LV 1', pos=(OUR_X, HP_Y + 0.032), height=0.022, color=(0.4, 0.8, 1.0), colorSpace='rgb')
opp_lbl = visual.TextStim(win, text='ENEMY  LV 1', pos=(OPP_X, HP_Y + 0.032), height=0.022, color=(0.9, 0.5, 0.4), colorSpace='rgb')
pips = [visual.Circle(win, radius=0.010, pos=(OPP_X - 0.02 + i * 0.02, HP_Y - 0.035),
                      lineColor=(0.6, 0.35, 0.3), fillColor=(-0.6, -0.6, -0.5), colorSpace='rgb') for i in range(WIN_STREAK)]
pips_lbl = visual.TextStim(win, text='enemy powers up in', pos=(OPP_X, HP_Y - 0.065), height=0.016, color=(0.6, 0.4, 0.35), colorSpace='rgb')


def show_message(text):
    msg = visual.TextStim(win, text=text, height=0.042, color='white', wrapWidth=1.5, alignText='center', colorSpace='rgb')
    msg.draw(); win.flip()
    event.clearEvents()
    while True:
        k = event.getKeys()
        if 'space' in k:
            return
        if 'escape' in k:
            win.close(); core.quit()


show_message(
    "SCROLL the mouse wheel UP to raise the line.\n\n"
    "Your control is DELAYED by 8 seconds - what you see now is what you did 8 s ago.\n\n"
    "Keep the line HIGH to fill your POWER meter and LEVEL UP your robot (bigger, stronger).\n"
    "The enemy is even at first; each time you win 3 fights in a row, it levels up too.\n\n"
    "Press SPACE to start.")

# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------
clock = core.Clock(); clock.reset()
effort = 0.0
noise = 0.0
history = deque([(0.0, 50.0)])
ys = deque([sy(50)] * N_TRACE, maxlen=N_TRACE)
last_t = 0.0
next_sample = 0.0
next_dip = random.uniform(DIP_MIN, DIP_MAX)
next_log = 0.0

meter = 0.0
our_level = 1
enemy_level = 1
streak = 0
wins, losses = 0, 0
our_hp, opp_hp = GAME_HP, GAME_HP
our_atk_t, opp_atk_t = ATTACK_INTERVAL, ATTACK_INTERVAL
our_lunge, opp_lunge = 0.0, 0.0
flash_until, flash_txt = 0.0, ''
decay_k = math.log(2) / EFFORT_HALFLIFE
mouse.getWheelRel()

while True:
    now = clock.getTime()
    dt = min(now - last_t, 0.05)
    last_t = now
    if now >= SESSION_DURATION or 'escape' in event.getKeys():
        break

    # ---------------- signal ----------------
    dy = mouse.getWheelRel()[1]
    if INVERT_WHEEL:
        dy = -dy
    effort += dy * SCROLL_GAIN
    effort *= math.exp(-decay_k * dt)
    if now >= next_dip:
        if random.random() < P_RETURN:
            effort -= random.uniform(*RETURN_FRAC) * effort
        else:
            effort -= random.uniform(*DROP_AMOUNT)
        next_dip = now + random.uniform(DIP_MIN, DIP_MAX)
    effort = max(-10.0, min(55.0, effort))

    noise += (-noise / NOISE_TAU) * dt + NOISE_SD * math.sqrt(2.0 / NOISE_TAU) * math.sqrt(dt) * random.gauss(0, 1)
    control = max(0.0, min(100.0, 50.0 + effort + noise))
    history.append((now, control))

    # ---------------- exact 8 s delay ----------------
    tt = now - DELAY
    while len(history) >= 2 and history[1][0] <= tt:
        history.popleft()
    if len(history) >= 2 and history[0][0] <= tt <= history[1][0]:
        a, b = history[0], history[1]
        f = (tt - a[0]) / (b[0] - a[0]) if b[0] != a[0] else 0.0
        disp = a[1] + f * (b[1] - a[1])
    else:
        disp = history[0][1]

    g = 0
    while now >= next_sample and g < 5:
        ys.append(sy(disp)); next_sample += SAMPLE_DT; g += 1

    # ---------------- OUR meter -> our level ----------------
    if our_level < MAX_LEVEL:
        meter += disp * dt                      # fill rate = the (delayed) line value
        need = our_cost(our_level)
        if meter >= need:
            meter -= need
            our_level += 1
            flash_txt, flash_until = 'YOU LEVEL UP!', now + 1.3
        meter_prog = min(1.0, meter / our_cost(our_level)) if our_level < MAX_LEVEL else 1.0
    else:
        meter_prog = 1.0

    # ---------------- fight (strength set by level) ----------------
    our_str = strength(our_level)
    opp_str = strength(enemy_level)
    our_atk_t -= dt
    if our_atk_t <= 0:
        our_atk_t = ATTACK_INTERVAL * random.uniform(0.85, 1.15)
        opp_hp -= DMG_K * our_str * random.uniform(1 - DMG_JITTER, 1 + DMG_JITTER)
        our_lunge = 0.15
    opp_atk_t -= dt
    if opp_atk_t <= 0:
        opp_atk_t = ATTACK_INTERVAL * random.uniform(0.85, 1.15)
        our_hp -= DMG_K * opp_str * random.uniform(1 - DMG_JITTER, 1 + DMG_JITTER)
        opp_lunge = 0.15
    our_lunge = max(0.0, our_lunge - dt)
    opp_lunge = max(0.0, opp_lunge - dt)

    # ---------------- resolve a game ----------------
    if opp_hp <= 0:
        wins += 1
        streak += 1
        our_hp = opp_hp = GAME_HP
        our_atk_t = opp_atk_t = ATTACK_INTERVAL
        if streak >= WIN_STREAK and enemy_level < MAX_LEVEL:
            enemy_level += 1
            streak = 0
            flash_txt, flash_until = 'ENEMY LEVELS UP!', now + 1.3
        else:
            flash_txt, flash_until = 'WIN!', now + 0.8
    elif our_hp <= 0:
        losses += 1
        streak = 0
        our_hp = opp_hp = GAME_HP
        our_atk_t = opp_atk_t = ATTACK_INTERVAL
        flash_txt, flash_until = 'LOST!', now + 0.8

    # ---------------- draw ----------------
    frame_top.draw(); frame_bot.draw(); mid_line.draw()
    trace.vertices = np.column_stack([xs, np.array(ys)])
    trace.draw()
    ground_line.draw()

    bar_out.draw()
    fh = (BAR_H - 0.010) * meter_prog
    bar_fill.height = max(0.0, fh)
    bar_fill.pos = (BAR_X, BAR_BOT + 0.005 + fh / 2.0)
    bar_fill.draw()
    pow_lbl.draw()
    our_lvl_lbl.text = 'LV %d' % our_level
    our_lvl_lbl.draw()

    wl_lbl.text = 'W %d   L %d' % (wins, losses)
    wl_lbl.draw()

    ox = 0.035 * (our_lunge / 0.15)
    px = -0.035 * (opp_lunge / 0.15)
    bob = 0.003 * math.sin(now * 4.0)
    place(ours, OUR_X + ox, GROUND + bob, scale_for(our_level, BASE_SCALE), flip=1)
    place(opp,  OPP_X + px, GROUND - bob, scale_for(enemy_level, BASE_SCALE), flip=-1)
    for p in ours: p.draw()
    for p in opp:  p.draw()

    for bg, fg, hp, x in ((our_hp_bg, our_hp_fg, our_hp, OUR_X),
                          (opp_hp_bg, opp_hp_fg, opp_hp, OPP_X)):
        bg.draw()
        frac = max(0.0, min(1.0, hp / GAME_HP))
        fg.width = max(0.001, HP_W * frac)
        fg.pos = (x - HP_W / 2.0 + fg.width / 2.0, HP_Y)
        fg.draw()
    you_lbl.text = 'YOU  LV %d' % our_level
    opp_lbl.text = 'ENEMY  LV %d' % enemy_level
    you_lbl.draw(); opp_lbl.draw()
    pips_lbl.draw()
    for i, pip in enumerate(pips):
        pip.fillColor = (0.9, 0.5, 0.2) if i < streak else (-0.6, -0.6, -0.5)
        pip.draw()

    if now < flash_until:
        flash.text = flash_txt
        flash.draw()

    win.flip()

    if now >= next_log:
        lw.writerow(["%.2f" % now, "%.2f" % control, "%.2f" % disp, "%.1f" % meter,
                     our_level, enemy_level, streak, "%.1f" % our_hp, "%.1f" % opp_hp, wins, losses])
        next_log += 0.1

# ---------------------------------------------------------------------------
log_file.close()
with open(base + "_summary.csv", 'w', newline='') as sf:
    sw = csv.writer(sf)
    sw.writerow(['participant', 'session', 'duration_s', 'our_level', 'enemy_level', 'wins', 'losses'])
    sw.writerow([pid, ses, "%.1f" % clock.getTime(), our_level, enemy_level, wins, losses])

show_message("Done!\n\nYour level: %d\nEnemy level: %d\nWins: %d    Losses: %d\n\nPress SPACE to exit."
             % (our_level, enemy_level, wins, losses))
win.close()
core.quit()
