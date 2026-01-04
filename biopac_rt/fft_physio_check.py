import argparse
import numpy as np

# ==============================
# ADD THESE HELPERS (paste near the top, after imports)
# ==============================
import numpy as np

def _band_mask(freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (freqs >= lo) & (freqs <= hi)

def _peak_candidates_in_band(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float, k: int = 5):
    m = _band_mask(freqs, lo, hi)
    f = freqs[m]
    p = psd[m]
    if f.size == 0:
        return []
    # take top-k by power
    idx = np.argsort(p)[::-1][:k]
    out = [(float(f[i]), float(p[i])) for i in idx]
    # sort back by frequency just for readability? keep by power order
    return out

def _prominence_db(peak_power: float, band_powers: np.ndarray) -> float:
    # median of band is robust to harmonics/spikes
    med = float(np.median(band_powers)) if band_powers.size else np.nan
    if not np.isfinite(med) or med <= 0 or not np.isfinite(peak_power) or peak_power <= 0:
        return float("nan")
    return float(10.0 * np.log10(peak_power / med))

def summarize_peak_quality(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float, label: str):
    m = _band_mask(freqs, lo, hi)
    if not np.any(m):
        print(f"\n[{label} quality]")
        print("  (no frequencies in band)")
        return

    band_p = psd[m]
    cands = _peak_candidates_in_band(freqs, psd, lo, hi, k=5)

    # If everything is flat/noisy, cands still exists, but prominence will be small
    print(f"\n[{label} quality]")
    if not cands:
        print("  (no peaks found)")
        return

    # best + second best by power
    best_f, best_p = cands[0]
    best_db = _prominence_db(best_p, band_p)

    # Find second-best that is not basically the same bin
    second = None
    for f2, p2 in cands[1:]:
        if abs(f2 - best_f) > 1e-12:
            second = (f2, p2)
            break

    print(f"  Band median power: {float(np.median(band_p)):.3e}")
    print(f"  Best peak:   {best_f:7.3f} Hz ({best_f*60:6.1f} bpm)  power={best_p:.3e}  prominence={best_db:6.2f} dB")
    if second is not None:
        f2, p2 = second
        db2 = _prominence_db(p2, band_p)
        print(f"  2nd peak:    {f2:7.3f} Hz ({f2*60:6.1f} bpm)  power={p2:.3e}  prominence={db2:6.2f} dB")
    else:
        print("  2nd peak:    (none distinct)")

    # quick guidance
    # (not “medical”, just signal quality heuristics)
    if np.isfinite(best_db):
        if best_db < 3:
            qual = "weak/flat (likely noisy or motion-dominated)"
        elif best_db < 6:
            qual = "usable but not great"
        elif best_db < 10:
            qual = "good"
        else:
            qual = "excellent"
        print(f"  Heuristic: {qual} (prominence dB)")

def _hann(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones((n,), dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / (n - 1))

def welch_psd_numpy(x: np.ndarray, fs: float, nperseg: int, noverlap: int):
    """
    Small Welch PSD (no scipy). Returns (freqs, psd).
    Uses Hann window + mean removal per segment.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < max(8, nperseg):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    step = max(1, nperseg - noverlap)
    win = _hann(nperseg)
    win_pow = np.sum(win**2)

    # collect periodograms
    psd_acc = None
    nseg = 0
    for start in range(0, x.size - nperseg + 1, step):
        seg = x[start:start + nperseg]
        seg = seg - np.mean(seg)
        segw = seg * win
        spec = np.fft.rfft(segw)
        p = (np.abs(spec) ** 2) / (fs * win_pow)
        if psd_acc is None:
            psd_acc = p
        else:
            psd_acc += p
        nseg += 1

    if psd_acc is None or nseg == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    psd = psd_acc / float(nseg)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return freqs, psd

def window_peak_stability(x: np.ndarray, fs: float, lo: float, hi: float,
                          win_s: float = 60.0, overlap: float = 0.5):
    """
    Sliding-window peak frequency (in band). Returns bpm list.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < int(fs * win_s):
        return []

    nperwin = int(round(fs * win_s))
    step = int(round(nperwin * (1.0 - overlap)))
    step = max(1, step)

    bpms = []
    # Welch inside each window: use 4s segments (good freq res for 0-3 Hz)
    nperseg = int(round(fs * 4.0))
    nperseg = max(256, min(nperseg, nperwin))
    noverlap = int(round(nperseg * 0.5))

    for start in range(0, x.size - nperwin + 1, step):
        w = x[start:start + nperwin]
        freqs, psd = welch_psd_numpy(w, fs=fs, nperseg=nperseg, noverlap=noverlap)
        if freqs.size == 0:
            continue
        m = _band_mask(freqs, lo, hi)
        if not np.any(m):
            continue
        f_band = freqs[m]
        p_band = psd[m]
        # --- harmonic-safe peak selection ---
        K = 5
        idx = np.argsort(p_band)[::-1][:K]
        p_max = p_band[idx[0]]

        # keep peaks within 6 dB of strongest (≈ factor 4)
        keep = idx[p_band[idx] >= p_max / 4.0]
        f0 = float(np.min(f_band[keep]))

        bpms.append(f0 * 60.0)
    return bpms

def print_window_stability(x: np.ndarray, fs: float, lo: float, hi: float, label: str):
    bpms = window_peak_stability(x, fs, lo, hi, win_s=60.0, overlap=0.5)
    print(f"\n[{label} peak stability | 60s windows | 50% overlap]")
    if len(bpms) < 3:
        print("  (not enough windows to summarize)")
        return
    arr = np.asarray(bpms, dtype=np.float64)
    med = float(np.median(arr))
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    iqr = q75 - q25
    print(f"  windows: {len(bpms)}")
    print(f"  bpm median: {med:6.1f} | IQR: {iqr:5.1f} (Q25={q25:5.1f}, Q75={q75:5.1f})")


def read_5col(path, max_rows=None):
    rows = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.replace(",", " ").split()
            if len(parts) < 5:
                continue
            try:
                vals = [float(x) for x in parts[:5]]
            except ValueError:
                continue
            rows.append(vals)
            if max_rows is not None and len(rows) >= max_rows:
                break
    return np.asarray(rows, dtype=np.float64)

def welch_psd(x, fs, nperseg=None):
    try:
        from scipy.signal import welch
        if nperseg is None:
            # ~8s window; enough resolution for 0–5 Hz but not too long
            nperseg = int(min(len(x), max(256, int(fs * 8))))
        f, pxx = welch(x, fs=fs, nperseg=nperseg, detrend="constant", scaling="density")
        return f, pxx
    except Exception:
        x = x - np.nanmean(x)
        n = len(x)
        freqs = np.fft.rfftfreq(n, d=1.0/fs)
        spec = (np.abs(np.fft.rfft(x)) ** 2) / max(n, 1)
        return freqs, spec

def top_peaks(f, pxx, band, k=5):
    lo, hi = band
    m = (f >= lo) & (f <= hi) & np.isfinite(pxx)
    ff = f[m]
    pp = pxx[m]
    if ff.size == 0:
        return []
    idx = np.argsort(pp)[::-1][:k]
    return [(float(ff[i]), float(pp[i])) for i in idx]

def hz_to_bpm(hz):
    return hz * 60.0

def prep(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.nanmean(x)
    s = np.nanstd(x)
    # clip extreme spikes so PSD scaling doesn't get ruined
    if np.isfinite(s) and s > 0:
        x = np.clip(x, -10*s, 10*s)
    return x

def maybe_downsample(x, fs_in, fs_out):
    if fs_out <= 0 or fs_out >= fs_in:
        return x, fs_in
    try:
        from scipy.signal import resample_poly
        from fractions import Fraction
        frac = Fraction(fs_out / fs_in).limit_denominator(1000)
        up, down = frac.numerator, frac.denominator
        y = resample_poly(x, up, down)
        return y, fs_out
    except Exception as e:
        raise RuntimeError(f"Downsample requested but failed (need scipy): {e}")

def main():
    ap = argparse.ArgumentParser(
        description="Quick PSD/FFT sanity check for BIOPAC 5-col physio file (trigger, resp, ppg, ecg, eda)."
    )
    ap.add_argument("--csv", required=True)
    ap.add_argument("--fs", type=float, required=True, help="TRUE sample rate of the file (Hz).")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--maxhz", type=float, default=5.0)
    ap.add_argument("--downsample", type=float, default=0.0,
                    help="If >0, actually resample to this Hz before PSD (recommended).")
    args = ap.parse_args()

    fs0 = float(args.fs)
    n = int(round(args.seconds * fs0))
    data = read_5col(args.csv, max_rows=n)
    if data.size == 0:
        raise RuntimeError("No readable rows found.")

    trig = data[:, 0]
    resp = data[:, 1]
    ppg  = data[:, 2]
    ecg  = data[:, 3]

    # Optional REAL downsampling (same for all channels)
    resp, fs = maybe_downsample(resp, fs0, float(args.downsample))
    ppg,  _  = maybe_downsample(ppg,  fs0, float(args.downsample))
    ecg,  _  = maybe_downsample(ecg,  fs0, float(args.downsample))
    trig, _  = maybe_downsample(trig, fs0, float(args.downsample))

    resp0 = prep(resp)
    ppg0  = prep(ppg)
    ecg0  = prep(ecg)

    f_r, p_r = welch_psd(resp0, fs)
    f_p, p_p = welch_psd(ppg0,  fs)
    f_e, p_e = welch_psd(ecg0,  fs)

    # Expected ranges (rough adult, resting-ish)
    resp_band_hz = (0.05, 0.6)    # 3–36 bpm
    card_band_hz = (0.7, 3.0)     # 42–180 bpm

    print("=" * 70)
    print(f"[INFO] Used fs={fs:.3f} Hz (file_fs={fs0:.3f} Hz), seconds≈{args.seconds:g}, samples={len(resp0)}")
    if args.downsample and args.downsample > 0:
        print(f"[INFO] REAL downsample applied: {fs0:.1f} -> {fs:.1f} Hz")
    else:
        print("[INFO] No resampling performed (frequency axis is truthful).")
    print()

    print("[EXPECTED ranges]")
    print(f"  Resp:   {resp_band_hz[0]:.2f}–{resp_band_hz[1]:.2f} Hz  (~{hz_to_bpm(resp_band_hz[0]):.0f}–{hz_to_bpm(resp_band_hz[1]):.0f} bpm)")
    print(f"  Card:   {card_band_hz[0]:.2f}–{card_band_hz[1]:.2f} Hz  (~{hz_to_bpm(card_band_hz[0]):.0f}–{hz_to_bpm(card_band_hz[1]):.0f} bpm)")
    print("  Note: during tasks/anxiety heart can go higher; respiration can vary a lot.")
    print()

    def dump_peaks(name, f, pxx, band, k=5):
        peaks = top_peaks(f, pxx, band, k=k)
        print(f"[{name} top peaks in {band[0]:.2f}–{band[1]:.2f} Hz]")
        if not peaks:
            print("  (none found in band)")
            return
        for hz, pw in peaks:
            print(f"  {hz:7.3f} Hz  ({hz_to_bpm(hz):6.1f} bpm)   power={pw:.3e}")

    dump_peaks("RESP", f_r, p_r, resp_band_hz, k=5)
    print()
    dump_peaks("PPG ", f_p, p_p, card_band_hz, k=5)
    print()
    dump_peaks("ECG ", f_e, p_e, card_band_hz, k=5)

    # ------------------------------
    # EXTRA sanity analysis (added)
    # ------------------------------
    summarize_peak_quality(f_r, p_r, resp_band_hz[0], resp_band_hz[1], "RESP")
    summarize_peak_quality(f_p, p_p, card_band_hz[0], card_band_hz[1], "PPG ")
    summarize_peak_quality(f_e, p_e, card_band_hz[0], card_band_hz[1], "ECG ")

    print_window_stability(resp0, fs, resp_band_hz[0], resp_band_hz[1], "RESP")
    print_window_stability(ppg0,  fs, card_band_hz[0], card_band_hz[1], "PPG ")
    print_window_stability(ecg0,  fs, card_band_hz[0], card_band_hz[1], "ECG ")

    print("=" * 70)


    if args.plot:
        import matplotlib.pyplot as plt

        def plot_psd(f, p, title):
            m = (f >= 0) & (f <= args.maxhz) & np.isfinite(p) & (p > 0)
            plt.figure()
            plt.plot(f[m], np.log10(p[m]))
            plt.xlabel("Hz")
            plt.ylabel("log10(PSD)")
            plt.title(title)

        plot_psd(f_p, p_p, "PPG Welch PSD (log10)")
        plot_psd(f_e, p_e, "ECG Welch PSD (log10)")
        plot_psd(f_r, p_r, "RESP Welch PSD (log10)")
        plt.show()



if __name__ == "__main__":
    main()
