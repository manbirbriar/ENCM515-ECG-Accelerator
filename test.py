"""
ENCM 515 - Group Project
Real-Time Wearable ECG Signal Processing Accelerator
MIT-BIH Arrhythmia Database

Run with:
    pip install wfdb numpy scipy matplotlib
    python ecg_main.py

What this script covers (5 ENCM 515 topics):
  1. DSP execution model      — FIR filter as a MAC pipeline
  2. Pipeline & hazard analysis — latency, II, CPI, RAW hazard
  3. Loop-level optimization  — loop unrolling, NumPy vectorization, speedup
  4. Precision-aware compute  — float64 vs float32 vs int16 fixed-point
  5. Performance modeling     — R-peak detection, Amdahl's Law
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import firwin, lfilter
import wfdb
import time

# ═══════════════════════════════════════════════════════════════
# GLOBAL PARAMETERS
# ═══════════════════════════════════════════════════════════════
RECORD       = '100'    # MIT-BIH record (try also '101', '105', '200')
DURATION_SEC = 10       # Seconds to load
FS           = 360      # MIT-BIH sampling rate (Hz)
N_SAMPLES    = FS * DURATION_SEC
N_TAPS       = 101      # FIR filter length (odd)
LOW_CUT      = 0.5      # Bandpass lower edge (Hz)
HIGH_CUT     = 40.0     # Bandpass upper edge (Hz)
N_RUNS       = 5        # Benchmark repetitions
CLOCK_MHZ    = 100      # Assumed DSP clock for pipeline model


# ═══════════════════════════════════════════════════════════════
# STEP 1 — ECG SIGNAL LOADING + FIR FILTER BASELINE  (Member 1)
# ═══════════════════════════════════════════════════════════════
def step1_load_and_filter():
    print("\n" + "═"*60)
    print("  STEP 1 — Signal Loading + FIR Filter Baseline")
    print("═"*60)

    # ── Load MIT-BIH record ──────────────────────────────────
    print(f"[MIT-BIH] Loading record {RECORD} from PhysioNet...")
    record = wfdb.rdrecord(RECORD, pn_dir='mitdb', sampto=N_SAMPLES)
    ann    = wfdb.rdann(RECORD, 'atr', pn_dir='mitdb', sampto=N_SAMPLES)

    signal = record.p_signal[:, 0]          # MLII lead
    t      = np.arange(len(signal)) / FS

    print(f"  Samples: {len(signal)}  |  Lead: {record.sig_name[0]}")
    print(f"  Expert-annotated beats in window: {len(ann.sample)}")

    # ── Design FIR bandpass filter ───────────────────────────
    nyq    = FS / 2.0
    coeffs = firwin(N_TAPS, [LOW_CUT/nyq, HIGH_CUT/nyq],
                    pass_zero=False, window='hamming')
    print(f"  FIR filter: {N_TAPS} taps, {LOW_CUT}–{HIGH_CUT} Hz")
    print(f"  MACs per sample: {N_TAPS}  |  Total MACs: {N_TAPS*N_SAMPLES:,}")

    # ── Baseline: pure Python FIR loop ───────────────────────
    def fir_python(sig, h):
        n, m = len(sig), len(h)
        out  = np.zeros(n)
        for i in range(m, n):
            acc = 0.0
            for k in range(m):
                acc += h[k] * sig[i - k]
            out[i] = acc
        return out

    # ── Optimised: NumPy / scipy vectorized ──────────────────
    def fir_numpy(sig, h):
        return lfilter(h, 1.0, sig)

    # ── Benchmark ────────────────────────────────────────────
    times_py, times_np = [], []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        out_py = fir_python(signal, coeffs)
        times_py.append((time.perf_counter() - t0) * 1000)

    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        out_np = fir_numpy(signal, coeffs)
        times_np.append((time.perf_counter() - t0) * 1000)

    avg_py = np.mean(times_py)
    avg_np = np.mean(times_np)
    speedup_s1 = avg_py / avg_np

    print(f"\n  Pure Python: {avg_py:.2f} ms  |  NumPy: {avg_np:.2f} ms"
          f"  |  Speedup: {speedup_s1:.1f}x")

    # ── Plot ──────────────────────────────────────────────────
    show = int(3 * FS)
    fig, axes = plt.subplots(3, 1, figsize=(13, 9))
    fig.suptitle(f'Step 1 — MIT-BIH Record {RECORD} | FIR Bandpass Filter', fontsize=12)

    axes[0].plot(t[:show], signal[:show], color='steelblue', lw=0.8, label='Raw ECG')
    r_win = ann.sample[ann.sample < show]
    axes[0].scatter(t[r_win], signal[r_win], color='red', s=40, zorder=5,
                    label='Annotated R-peaks')
    axes[0].set_title('Raw MIT-BIH signal (first 3 s)')
    axes[0].set_ylabel('Amplitude (mV)'); axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t[:show], out_np[:show], color='seagreen', lw=0.9, label='Filtered')
    axes[1].scatter(t[r_win], out_np[r_win], color='red', s=40, zorder=5,
                    label='Annotated R-peaks')
    axes[1].set_title('After FIR bandpass filter')
    axes[1].set_ylabel('Amplitude (mV)'); axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    bars = axes[2].bar(['Pure Python', 'NumPy'], [avg_py, avg_np],
                       color=['tomato', 'seagreen'], width=0.4, edgecolor='gray')
    for bar, val in zip(bars, [avg_py, avg_np]):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.2f} ms', ha='center', fontsize=10)
    axes[2].set_title(f'Filter latency — {speedup_s1:.1f}x speedup')
    axes[2].set_ylabel('Latency (ms)'); axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('ecg_step1.png', dpi=150); plt.show()
    print("  [Saved ecg_step1.png]")

    return signal, out_np, coeffs, t, ann, avg_py, avg_np, speedup_s1


# ═══════════════════════════════════════════════════════════════
# STEP 2 — DSP PIPELINE MODELING  (Member 2)
# ═══════════════════════════════════════════════════════════════
def step2_pipeline_model(avg_py, avg_np):
    print("\n" + "═"*60)
    print("  STEP 2 — DSP Pipeline Modeling")
    print("═"*60)

    cycle_ns = 1000 / CLOCK_MHZ   # ns per clock cycle
    n_stages = 4                   # Fetch, Multiply, Accumulate, Write-back

    # ── Ideal pipeline (II = 1) ───────────────────────────────
    II_ideal      = 1
    total_ideal   = N_SAMPLES * II_ideal + (n_stages - 1)
    cpi_ideal     = total_ideal / N_SAMPLES
    tp_ideal      = N_SAMPLES / total_ideal
    time_ideal_ms = total_ideal * cycle_ns / 1e6

    # ── Stalled pipeline (RAW hazard, 1 stall per MAC) ────────
    II_stalled      = N_TAPS * 2          # multiply + stall per tap
    total_stalled   = N_SAMPLES * II_stalled + (n_stages - 1)
    cpi_stalled     = total_stalled / N_SAMPLES
    tp_stalled      = N_SAMPLES / total_stalled
    time_stalled_ms = total_stalled * cycle_ns / 1e6
    stall_frac      = (N_TAPS * N_SAMPLES) / total_stalled

    print(f"\n  {'Metric':<32} {'Ideal':>12} {'Stalled':>12}")
    print("  " + "-"*58)
    rows = [
        ("Initiation interval (II)",  f"{II_ideal}",           f"{II_stalled}"),
        ("CPI",                        f"{cpi_ideal:.2f}",      f"{cpi_stalled:.2f}"),
        ("Throughput (samp/cycle)",    f"{tp_ideal:.4f}",       f"{tp_stalled:.6f}"),
        ("Modeled time @ 100 MHz",     f"{time_ideal_ms:.4f} ms", f"{time_stalled_ms:.2f} ms"),
        ("Stall fraction",             "—",                     f"{stall_frac*100:.1f}%"),
    ]
    for label, iv, sv in rows:
        print(f"  {label:<32} {iv:>12} {sv:>12}")

    print(f"\n  Measured Python time: {avg_py:.2f} ms  (models stalled case)")
    print(f"  Measured NumPy time:  {avg_np:.2f} ms  (models ideal case)")

    # ── Pipeline timing diagram ───────────────────────────────
    stage_names  = ['Fetch', 'Multiply', 'Accumulate', 'Write']
    stage_colors = ['#4A90D9', '#5BA85B', '#C97C3A', '#A855A8']

    def draw_timing(ax, title, ii, n_show):
        n_cycles = n_show * ii + n_stages + 2
        ax.set_xlim(0, n_cycles); ax.set_ylim(-0.5, n_show - 0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Clock cycle'); ax.set_ylabel('Sample')
        ax.set_yticks(range(n_show))
        ax.set_yticklabels([f'x[{i}]' for i in range(n_show)], fontsize=8)
        ax.grid(True, axis='x', alpha=0.2); ax.set_facecolor('#f8f8f8')
        for s in range(n_show):
            for si, (sn, col) in enumerate(zip(stage_names, stage_colors)):
                cyc = s * ii + si
                if cyc < n_cycles:
                    rect = mpatches.FancyBboxPatch(
                        (cyc + 0.05, s - 0.38), 0.9, 0.76,
                        boxstyle="round,pad=0.02",
                        facecolor=col, edgecolor='white', lw=0.8, alpha=0.85)
                    ax.add_patch(rect)
                    ax.text(cyc + 0.5, s, sn[0], ha='center', va='center',
                            fontsize=7, color='white', fontweight='bold')
        patches = [mpatches.Patch(color=c, label=n)
                   for c, n in zip(stage_colors, stage_names)]
        ax.legend(handles=patches, loc='upper right', fontsize=8,
                  ncol=4, framealpha=0.9)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    fig.suptitle('Step 2 — FIR Pipeline Timing Diagram', fontsize=12)
    draw_timing(axes[0], 'Ideal pipeline (II = 1, no stalls)', ii=1, n_show=8)
    draw_timing(axes[1], 'Stalled pipeline (II = 6, RAW hazard, N_taps=3 for clarity)',
                ii=6, n_show=4)
    plt.tight_layout()
    plt.savefig('ecg_step2.png', dpi=150); plt.show()
    print("  [Saved ecg_step2.png]")

    return cpi_ideal, cpi_stalled, stall_frac, II_ideal, II_stalled


# ═══════════════════════════════════════════════════════════════
# STEP 3 — LOOP UNROLLING & OPTIMIZATION  (Member 3)
# ═══════════════════════════════════════════════════════════════
def step3_loop_unrolling(signal, coeffs):
    print("\n" + "═"*60)
    print("  STEP 3 — Loop Unrolling & Optimization")
    print("═"*60)

    n = len(signal)
    m = len(coeffs)

    # ── Baseline: no unrolling ────────────────────────────────
    def fir_unroll_1(sig, h):
        out = np.zeros(len(sig))
        for i in range(len(h), len(sig)):
            acc = 0.0
            for k in range(len(h)):
                acc += h[k] * sig[i - k]
            out[i] = acc
        return out

    # ── 2x unrolling ──────────────────────────────────────────
    def fir_unroll_2(sig, h):
        out = np.zeros(len(sig))
        m   = len(h)
        for i in range(m, len(sig)):
            acc0 = acc1 = 0.0
            for k in range(0, m - 1, 2):
                acc0 += h[k]   * sig[i - k]
                acc1 += h[k+1] * sig[i - k - 1]
            if m % 2 != 0:
                acc0 += h[m-1] * sig[i - m + 1]
            out[i] = acc0 + acc1
        return out

    # ── 4x unrolling ──────────────────────────────────────────
    def fir_unroll_4(sig, h):
        out = np.zeros(len(sig))
        m   = len(h)
        for i in range(m, len(sig)):
            acc0 = acc1 = acc2 = acc3 = 0.0
            for k in range(0, m - 3, 4):
                acc0 += h[k]   * sig[i - k]
                acc1 += h[k+1] * sig[i - k - 1]
                acc2 += h[k+2] * sig[i - k - 2]
                acc3 += h[k+3] * sig[i - k - 3]
            for k in range((m // 4) * 4, m):
                acc0 += h[k] * sig[i - k]
            out[i] = acc0 + acc1 + acc2 + acc3
        return out

    # ── NumPy vectorized (best case reference) ────────────────
    def fir_numpy(sig, h):
        return lfilter(h, 1.0, sig)

    implementations = [
        ('No unrolling (1x)',  fir_unroll_1),
        ('2x unrolled',        fir_unroll_2),
        ('4x unrolled',        fir_unroll_4),
        ('NumPy vectorized',   fir_numpy),
    ]

    results = {}
    print(f"\n  {'Implementation':<22} {'Avg (ms)':>10} {'Speedup':>10}")
    print("  " + "-"*44)

    baseline_ms = None
    for name, fn in implementations:
        times = []
        for _ in range(N_RUNS):
            t0  = time.perf_counter()
            out = fn(signal, coeffs)
            times.append((time.perf_counter() - t0) * 1000)
        avg = np.mean(times)
        if baseline_ms is None:
            baseline_ms = avg
        speedup = baseline_ms / avg
        results[name] = {'avg_ms': avg, 'speedup': speedup}
        print(f"  {name:<22} {avg:>10.2f} {speedup:>10.1f}x")

    # ── Plot ──────────────────────────────────────────────────
    names    = list(results.keys())
    avgs     = [results[n]['avg_ms'] for n in names]
    speedups = [results[n]['speedup'] for n in names]
    colors   = ['tomato', '#E8A838', '#5BA85B', 'steelblue']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Step 3 — Loop Unrolling Optimization', fontsize=12)

    axes[0].bar(names, avgs, color=colors, edgecolor='gray', width=0.5)
    for i, v in enumerate(avgs):
        axes[0].text(i, v + 0.5, f'{v:.2f} ms', ha='center', fontsize=9)
    axes[0].set_title('Filter latency by implementation')
    axes[0].set_ylabel('Latency (ms)')
    axes[0].tick_params(axis='x', labelrotation=15)
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(names, speedups, color=colors, edgecolor='gray', width=0.5)
    for i, v in enumerate(speedups):
        axes[1].text(i, v + 0.2, f'{v:.1f}x', ha='center', fontsize=9)
    axes[1].set_title('Speedup over baseline (no unrolling)')
    axes[1].set_ylabel('Speedup')
    axes[1].tick_params(axis='x', labelrotation=15)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('ecg_step3.png', dpi=150); plt.show()
    print("  [Saved ecg_step3.png]")

    return results


# ═══════════════════════════════════════════════════════════════
# STEP 4 — PRECISION-AWARE COMPUTATION  (Member 4)
# ═══════════════════════════════════════════════════════════════
def step4_precision_analysis(signal, coeffs):
    print("\n" + "═"*60)
    print("  STEP 4 — Precision Analysis (float64 / float32 / int16)")
    print("═"*60)

    # ── Reference: float64 ───────────────────────────────────
    sig64   = signal.astype(np.float64)
    coef64  = coeffs.astype(np.float64)
    ref_out = lfilter(coef64, 1.0, sig64)

    # ── float32 ──────────────────────────────────────────────
    sig32   = signal.astype(np.float32)
    coef32  = coeffs.astype(np.float32)

    # ── int16 fixed-point ─────────────────────────────────────
    # Scale signal to int16 range [-32768, 32767]
    # Typical ECG amplitude is ±2 mV; scale factor maps mV → int16
    SCALE = 32767 / (np.max(np.abs(signal)) + 1e-9)
    sig16   = np.clip(np.round(signal * SCALE), -32768, 32767).astype(np.int16)
    # Scale coefficients to fixed-point (Q15 format: multiply by 2^15)
    Q = 2**14
    coef16  = np.clip(np.round(coeffs * Q), -32768, 32767).astype(np.int16)

    def fir_fixed_point(sig, h, q_factor):
        """Fixed-point FIR using integer arithmetic."""
        sig_i = sig.astype(np.int32)
        h_i   = h.astype(np.int32)
        n, m  = len(sig_i), len(h_i)
        out   = np.zeros(n, dtype=np.int32)
        for i in range(m, n):
            acc = np.int64(0)
            for k in range(m):
                acc += np.int64(h_i[k]) * np.int64(sig_i[i - k])
            out[i] = np.int32(acc >> int(np.log2(q_factor)))
        return out.astype(np.float64) / SCALE   # back to mV for comparison

    precisions = {
        'float64 (reference)': (sig64, coef64, None),
        'float32':             (sig32, coef32, None),
        'int16 fixed-point':   (sig16, coef16, Q),
    }

    print(f"\n  {'Precision':<24} {'Time (ms)':>10} {'Max err (mV)':>14} {'SNR (dB)':>10}")
    print("  " + "-"*60)

    precision_results = {}
    for name, (s, h, q) in precisions.items():
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            if q is not None:
                out = fir_fixed_point(s, h, q)
            else:
                out = lfilter(h.astype(np.float64) if name == 'float32'
                              else h, 1.0,
                              s.astype(np.float64) if name == 'float32'
                              else s)
            times.append((time.perf_counter() - t0) * 1000)

        avg    = np.mean(times)
        # Re-run once cleanly for error analysis
        if q is not None:
            out = fir_fixed_point(s, h, q)
        else:
            out = lfilter(h, 1.0, s)
            if name == 'float32':
                out = out.astype(np.float64)

        err     = np.abs(out - ref_out)
        max_err = np.max(err[N_TAPS:])    # skip filter startup transient
        signal_power = np.mean(ref_out[N_TAPS:] ** 2)
        noise_power  = np.mean(err[N_TAPS:] ** 2) + 1e-12
        snr_db = 10 * np.log10(signal_power / noise_power)

        precision_results[name] = {
            'avg_ms': avg, 'max_err': max_err, 'snr_db': snr_db, 'output': out
        }
        print(f"  {name:<24} {avg:>10.2f} {max_err:>14.6f} {snr_db:>10.2f}")

    # ── Plot ──────────────────────────────────────────────────
    show = int(3 * FS)
    t    = np.arange(N_SAMPLES) / FS

    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    fig.suptitle('Step 4 — Precision Analysis: float64 vs float32 vs int16', fontsize=12)

    colors_p = {'float64 (reference)': 'steelblue',
                'float32': 'seagreen',
                'int16 fixed-point': 'tomato'}

    for name, res in precision_results.items():
        axes[0].plot(t[:show], res['output'][:show],
                     label=name, color=colors_p[name],
                     lw=1.2 if 'reference' in name else 0.8,
                     alpha=1.0 if 'reference' in name else 0.75)
    axes[0].set_title('Filtered ECG by precision type (first 3 s)')
    axes[0].set_ylabel('Amplitude (mV)'); axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    for name, res in precision_results.items():
        if 'reference' not in name:
            err = np.abs(res['output'] - ref_out)
            axes[1].plot(t[:show], err[:show],
                         label=f'{name} error', color=colors_p[name], lw=0.8)
    axes[1].set_title('Absolute error vs float64 reference')
    axes[1].set_ylabel('Error (mV)'); axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    names_p  = list(precision_results.keys())
    snrs     = [precision_results[n]['snr_db'] for n in names_p]
    bars     = axes[2].bar(names_p, snrs,
                           color=[colors_p[n] for n in names_p],
                           edgecolor='gray', width=0.4)
    for bar, v in zip(bars, snrs):
        axes[2].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5, f'{v:.1f} dB',
                     ha='center', fontsize=10)
    axes[2].set_title('Signal-to-noise ratio by precision')
    axes[2].set_ylabel('SNR (dB)'); axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('ecg_step4.png', dpi=150); plt.show()
    print("  [Saved ecg_step4.png]")

    return precision_results


# ═══════════════════════════════════════════════════════════════
# STEP 5 — R-PEAK DETECTION + AMDAHL'S LAW  (Member 5)
# ═══════════════════════════════════════════════════════════════
def step5_rpeaks_and_amdahl(filtered_signal, ann, speedup_s1, stall_frac):
    print("\n" + "═"*60)
    print("  STEP 5 — R-Peak Detection + Amdahl's Law")
    print("═"*60)

    t = np.arange(N_SAMPLES) / FS

    # ── Pan-Tompkins R-peak detection ─────────────────────────
    def pan_tompkins(sig, fs):
        """
        Simplified Pan-Tompkins algorithm:
          1. Differentiate  → emphasises QRS slope
          2. Square         → all positive, amplifies large slopes
          3. Moving average → integrates energy over QRS width
          4. Threshold      → adaptive peak detection
        """
        # 1. Differentiate
        diff = np.diff(sig, prepend=sig[0])

        # 2. Square
        squared = diff ** 2

        # 3. Moving-window integration (~150 ms window)
        win = int(0.15 * fs)
        kernel = np.ones(win) / win
        integrated = np.convolve(squared, kernel, mode='same')

        # 4. Adaptive threshold peak detection
        threshold  = 0.5 * np.max(integrated)
        min_dist   = int(0.2 * fs)    # minimum 200 ms between peaks (max 300 BPM)
        peaks      = []
        last_peak  = -min_dist

        for i in range(1, len(integrated) - 1):
            if (integrated[i] > threshold and
                    integrated[i] >= integrated[i-1] and
                    integrated[i] >= integrated[i+1] and
                    i - last_peak >= min_dist):
                peaks.append(i)
                last_peak = i

        return np.array(peaks), integrated

    detected_peaks, integrated = pan_tompkins(filtered_signal, FS)

    # ── Accuracy vs MIT-BIH annotations ──────────────────────
    expert_peaks = ann.sample
    tolerance    = int(0.05 * FS)    # ±50 ms tolerance window

    true_pos  = 0
    matched   = set()
    for dp in detected_peaks:
        for i, ep in enumerate(expert_peaks):
            if abs(dp - ep) <= tolerance and i not in matched:
                true_pos += 1
                matched.add(i)
                break

    false_pos = len(detected_peaks) - true_pos
    false_neg = len(expert_peaks)   - true_pos

    sensitivity = true_pos / (true_pos + false_neg + 1e-9) * 100
    precision   = true_pos / (true_pos + false_pos + 1e-9) * 100
    f1          = 2 * sensitivity * precision / (sensitivity + precision + 1e-9)

    print(f"\n  R-peak detection (Pan-Tompkins):")
    print(f"  Expert annotations:  {len(expert_peaks)}")
    print(f"  Detected peaks:      {len(detected_peaks)}")
    print(f"  True positives:      {true_pos}")
    print(f"  Sensitivity:         {sensitivity:.1f}%")
    print(f"  Precision:           {precision:.1f}%")
    print(f"  F1 score:            {f1:.1f}%")

    # ── Amdahl's Law ──────────────────────────────────────────
    # Parallelisable fraction = fraction of time spent in FIR filter
    # (everything else: loading, peak detection, I/O = serial fraction)
    p_values     = np.linspace(0, 1, 200)      # parallelisable fraction
    n_processors = [1, 2, 4, 8, 16, 32, 64]   # number of parallel units

    def amdahl(p, n):
        return 1.0 / ((1 - p) + p / n)

    # Use stall_frac as our estimate of the parallelisable fraction
    # (stall cycles are the ones that benefit from pipelining / SIMD)
    p_estimated = stall_frac
    max_speedup = amdahl(p_estimated, n=float('inf'))  # theoretical ceiling

    print(f"\n  Amdahl's Law:")
    print(f"  Parallelisable fraction (p): {p_estimated*100:.1f}%  "
          f"(from Step 2 stall fraction)")
    print(f"  Theoretical max speedup:     {max_speedup:.2f}x")
    print(f"  Measured speedup (Step 1):   {speedup_s1:.1f}x")
    print(f"  {'N cores':<10} {'Predicted speedup':>18}")
    for n in n_processors:
        su = amdahl(p_estimated, n)
        print(f"  {n:<10} {su:>18.2f}x")

    # ── Plots ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Step 5 — R-Peak Detection + Amdahl\'s Law', fontsize=12)

    # Panel 1: Filtered ECG with detected and annotated peaks
    show = int(3 * FS)
    axes[0].plot(t[:show], filtered_signal[:show],
                 color='steelblue', lw=0.8, label='Filtered ECG')
    det_in_win = detected_peaks[detected_peaks < show]
    axes[0].scatter(t[det_in_win], filtered_signal[det_in_win],
                    color='seagreen', s=60, zorder=5, marker='^',
                    label='Detected R-peaks')
    exp_in_win = expert_peaks[expert_peaks < show]
    axes[0].scatter(t[exp_in_win], filtered_signal[exp_in_win],
                    color='red', s=30, zorder=4, marker='x',
                    label='Expert annotations')
    axes[0].set_title('Pan-Tompkins R-peak detection (first 3 s)')
    axes[0].set_xlabel('Time (s)'); axes[0].set_ylabel('Amplitude (mV)')
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    # Panel 2: Detection metrics bar chart
    metrics = ['Sensitivity', 'Precision', 'F1 Score']
    values  = [sensitivity, precision, f1]
    bars2   = axes[1].bar(metrics, values,
                          color=['steelblue', 'seagreen', 'darkorange'],
                          edgecolor='gray', width=0.4)
    for bar, v in zip(bars2, values):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
    axes[1].set_ylim(0, 115)
    axes[1].set_title('Detection accuracy vs MIT-BIH annotations')
    axes[1].set_ylabel('Score (%)'); axes[1].grid(True, alpha=0.3, axis='y')

    # Panel 3: Amdahl's Law curves
    colors_a = plt.cm.viridis(np.linspace(0.2, 0.9, len(n_processors)))
    for n, col in zip(n_processors, colors_a):
        speedups = [amdahl(p, n) for p in p_values]
        axes[2].plot(p_values * 100, speedups, color=col, lw=1.5, label=f'N={n}')
    axes[2].axvline(p_estimated * 100, color='red', lw=1.5, ls='--',
                    label=f'Our p = {p_estimated*100:.0f}%')
    axes[2].axhline(speedup_s1, color='orange', lw=1.5, ls=':',
                    label=f'Measured speedup = {speedup_s1:.1f}x')
    axes[2].set_title("Amdahl's Law — speedup vs parallelisable fraction")
    axes[2].set_xlabel('Parallelisable fraction p (%)')
    axes[2].set_ylabel('Speedup')
    axes[2].legend(fontsize=7, ncol=2); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ecg_step5.png', dpi=150); plt.show()
    print("  [Saved ecg_step5.png]")

    return sensitivity, precision, f1, max_speedup


# ═══════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════
def print_summary(avg_py, avg_np, speedup_s1,
                  cpi_ideal, cpi_stalled, stall_frac,
                  unroll_results,
                  precision_results,
                  sensitivity, precision_score, f1, max_speedup):
    print("\n" + "═"*60)
    print("  FINAL SUMMARY — ENCM 515 ECG Accelerator Project")
    print("═"*60)
    print(f"  Dataset:           MIT-BIH record {RECORD}, {DURATION_SEC}s @ {FS} Hz")
    print(f"  FIR filter:        {N_TAPS} taps, {LOW_CUT}–{HIGH_CUT} Hz bandpass")
    print()
    print(f"  [Step 1] Python baseline:     {avg_py:.2f} ms")
    print(f"  [Step 1] NumPy optimized:     {avg_np:.2f} ms")
    print(f"  [Step 1] Speedup:             {speedup_s1:.1f}x")
    print()
    print(f"  [Step 2] Ideal CPI:           {cpi_ideal:.2f}")
    print(f"  [Step 2] Stalled CPI:         {cpi_stalled:.2f}")
    print(f"  [Step 2] Stall fraction:      {stall_frac*100:.1f}%")
    print()
    for name, res in unroll_results.items():
        print(f"  [Step 3] {name:<22} {res['avg_ms']:.2f} ms  "
              f"({res['speedup']:.1f}x)")
    print()
    for name, res in precision_results.items():
        print(f"  [Step 4] {name:<24} SNR = {res['snr_db']:.1f} dB")
    print()
    print(f"  [Step 5] R-peak sensitivity:  {sensitivity:.1f}%")
    print(f"  [Step 5] R-peak precision:    {precision_score:.1f}%")
    print(f"  [Step 5] F1 score:            {f1:.1f}%")
    print(f"  [Step 5] Amdahl max speedup:  {max_speedup:.2f}x  "
          f"(p = {stall_frac*100:.0f}%)")
    print("═"*60)
    print("  Output files: ecg_step1.png  ecg_step2.png")
    print("                ecg_step3.png  ecg_step4.png  ecg_step5.png")
    print("═"*60)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("ENCM 515 — Real-Time Wearable ECG Signal Processing Accelerator")
    print("MIT-BIH Arrhythmia Database | Combined simulation script\n")

    signal, filtered, coeffs, t, ann, avg_py, avg_np, speedup_s1 = \
        step1_load_and_filter()

    cpi_ideal, cpi_stalled, stall_frac, II_ideal, II_stalled = \
        step2_pipeline_model(avg_py, avg_np)

    unroll_results = step3_loop_unrolling(signal, coeffs)

    precision_results = step4_precision_analysis(signal, coeffs)

    sensitivity, precision_score, f1, max_speedup = \
        step5_rpeaks_and_amdahl(filtered, ann, speedup_s1, stall_frac)

    print_summary(avg_py, avg_np, speedup_s1,
                  cpi_ideal, cpi_stalled, stall_frac,
                  unroll_results, precision_results,
                  sensitivity, precision_score, f1, max_speedup)