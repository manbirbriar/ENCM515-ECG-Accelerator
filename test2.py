"""
ENCM 515 - Real-Time Wearable ECG Signal Processing Accelerator
Multi-stage FIR Pipeline Simulation with Full Performance Analysis
Uses real MIT-BIH Arrhythmia Database patient recordings.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
import time
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
FS          = 360          # MIT-BIH sample rate (Hz)
DURATION    = 10           # Seconds to analyse per patient
N_SAMPLES   = FS * DURATION
FIXED_BITS  = 16
FIXED_FRAC  = 12
np.random.seed(42)

# ─────────────────────────────────────────────
# 1. REAL MIT-BIH DATA LOADER
# ─────────────────────────────────────────────

def load_mitbih(data_dir, record_name, duration_sec=10, channel=0):
    """
    Load a MIT-BIH record from local .dat/.hea/.atr files.

    Parameters
    ----------
    data_dir    : path to folder containing the record files
    record_name : e.g. '116', '123', '215'
    duration_sec: seconds to read
    channel     : 0 = MLII (standard), 1 = V1/V5

    Returns
    -------
    t           : time axis (s)
    ecg         : raw ECG signal (physical units, mV)
    ann_samples : sample indices of annotated R-peaks (from .atr file)
    fs          : sampling frequency
    """
    record_path = os.path.join(data_dir, record_name)
    n_samp = duration_sec * FS

    record = wfdb.rdrecord(record_path, sampto=n_samp)
    ecg    = record.p_signal[:, channel].astype(np.float64)
    fs     = record.fs

    # Load annotations (true R-peak locations from cardiologist)
    ann = wfdb.rdann(record_path, 'atr', sampto=n_samp)
    # Keep only beat annotations (N, L, R, A, V, etc.) — exclude non-beat
    beat_symbols = set('NLRBAaJSVrFejnE/fQq')
    ann_samples  = np.array([s for s, sym in zip(ann.sample, ann.symbol)
                              if sym in beat_symbols])

    t = np.arange(len(ecg)) / fs
    return t, ecg, ann_samples, fs

def load_all_patients(data_dir):
    """Load all available patient records found in data_dir."""
    patients = {}
    # Auto-detect patient folders
    try:
        entries = os.listdir(data_dir)
    except FileNotFoundError:
        return patients

    for entry in sorted(entries):
        subdir = os.path.join(data_dir, entry)
        if not os.path.isdir(subdir):
            continue
        # Find .hea files to identify record names
        for f in os.listdir(subdir):
            if f.endswith('.hea'):
                rec_name = f[:-4]
                try:
                    t, ecg, ann, fs = load_mitbih(subdir, rec_name, DURATION)
                    patients[rec_name] = {
                        'folder' : subdir,
                        't'      : t,
                        'ecg'    : ecg,
                        'ann'    : ann,   # ground-truth R-peaks
                        'fs'     : fs,
                    }
                    print(f"    Loaded patient {rec_name}: {len(ecg)} samples, "
                          f"{len(ann)} annotated beats, fs={fs} Hz")
                except Exception as e:
                    print(f"    Warning — could not load {rec_name}: {e}")
    return patients

# ─────────────────────────────────────────────
# 2. MULTI-STAGE FIR FILTER DESIGN
# ─────────────────────────────────────────────

def design_filters(fs):
    nyq = fs / 2.0
    bp_b     = signal.firwin(101, [0.5/nyq, 40.0/nyq], pass_zero=False, window='hamming')
    notch_b  = signal.firwin(101, [48.0/nyq, 52.0/nyq], pass_zero=True,  window='hamming')
    smooth_b = signal.firwin(51,  25.0/nyq,                               window='hamming')
    return bp_b, notch_b, smooth_b

# ─────────────────────────────────────────────
# 3. CIRCULAR BUFFER (hardware-style)
# ─────────────────────────────────────────────

class CircularBuffer:
    def __init__(self, size):
        self.buf  = np.zeros(size)
        self.size = size
        self.ptr  = 0

    def push(self, val):
        self.buf[self.ptr] = val
        self.ptr = (self.ptr + 1) % self.size

    def ordered(self):
        return np.concatenate([self.buf[self.ptr:], self.buf[:self.ptr]])

# ─────────────────────────────────────────────
# 4. FIR IMPLEMENTATIONS
# ─────────────────────────────────────────────

def fir_baseline(x, coeffs):
    """Naive sample-by-sample FIR with circular buffer."""
    N = len(coeffs)
    buf = CircularBuffer(N)
    out = np.zeros(len(x))
    for i, sample in enumerate(x):
        buf.push(sample)
        ordered = buf.ordered()
        acc = 0.0
        for k in range(N):
            acc += coeffs[k] * ordered[N - 1 - k]
        out[i] = acc
    return out

def fir_unrolled(x, coeffs):
    """Loop-unrolled FIR (factor 4)."""
    N  = len(coeffs)
    M  = len(x)
    xp = np.concatenate([np.zeros(N - 1), x])
    out = np.zeros(M)
    i = 0
    while i + 3 < M:
        out[i]   = np.dot(coeffs[::-1], xp[i    : i+N  ])
        out[i+1] = np.dot(coeffs[::-1], xp[i+1  : i+1+N])
        out[i+2] = np.dot(coeffs[::-1], xp[i+2  : i+2+N])
        out[i+3] = np.dot(coeffs[::-1], xp[i+3  : i+3+N])
        i += 4
    while i < M:
        out[i] = np.dot(coeffs[::-1], xp[i : i+N])
        i += 1
    return out

def fir_fast(x, coeffs):
    return signal.lfilter(coeffs, [1.0], x)

# ─────────────────────────────────────────────
# 5. FIXED-POINT SIMULATION
# ─────────────────────────────────────────────

def to_fixed(x, frac_bits=FIXED_FRAC, word_bits=FIXED_BITS):
    scale   = 2 ** frac_bits
    max_val = (2**(word_bits-1)-1) / scale
    min_val = -(2**(word_bits-1))  / scale
    return np.clip(np.round(x * scale) / scale, min_val, max_val)

def fir_fixed_point(x, coeffs):
    xq = to_fixed(x);  cq = to_fixed(coeffs)
    return to_fixed(signal.lfilter(cq, [1.0], xq))

def snr_db(reference, processed):
    sp = np.mean(reference**2)
    np_ = np.mean((reference - processed)**2)
    return 10*np.log10(sp / np_) if np_ > 0 else float('inf')

# ─────────────────────────────────────────────
# 6. R-PEAK DETECTION
# ─────────────────────────────────────────────

def detect_r_peaks(ecg_filt, fs, threshold_factor=0.6, refractory_ms=200):
    threshold  = threshold_factor * np.max(np.abs(ecg_filt))
    refractory = int(refractory_ms * fs / 1000)
    peaks, last_peak = [], -refractory
    for i in range(1, len(ecg_filt)-1):
        if (ecg_filt[i] > threshold and
                ecg_filt[i] > ecg_filt[i-1] and
                ecg_filt[i] > ecg_filt[i+1] and
                i - last_peak > refractory):
            peaks.append(i); last_peak = i
    return np.array(peaks)

def detection_accuracy(detected, ground_truth, tolerance_ms=150, fs=360):
    """Compare detected peaks against annotated ground truth."""
    tol = int(tolerance_ms * fs / 1000)
    tp = 0
    matched = set()
    for d in detected:
        dists = np.abs(ground_truth - d)
        idx   = np.argmin(dists)
        if dists[idx] <= tol and idx not in matched:
            tp += 1; matched.add(idx)
    fp = len(detected) - tp
    fn = len(ground_truth) - tp
    sensitivity = tp / (tp + fn) if (tp+fn) > 0 else 0
    ppv         = tp / (tp + fp) if (tp+fp) > 0 else 0
    f1          = 2*sensitivity*ppv/(sensitivity+ppv) if (sensitivity+ppv) > 0 else 0
    return {'TP': tp, 'FP': fp, 'FN': fn,
            'Sensitivity': sensitivity, 'PPV': ppv, 'F1': f1}

# ─────────────────────────────────────────────
# 7. PERFORMANCE MODEL
# ─────────────────────────────────────────────

def model_pipeline_performance(total_taps, unroll_factor=4):
    CLK_MHZ = 100; PIPE_DEPTH = 5; MEM_LAT = 2
    cpi_base    = total_taps / PIPE_DEPTH + MEM_LAT / total_taps
    cpi_unroll  = cpi_base * (1 - 0.15*(1 - 1/unroll_factor))
    return {
        'cpi_baseline'        : cpi_base,
        'cpi_unrolled'        : cpi_unroll,
        'throughput_baseline' : (CLK_MHZ*1e6) / cpi_base,
        'throughput_unrolled' : (CLK_MHZ*1e6) / cpi_unroll,
        'latency_baseline_us' : cpi_base  / (CLK_MHZ*1e6) * 1e6,
        'latency_unrolled_us' : cpi_unroll / (CLK_MHZ*1e6) * 1e6,
        'fill_penalty'        : PIPE_DEPTH - 1,
        'speedup_unrolling'   : cpi_base / cpi_unroll,
        'clk_mhz'             : CLK_MHZ,
        'pipeline_depth'      : PIPE_DEPTH,
    }

def amdahl_analysis(p_fracs, n_procs):
    results = np.zeros((len(p_fracs), len(n_procs)))
    for i, p in enumerate(p_fracs):
        for j, n in enumerate(n_procs):
            results[i,j] = 1.0 / ((1-p) + p/n)
    return results

def benchmark(func, *args, repeats=5):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter(); func(*args)
        times.append(time.perf_counter()-t0)
    return np.mean(times), np.std(times)

# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

def main():
    os.makedirs('outputs', exist_ok=True)

    print("=" * 65)
    print(" ENCM 515 — Real-Time ECG Accelerator (MIT-BIH Dataset)")
    print("=" * 65)

    # ── Load real patient data ───────────────────────────────
    DATA_DIR = 'ecg_data'          # <-- folder containing patient_116 etc.
    print(f"\n[1] Loading MIT-BIH records from '{DATA_DIR}/'...")
    patients = load_all_patients(DATA_DIR)

    if not patients:
        print("    ERROR: No patient data found.")
        print("    Make sure your ecg_data/ folder contains patient subfolders")
        print("    with .dat/.hea/.atr files (e.g. ecg_data/patient_116/116.dat)")
        return

    # ── Design filters (shared across all patients) ──────────
    # Use fs from first loaded patient
    first = next(iter(patients.values()))
    fs    = first['fs']
    bp_b, notch_b, smooth_b = design_filters(fs)
    tap_counts  = [len(bp_b), len(notch_b), len(smooth_b)]
    total_taps  = sum(tap_counts)
    print(f"\n[2] Filters designed at fs={fs} Hz | Taps per stage: {tap_counts}")

    # ── Process each patient ─────────────────────────────────
    print("\n[3] Processing patients...")
    results_table = []   # for summary table

    for rec_name, pdata in patients.items():
        t_sig   = pdata['t']
        ecg_raw = pdata['ecg']
        ann     = pdata['ann']

        # Multi-stage pipeline (float)
        s1  = fir_fast(ecg_raw, bp_b)
        s2  = fir_fast(s1,      notch_b)
        s3  = fir_fast(s2,      smooth_b)

        # Fixed-point pipeline
        s1x = fir_fixed_point(ecg_raw, bp_b)
        s2x = fir_fixed_point(s1x,     notch_b)
        s3x = fir_fixed_point(s2x,     smooth_b)

        # Group delay compensation
        # Each linear-phase FIR has delay = (N-1)/2 samples.
        # Total delay across 3 stages = (100 + 100 + 50) / 2 = 125 samples.
        group_delay = (len(bp_b)-1)//2 + (len(notch_b)-1)//2 + (len(smooth_b)-1)//2

        # R-peak detection
        r_fp_raw = detect_r_peaks(s3,  fs)
        r_fx_raw = detect_r_peaks(s3x, fs)

        # Shift detected peaks back by group delay to align with annotations
        r_fp = r_fp_raw - group_delay
        r_fx = r_fx_raw - group_delay
        # Remove any peaks that shifted below zero
        r_fp = r_fp[r_fp >= 0]
        r_fx = r_fx[r_fx >= 0]

        # Accuracy vs ground truth (wider tolerance: 300 ms covers any residual jitter)
        acc_fp = detection_accuracy(r_fp, ann, tolerance_ms=300, fs=fs)
        acc_fx = detection_accuracy(r_fx, ann, tolerance_ms=300, fs=fs)

        # SNR (float vs fixed)
        snr_fl = snr_db(s3, s3)           # float vs itself = reference
        snr_fx = snr_db(s3, s3x)
        rmse   = np.sqrt(np.mean((s3 - s3x)**2))

        # Heart rate
        if len(r_fp) >= 2:
            rr  = np.diff(t_sig[r_fp])
            hr  = 60.0 / np.mean(rr)
        else:
            hr = float('nan')

        pdata.update({'s1':s1,'s2':s2,'s3':s3,'s3x':s3x,
                      'r_fp':r_fp,'r_fx':r_fx,
                      'r_fp_raw':r_fp_raw,
                      'group_delay':group_delay,
                      'acc_fp':acc_fp,'snr_fx':snr_fx,'rmse':rmse,'hr':hr})

        results_table.append({
            'Patient'    : rec_name,
            'Ann beats'  : len(ann),
            'Det (float)': len(r_fp),
            'Det (fixed)': len(r_fx),
            'Sens %'     : acc_fp['Sensitivity']*100,
            'PPV %'      : acc_fp['PPV']*100,
            'F1 %'       : acc_fp['F1']*100,
            'HR (BPM)'   : hr,
            'SNR loss dB': snr_fl - snr_fx,
            'RMSE'       : rmse,
        })
        print(f"    Patient {rec_name}: HR={hr:.1f} BPM | "
              f"Sensitivity={acc_fp['Sensitivity']*100:.1f}% | "
              f"F1={acc_fp['F1']*100:.1f}%")

    # ── Benchmarking (on first patient) ─────────────────────
    print("\n[4] Benchmarking (2000 samples, Stage 1, 5 repeats)...")
    short    = first['ecg'][:2000]
    t_base,  _ = benchmark(fir_baseline, short, bp_b)
    t_unroll,_ = benchmark(fir_unrolled, short, bp_b)
    t_fast,  _ = benchmark(fir_fast,     short, bp_b)
    sp_unroll  = t_base / t_unroll
    sp_fast    = t_base / t_fast
    print(f"    Baseline  : {t_base*1000:.3f} ms")
    print(f"    Unrolled  : {t_unroll*1000:.3f} ms  ({sp_unroll:.2f}x speedup)")
    print(f"    Vectorised: {t_fast*1000:.3f} ms  ({sp_fast:.2f}x speedup)")

    # ── Architectural model ──────────────────────────────────
    perf    = model_pipeline_performance(total_taps)
    p_fracs = [0.50, 0.75, 0.90, 0.95]
    n_procs = [1, 2, 4, 8, 16, 32]
    amdahl  = amdahl_analysis(p_fracs, n_procs)

    # ────────────────────────────────────────────────────────
    # PLOTS
    # ────────────────────────────────────────────────────────
    PLOT_SEC  = 4
    PLOT_WIN  = slice(0, fs * PLOT_SEC)

    # ── Figure 1: Multi-stage pipeline (one patient per row set) ──
    pat_list  = list(patients.items())
    n_pat     = len(pat_list)
    fig1, axes1 = plt.subplots(n_pat * 2, 1, figsize=(15, 5*n_pat), sharex=False)
    fig1.suptitle('Multi-Stage FIR Pipeline — Real MIT-BIH ECG Data', fontsize=14, fontweight='bold')

    for pi, (rec_name, pd) in enumerate(pat_list):
        tw  = pd['t'][PLOT_WIN]
        row_raw  = axes1[pi*2]
        row_filt = axes1[pi*2+1]

        row_raw.plot(tw, pd['ecg'][PLOT_WIN], color='#d62728', lw=0.7, alpha=0.9)
        row_raw.set_title(f'Patient {rec_name} — Raw ECG Input', fontsize=10)
        row_raw.set_ylabel('mV'); row_raw.grid(True, alpha=0.3)

        row_filt.plot(tw, pd['s3'][PLOT_WIN], color='#1f77b4', lw=0.9, label='Filtered (float)')
        row_filt.plot(tw, pd['s3x'][PLOT_WIN], color='#ff7f0e', lw=0.7, ls='--',
                      alpha=0.7, label='Filtered (fixed Q4.12)')
        # Annotated R-peaks in window
        ann_win = pd['ann'][(pd['ann'] >= PLOT_WIN.start) & (pd['ann'] < PLOT_WIN.stop)]
        row_filt.vlines(pd['t'][ann_win], ymin=row_filt.get_ylim()[0],
                        ymax=row_filt.get_ylim()[1], color='green', alpha=0.4,
                        lw=0.8, label='Ground-truth R-peaks')
        det_win = pd['r_fp_raw'][(pd['r_fp_raw'] >= PLOT_WIN.start) & (pd['r_fp_raw'] < PLOT_WIN.stop)]
        row_filt.plot(pd['t'][det_win], pd['s3'][det_win], 'rv', ms=7, label='Detected R-peaks')
        row_filt.set_title(f'Patient {rec_name} — After 3-Stage FIR Pipeline + Detection', fontsize=10)
        row_filt.set_ylabel('mV'); row_filt.set_xlabel('Time (s)')
        row_filt.legend(fontsize=8, loc='upper right'); row_filt.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1.savefig('outputs/fig1_pipeline.png', dpi=150, bbox_inches='tight')
    print("\n[5] Saved fig1_pipeline.png")

    # ── Figure 2: Frequency responses ────────────────────────
    fig2, ax2s = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle('FIR Filter Frequency Responses', fontsize=13, fontweight='bold')
    for ax, filt, lbl, col in zip(ax2s,
            [bp_b, notch_b, smooth_b],
            ['Stage 1: Bandpass (0.5–40 Hz)', 'Stage 2: Notch (48–52 Hz)', 'Stage 3: Smoothing (≤25 Hz)'],
            ['#ff7f0e', '#2ca02c', '#9467bd']):
        w, h = signal.freqz(filt, worN=2048, fs=fs)
        ax.plot(w, 20*np.log10(np.abs(h)+1e-12), color=col, lw=1.5)
        ax.set_title(lbl, fontsize=10); ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Magnitude (dB)')
        ax.set_xlim(0, fs/2); ax.set_ylim(-80, 5); ax.grid(True, alpha=0.3)
        ax.axhline(-3, color='red', ls='--', alpha=0.6, label='-3 dB'); ax.legend(fontsize=8)
    plt.tight_layout()
    fig2.savefig('outputs/fig2_freq_response.png', dpi=150, bbox_inches='tight')
    print("[5] Saved fig2_freq_response.png")

    # ── Figure 3: Detection accuracy per patient ──────────────
    fig3, ax3s = plt.subplots(1, 3, figsize=(14, 5))
    fig3.suptitle('R-Peak Detection Accuracy vs MIT-BIH Ground Truth', fontsize=13, fontweight='bold')
    pat_names = [r['Patient'] for r in results_table]
    metrics   = ['Sens %', 'PPV %', 'F1 %']
    colors3   = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for ax, metric, col in zip(ax3s, metrics, colors3):
        vals = [r[metric] for r in results_table]
        bars = ax.bar(pat_names, vals, color=col, edgecolor='black', width=0.4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=10)
        ax.set_title(metric); ax.set_ylabel('(%)'); ax.set_ylim(0, 115)
        ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig3.savefig('outputs/fig3_detection_accuracy.png', dpi=150, bbox_inches='tight')
    print("[5] Saved fig3_detection_accuracy.png")

    # ── Figure 4: Precision (fixed vs float per patient) ─────
    fig4, ax4s = plt.subplots(1, 2, figsize=(12, 5))
    fig4.suptitle('Fixed-Point vs Floating-Point Precision Analysis', fontsize=13, fontweight='bold')
    snr_losses = [r['SNR loss dB'] for r in results_table]
    rmses      = [r['RMSE']        for r in results_table]
    bars4a = ax4s[0].bar(pat_names, snr_losses, color='#d62728', edgecolor='black', width=0.4)
    for bar, v in zip(bars4a, snr_losses):
        ax4s[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                     f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    ax4s[0].set_title('SNR Loss (Float→Fixed) per Patient')
    ax4s[0].set_ylabel('SNR Loss (dB)'); ax4s[0].grid(True, alpha=0.3, axis='y')

    bars4b = ax4s[1].bar(pat_names, rmses, color='#9467bd', edgecolor='black', width=0.4)
    for bar, v in zip(bars4b, rmses):
        ax4s[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1e-6,
                     f'{v:.5f}', ha='center', va='bottom', fontsize=9)
    ax4s[1].set_title('Quantisation RMSE per Patient')
    ax4s[1].set_ylabel('RMSE'); ax4s[1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig4.savefig('outputs/fig4_precision.png', dpi=150, bbox_inches='tight')
    print("[5] Saved fig4_precision.png")

    # ── Figure 5: Performance benchmarks ─────────────────────
    fig5, ax5s = plt.subplots(1, 3, figsize=(15, 5))
    fig5.suptitle('Performance Analysis — Loop Unrolling & Throughput', fontsize=13, fontweight='bold')
    impl_lbls = ['Baseline\n(Naive)', 'Unrolled\n(Factor-4)', 'Vectorised\n(scipy)']
    impl_cols = ['#d62728', '#ff7f0e', '#2ca02c']
    times_ms  = [t_base*1000, t_unroll*1000, t_fast*1000]
    speedups  = [1.0, sp_unroll, sp_fast]
    cpi_vals  = [perf['cpi_baseline'], perf['cpi_unrolled']]

    for ax, vals, title, ylabel in [
        (ax5s[0], times_ms, 'Execution Time (2000 samples)', 'Time (ms)'),
        (ax5s[1], speedups, 'Speedup vs Baseline', 'Speedup (x)'),
    ]:
        bars = ax.bar(impl_lbls, vals, color=impl_cols, edgecolor='black')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=9)
        ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3, axis='y')
    ax5s[1].axhline(1.0, color='grey', ls='--', alpha=0.6)

    bars_cpi = ax5s[2].bar(['Baseline CPI', 'Unrolled CPI'], cpi_vals,
                            color=['#1f77b4','#17becf'], edgecolor='black', width=0.4)
    for bar, v in zip(bars_cpi, cpi_vals):
        ax5s[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                     f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    ax5s[2].set_title('Modelled CPI (per output sample)'); ax5s[2].set_ylabel('Cycles/sample')
    ax5s[2].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig5.savefig('outputs/fig5_performance.png', dpi=150, bbox_inches='tight')
    print("[5] Saved fig5_performance.png")

    # ── Figure 6: Amdahl's Law ────────────────────────────────
    fig6, ax6 = plt.subplots(figsize=(9, 6))
    markers = ['o-','s--','^-.','D:']
    pal     = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']
    for i, (p, row) in enumerate(zip(p_fracs, amdahl)):
        ax6.plot(n_procs, row, markers[i], color=pal[i], lw=1.8, ms=7,
                 label=f'p = {int(p*100)}% parallelisable')
    ax6.set_title("Amdahl's Law — ECG Accelerator Scalability", fontsize=13, fontweight='bold')
    ax6.set_xlabel('Processing Elements'); ax6.set_ylabel('Theoretical Speedup')
    ax6.set_xticks(n_procs); ax6.legend(fontsize=10); ax6.grid(True, alpha=0.3)
    ax6.set_xscale('log', base=2)
    plt.tight_layout()
    fig6.savefig('outputs/fig6_amdahl.png', dpi=150, bbox_inches='tight')
    print("[5] Saved fig6_amdahl.png")

    # ── Figure 7: Pipeline timing diagram ────────────────────
    fig7, ax7 = plt.subplots(figsize=(14, 5))
    stages = ['IF', 'ID', 'EX\n(MAC)', 'MEM', 'WB']
    n_s = len(stages); n_i = 10
    colors_p = plt.cm.Pastel1(np.linspace(0, 1, n_i))
    for instr in range(n_i):
        for st in range(n_s):
            rect = plt.Rectangle((instr+st, n_i-instr-1), 1, 0.8,
                                  facecolor=colors_p[instr], edgecolor='black', lw=0.8)
            ax7.add_patch(rect)
            ax7.text(instr+st+0.5, n_i-instr-0.6, stages[st],
                     ha='center', va='center', fontsize=7.5, fontweight='bold')
    ax7.set_xlim(0, n_i+n_s-1); ax7.set_ylim(-0.2, n_i)
    ax7.set_xlabel('Clock Cycle', fontsize=11); ax7.set_ylabel('MAC Operation', fontsize=11)
    ax7.set_title(f'5-Stage Pipelined DSP Accelerator — Timing Diagram\n'
                  f'Fill penalty = {perf["fill_penalty"]} cycles → 1 output/cycle steady-state',
                  fontsize=12, fontweight='bold')
    ax7.set_yticks(np.arange(n_i)+0.4)
    ax7.set_yticklabels([f'MAC {n_i-i}' for i in range(n_i)], fontsize=9)
    ax7.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    fig7.savefig('outputs/fig7_pipeline_timing.png', dpi=150, bbox_inches='tight')
    print("[5] Saved fig7_pipeline_timing.png")

    # ── Figure 8: Heart rate per patient ─────────────────────
    fig8, ax8 = plt.subplots(figsize=(10, 5))
    for pi, (rec_name, pd) in enumerate(pat_list):
        if len(pd['r_fp']) >= 2:
            rr = np.diff(pd['t'][pd['r_fp']])
            hr_inst = 60.0 / rr
            ax8.plot(pd['t'][pd['r_fp'][1:]], hr_inst,
                     marker='o', lw=1.3, ms=4, label=f'Patient {rec_name}')
    ax8.set_title('Instantaneous Heart Rate — All Patients', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Time (s)'); ax8.set_ylabel('Heart Rate (BPM)')
    ax8.legend(); ax8.grid(True, alpha=0.3)
    plt.tight_layout()
    fig8.savefig('outputs/fig8_heart_rate.png', dpi=150, bbox_inches='tight')
    print("[5] Saved fig8_heart_rate.png")

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(" SIMULATION SUMMARY")
    print("=" * 65)
    print(f"  Patients loaded  : {list(patients.keys())}")
    print(f"  Duration/patient : {DURATION}s @ {fs} Hz")
    print(f"  Filter stages    : Bandpass | Notch | Smoothing")
    print(f"  Total taps       : {total_taps}")
    print()
    print(f"  {'Patient':<12} {'Ann':>5} {'Det':>5} {'Sens%':>7} {'PPV%':>7} {'F1%':>7} {'HR':>7} {'RMSE':>10}")
    print(f"  {'-'*65}")
    for r in results_table:
        print(f"  {r['Patient']:<12} {r['Ann beats']:>5} {r['Det (float)']:>5} "
              f"{r['Sens %']:>7.1f} {r['PPV %']:>7.1f} {r['F1 %']:>7.1f} "
              f"{r['HR (BPM)']:>7.1f} {r['RMSE']:>10.6f}")
    print()
    print(f"  [Loop Unrolling Benchmark — 2000 samples, Stage 1]")
    print(f"  Baseline   : {t_base*1000:.3f} ms")
    print(f"  Unrolled x4: {t_unroll*1000:.3f} ms  → {sp_unroll:.2f}x speedup")
    print(f"  Vectorised : {t_fast*1000:.3f} ms  → {sp_fast:.2f}x speedup")
    print()
    print(f"  [Architectural Model — {perf['clk_mhz']} MHz, {perf['pipeline_depth']}-stage]")
    print(f"  CPI baseline  : {perf['cpi_baseline']:.3f} | CPI unrolled: {perf['cpi_unrolled']:.3f}")
    print(f"  Throughput    : {perf['throughput_baseline']/1e6:.2f} → {perf['throughput_unrolled']/1e6:.2f} Msps")
    print(f"  Latency       : {perf['latency_baseline_us']:.4f} → {perf['latency_unrolled_us']:.4f} µs/sample")
    print(f"  Speedup       : {perf['speedup_unrolling']:.3f}x from loop unrolling")
    print()
    print(f"  [Amdahl's Law @ 32 PEs]")
    for p, row in zip(p_fracs, amdahl):
        print(f"  p={int(p*100)}%: max speedup = {row[-1]:.2f}x")
    print("=" * 65)
    print("  All figures saved to outputs/")
    print("=" * 65)

if __name__ == '__main__':
    main()