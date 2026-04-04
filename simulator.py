import wfdb
import numpy as np
import matplotlib.pyplot as plt

from clock_unit import ClockUnit
from data_uploader import DataUploader
from fifo import InputFIFO
from mac_unit import MACUnit
from squaring_unit import SquaringUnit
from mwi_unit import MWIUnit
from threshold_unit import ThresholdUnit
from data_recorder import DataRecorder
from config import (
  FIXED_POINT_SCALE, SAMPLE_RATE,
  CYCLES_PER_SAMPLE, FIFO_SIZE, DATA_RECORDER_CAPACITY
)

def get_patient_bpm(patient_number, data_dir="ecg_data"):
  record_path = f"{data_dir}/patient_{patient_number}/{patient_number}"
  record = wfdb.rdrecord(record_path)
  annotation = wfdb.rdann(record_path, "atr")
  r_peaks = annotation.sample[np.isin(annotation.symbol, ["N","L","R","B","A","a","J","S","V","F","e","j"])]
  rr_intervals = np.diff(r_peaks) / record.fs
  return round(60 / np.median(rr_intervals))

def load_data(record_path: str):
  record = wfdb.rdrecord(record_path)
  lead0 = record.p_signal[:, 0].astype(np.float32)
  lead1 = record.p_signal[:, 1].astype(np.float32)
  return lead0, lead1

def build_lane(name: str, samples: np.ndarray, is_fixed: bool, cycles_per_sample: int):
  raw_recorder       = DataRecorder(f"{name}_raw",       DATA_RECORDER_CAPACITY)
  lp_recorder        = DataRecorder(f"{name}_lp",        DATA_RECORDER_CAPACITY)
  hp_recorder        = DataRecorder(f"{name}_hp",        DATA_RECORDER_CAPACITY)
  dv_recorder        = DataRecorder(f"{name}_dv",        DATA_RECORDER_CAPACITY)
  squaring_recorder  = DataRecorder(f"{name}_squaring",  DATA_RECORDER_CAPACITY)
  mwi_recorder       = DataRecorder(f"{name}_mwi",       DATA_RECORDER_CAPACITY)
  threshold_recorder = DataRecorder(f"{name}_threshold", DATA_RECORDER_CAPACITY)

  fifo      = InputFIFO(f"{name}_fifo", FIFO_SIZE)
  mac       = MACUnit(f"{name}_mac", fifo, is_fixed_point=is_fixed)
  uploader  = DataUploader(f"{name}_uploader", samples, cycles_per_sample, fifo, mac)
  squaring  = SquaringUnit(f"{name}_squaring", is_fixed_point=is_fixed)
  mwi       = MWIUnit(f"{name}_mwi", is_fixed_point=is_fixed)
  threshold = ThresholdUnit(f"{name}_threshold", SAMPLE_RATE, is_fixed_point=is_fixed)

  mac.attach_raw_recorder(raw_recorder)
  mac.attach_lp_recorder(lp_recorder)
  mac.attach_hp_recorder(hp_recorder)
  mac.attach_dv_recorder(dv_recorder)
  squaring.attach_recorder(squaring_recorder)
  mwi.attach_recorder(mwi_recorder)
  threshold.attach_recorder(threshold_recorder)

  mac.connect(squaring).connect(mwi).connect(threshold)

  recorders = {
    "raw": raw_recorder, "lp": lp_recorder, "hp": hp_recorder,
    "dv": dv_recorder, "squaring": squaring_recorder,
    "mwi": mwi_recorder, "threshold": threshold_recorder,
  }

  return uploader, fifo, mac, squaring, mwi, threshold, recorders

def run_simulation(lead0_samples, lead1_samples, is_fixed: bool):
  mode = "Fixed" if is_fixed else "Float"
  total_samples = len(lead0_samples)

  up0, fifo0, mac0, sq0, mwi0, thresh0, rec0 = build_lane("lane0", lead0_samples, is_fixed, CYCLES_PER_SAMPLE)
  up1, fifo1, mac1, sq1, mwi1, thresh1, rec1 = build_lane("lane1", lead1_samples, is_fixed, CYCLES_PER_SAMPLE)

  clock = ClockUnit()

  # Progress logging every 10% of samples
  log_every = total_samples // 10
  last_logged = [0]
  def on_progress(cycle):
    samples_sent = up0.sample_index
    if samples_sent - last_logged[0] >= log_every:
      last_logged[0] = samples_sent
      pct = samples_sent / total_samples * 100
      print(
        f"  [{mode}] cycle={cycle:>10} | "
        f"samples={samples_sent}/{total_samples} ({pct:5.1f}%) | "
        f"peaks0={len(thresh0.peaks)} peaks1={len(thresh1.peaks)}"
      )

  # Kick off both uploaders
  up0.start(clock)
  up1.start(clock)

  clock.run(on_progress=on_progress)

  print(
    f"  [{mode}] Done — final cycle={clock.cycle} | "
    f"peaks0={len(thresh0.peaks)} peaks1={len(thresh1.peaks)} | "
    f"dropped0={fifo0.dropped_samples} dropped1={fifo1.dropped_samples}"
  )

  bpm0 = thresh0.get_bpm()
  bpm1 = thresh1.get_bpm()
  avg_bpm = (bpm0 + bpm1) / 2 if bpm0 > 0 and bpm1 > 0 else max(bpm0, bpm1)

  all_units = [up0, mac0, sq0, mwi0, thresh0, up1, mac1, sq1, mwi1, thresh1]
  lane0_units = [up0, mac0, sq0, mwi0, thresh0]
  lane1_units = [up1, mac1, sq1, mwi1, thresh1]

  return rec0, rec1, thresh0, thresh1, avg_bpm, clock, lane0_units, lane1_units, fifo0, fifo1

def print_performance(units: list, label: str):
  print(f"\n{label} Performance:")
  print(f"  {'Unit':<30} {'Busy':>8} {'Idle':>8} {'Stalled':>8} {'Util':>8}")
  print(f"  {'-'*64}")
  for u in units:
    if hasattr(u, 'busy_cycles'):
      print(f"  {u.name:<30} {u.busy_cycles:>8} {u.idle_cycles:>8} {u.stalled_cycles:>8} {u.utilization:>8.1%}")

def plot_recorders(recorders: dict, label: str):
  stages = ["raw", "lp", "hp", "dv", "squaring", "mwi", "threshold"]
  n = len(stages)
  fig, axes = plt.subplots(n, 1, figsize=(12, 3*n), sharex=True)
  for ax, stage in zip(axes, stages):
    signal = recorders[stage].get_signal()
    ax.plot(signal)
    ax.set_title(recorders[stage].name)
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
  axes[-1].set_xlabel("Sample")
  plt.suptitle(label, fontsize=16)
  plt.tight_layout()
  plt.show()

def compute_rmse(float_recorders, fixed_recorders):
  stages = ["raw", "lp", "hp", "dv", "squaring", "mwi"]
  print("\nFloat vs Fixed RMSE:")
  for stage in stages:
    f_sig = np.array(float_recorders[stage].get_signal())
    x_sig = np.array(fixed_recorders[stage].get_signal()) / FIXED_POINT_SCALE
    min_len = min(len(f_sig), len(x_sig))
    if min_len == 0:
      continue
    rmse = np.sqrt(np.mean((f_sig[:min_len] - x_sig[:min_len])**2))
    print(f"  {stage:<12} RMSE: {rmse:.8f}")

if __name__ == "__main__":
  patient_number = 116
  record_path = f"ecg_data/patient_{patient_number}/{patient_number}"

  float_lead0, float_lead1 = load_data(record_path)
  fixed_lead0 = (float_lead0 * FIXED_POINT_SCALE).astype(np.int32)
  fixed_lead1 = (float_lead1 * FIXED_POINT_SCALE).astype(np.int32)

  actual_bpm = get_patient_bpm(patient_number)

  print("Running Floating Point Simulation...")
  f_rec0, f_rec1, f_thresh0, f_thresh1, float_bpm, f_clock, f_units0, f_units1, f_fifo0, f_fifo1 = \
    run_simulation(float_lead0, float_lead1, is_fixed=False)

  print("\nRunning Fixed Point Simulation...")
  x_rec0, x_rec1, x_thresh0, x_thresh1, fixed_bpm, x_clock, x_units0, x_units1, x_fifo0, x_fifo1 = \
    run_simulation(fixed_lead0, fixed_lead1, is_fixed=True)

  print(f"\nPatient #{patient_number}:")
  print(f"  Actual BPM (Database): {actual_bpm}")
  print(f"  Float Mode BPM:        {float_bpm:.1f}")
  print(f"  Fixed Mode BPM:        {fixed_bpm:.1f}")
  print(f"\n  Float final cycle: {f_clock.cycle}")
  print(f"  Fixed final cycle: {x_clock.cycle}")

  print_performance(f_units0, "Float Lane 0")
  print_performance(f_units1, "Float Lane 1")
  print_performance(x_units0, "Fixed Lane 0")
  print_performance(x_units1, "Fixed Lane 1")

  print(f"\nFIFO Stats:")
  print(f"  Float Lane 0: max_depth={f_fifo0.max_depth} dropped={f_fifo0.dropped_samples}")
  print(f"  Float Lane 1: max_depth={f_fifo1.max_depth} dropped={f_fifo1.dropped_samples}")
  print(f"  Fixed Lane 0: max_depth={x_fifo0.max_depth} dropped={x_fifo0.dropped_samples}")
  print(f"  Fixed Lane 1: max_depth={x_fifo1.max_depth} dropped={x_fifo1.dropped_samples}")

  compute_rmse(f_rec0, x_rec0)

  plot_recorders(f_rec0, f"Patient {patient_number} - Float Lane 0")
  plot_recorders(x_rec0, f"Patient {patient_number} - Fixed Lane 0")
