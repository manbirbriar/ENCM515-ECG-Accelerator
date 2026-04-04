import wfdb
import numpy as np
import matplotlib.pyplot as plt

from clock_unit import ClockUnit
from data_uploader import DataUploader
from fifo import InputFIFO
from shared_mac_unit import SharedMACUnit
from squaring_unit import SquaringUnit
from mwi_unit import MWIUnit
from threshold_unit import ThresholdUnit
from data_recorder import DataRecorder
from config import (
  FIXED_POINT_SCALE, SAMPLE_RATE,
  CYCLES_PER_SAMPLE, FIFO_SIZE, DATA_RECORDER_CAPACITY,
  MAX_SAMPLES
)

# --- Data loading ---

def get_patient_bpm(patient_number, data_dir="ecg_data"):
  record_path = f"{data_dir}/patient_{patient_number}/{patient_number}"
  record = wfdb.rdrecord(record_path)
  annotation = wfdb.rdann(record_path, "atr")
  r_peaks = annotation.sample[np.isin(annotation.symbol, ["N","L","R","B","A","a","J","S","V","F","e","j"])]
  rr_intervals = np.diff(r_peaks) / record.fs
  return round(60 / np.median(rr_intervals))

def load_data(record_path: str):
  record = wfdb.rdrecord(record_path)
  lead0 = record.p_signal[:, 0].astype(np.float32)[:MAX_SAMPLES]
  lead1 = record.p_signal[:, 1].astype(np.float32)[:MAX_SAMPLES]
  return lead0, lead1

# --- Pipeline builder ---

def build_shared_pipeline(lead0_samples, lead1_samples, is_fixed: bool):
  """
  Builds a two-lead pipeline with a single shared MAC unit.
  Lead 0 and Lead 1 share one MAC but have separate downstream units.

  Architecture:
    FIFO0 ─┐
           ├─► SharedMAC ─► Squaring0 ─► MWI0 ─► Threshold0
    FIFO1 ─┘            └─► Squaring1 ─► MWI1 ─► Threshold1
  """
  dtype = int if is_fixed else float

  # FIFOs
  fifo0 = InputFIFO("fifo0", FIFO_SIZE)
  fifo1 = InputFIFO("fifo1", FIFO_SIZE)

  # Uploaders (ADC models)
  # phase_offset=1 on uploader1 models realistic ADC clock skew between leads —
  # in real hardware two ADC channels are never perfectly synchronized.
  # This causes the two FIFOs to receive samples 1 cycle apart, which triggers
  # the structural hazard in the shared MAC when one FIFO has data and the other doesn't.
  uploader0 = DataUploader("uploader0", lead0_samples, CYCLES_PER_SAMPLE, fifo0, phase_offset=0)
  uploader1 = DataUploader("uploader1", lead1_samples, CYCLES_PER_SAMPLE, fifo1, phase_offset=1)

  # Shared MAC
  mac = SharedMACUnit("shared_mac", fifo0, fifo1, is_fixed_point=is_fixed)

  # Per-lead downstream units
  squaring0 = SquaringUnit("squaring0", is_fixed_point=is_fixed)
  squaring1 = SquaringUnit("squaring1", is_fixed_point=is_fixed)

  mwi0 = MWIUnit("mwi0", is_fixed_point=is_fixed)
  mwi1 = MWIUnit("mwi1", is_fixed_point=is_fixed)

  threshold0 = ThresholdUnit("threshold0", SAMPLE_RATE, is_fixed_point=is_fixed)
  threshold1 = ThresholdUnit("threshold1", SAMPLE_RATE, is_fixed_point=is_fixed)

  # Recorders — lead 0
  raw0_rec   = DataRecorder("lead0_raw",       DATA_RECORDER_CAPACITY)
  lp0_rec    = DataRecorder("lead0_lp",        DATA_RECORDER_CAPACITY)
  hp0_rec    = DataRecorder("lead0_hp",        DATA_RECORDER_CAPACITY)
  dv0_rec    = DataRecorder("lead0_dv",        DATA_RECORDER_CAPACITY)
  sq0_rec    = DataRecorder("lead0_squaring",  DATA_RECORDER_CAPACITY)
  mwi0_rec   = DataRecorder("lead0_mwi",       DATA_RECORDER_CAPACITY)
  thr0_rec   = DataRecorder("lead0_threshold", DATA_RECORDER_CAPACITY)

  # Recorders — lead 1
  raw1_rec   = DataRecorder("lead1_raw",       DATA_RECORDER_CAPACITY)
  lp1_rec    = DataRecorder("lead1_lp",        DATA_RECORDER_CAPACITY)
  hp1_rec    = DataRecorder("lead1_hp",        DATA_RECORDER_CAPACITY)
  dv1_rec    = DataRecorder("lead1_dv",        DATA_RECORDER_CAPACITY)
  sq1_rec    = DataRecorder("lead1_squaring",  DATA_RECORDER_CAPACITY)
  mwi1_rec   = DataRecorder("lead1_mwi",       DATA_RECORDER_CAPACITY)
  thr1_rec   = DataRecorder("lead1_threshold", DATA_RECORDER_CAPACITY)

  # Attach recorders to shared MAC
  mac.attach_recorders_l0(raw0_rec, lp0_rec, hp0_rec, dv0_rec)
  mac.attach_recorders_l1(raw1_rec, lp1_rec, hp1_rec, dv1_rec)

  # Attach recorders to downstream units
  squaring0.attach_recorder(sq0_rec)
  squaring1.attach_recorder(sq1_rec)
  mwi0.attach_recorder(mwi0_rec)
  mwi1.attach_recorder(mwi1_rec)
  threshold0.attach_recorder(thr0_rec)
  threshold1.attach_recorder(thr1_rec)

  # Wire lead 0 pipeline: MAC → squaring0 → mwi0 → threshold0
  mac.connect(squaring0).connect(mwi0).connect(threshold0)

  # Wire lead 1 pipeline: MAC → squaring1 → mwi1 → threshold1
  mac.connect_l1(squaring1).connect(mwi1).connect(threshold1)

  all_units = [uploader0, uploader1, fifo0, fifo1, mac,
               squaring0, mwi0, threshold0,
               squaring1, mwi1, threshold1]

  recorders0 = {
    "raw": raw0_rec, "lp": lp0_rec, "hp": hp0_rec, "dv": dv0_rec,
    "squaring": sq0_rec, "mwi": mwi0_rec, "threshold": thr0_rec,
  }
  recorders1 = {
    "raw": raw1_rec, "lp": lp1_rec, "hp": hp1_rec, "dv": dv1_rec,
    "squaring": sq1_rec, "mwi": mwi1_rec, "threshold": thr1_rec,
  }

  return all_units, fifo0, fifo1, mac, squaring0, squaring1, mwi0, mwi1, threshold0, threshold1, recorders0, recorders1

# --- Simulation runner ---

def run_simulation(lead0_samples, lead1_samples, is_fixed: bool):
  mode = "Fixed" if is_fixed else "Float"
  total_samples = len(lead0_samples)

  (all_units, fifo0, fifo1, mac,
   squaring0, squaring1, mwi0, mwi1,
   threshold0, threshold1,
   recorders0, recorders1) = build_shared_pipeline(lead0_samples, lead1_samples, is_fixed)

  clock = ClockUnit()
  clock.subscribe_many(all_units)

  uploader0, uploader1 = all_units[0], all_units[1]

  log_interval = CYCLES_PER_SAMPLE * 1000
  while True:
    clock.tick()

    if clock.cycle % log_interval == 0:
      samples_sent = uploader0.sample_index
      pct = samples_sent / total_samples * 100
      print(
        f"  [{mode}] cycle={clock.cycle:>8} | "
        f"samples={samples_sent}/{total_samples} ({pct:5.1f}%) | "
        f"mac_state={mac.state} | "
        f"fifo_stalls={mac.fifo_stall_cycles} | "
        f"output_stalls={mac.output_stall_cycles} | "
        f"peaks0={len(threshold0.peaks)} peaks1={len(threshold1.peaks)}"
      )

    uploaders_done = not uploader0.active and not uploader1.active
    fifos_empty = fifo0.is_empty() and fifo1.is_empty()
    pipeline_drained = all(
      not u.busy and u.output_data is None and u.input_data is None
      for u in all_units
      if hasattr(u, 'busy')
    ) and mac.output_data_l0 is None and mac.output_data_l1 is None

    if uploaders_done and fifos_empty and pipeline_drained:
      break

  print(
    f"  [{mode}] Done — total_cycles={clock.cycle} | "
    f"peaks0={len(threshold0.peaks)} peaks1={len(threshold1.peaks)} | "
    f"fifo_stalls={mac.fifo_stall_cycles} | "
    f"output_stalls={mac.output_stall_cycles} | "
    f"dropped0={fifo0.dropped_samples} dropped1={fifo1.dropped_samples}"
  )

  bpm0 = threshold0.get_bpm()
  bpm1 = threshold1.get_bpm()
  avg_bpm = (bpm0 + bpm1) / 2 if bpm0 > 0 and bpm1 > 0 else max(bpm0, bpm1)

  return (recorders0, recorders1, threshold0, threshold1,
          avg_bpm, clock, all_units, mac, fifo0, fifo1)

# --- Reporting ---

def print_performance(all_units, mac, label: str):
  print(f"\n{label} Performance:")
  print(f"  {'Unit':<30} {'Busy':>8} {'Idle':>8} {'Stalled':>8} {'Util':>8}")
  print(f"  {'-'*64}")
  for u in all_units:
    if hasattr(u, 'busy_cycles'):
      print(f"  {u.name:<30} {u.busy_cycles:>8} {u.idle_cycles:>8} {u.stalled_cycles:>8} {u.utilization:>8.1%}")

  print(f"\n  Shared MAC stall breakdown:")
  print(f"    FIFO stalls (structural hazard): {mac.fifo_stall_cycles}")
  print(f"    Output stalls (pipeline hazard): {mac.output_stall_cycles}")

def print_fifo_stats(fifo0, fifo1, label: str):
  print(f"\n{label} FIFO Stats:")
  print(f"  FIFO0 max depth: {fifo0.max_depth}/{fifo0.capacity} | dropped: {fifo0.dropped_samples}")
  print(f"  FIFO1 max depth: {fifo1.max_depth}/{fifo1.capacity} | dropped: {fifo1.dropped_samples}")

# --- RMSE ---

def compute_rmse(float_recorders, fixed_recorders):
  stages = ["raw", "lp", "hp", "dv", "squaring", "mwi"]
  print("\nFloat vs Fixed RMSE (Lead 0):")
  for stage in stages:
    f_sig = np.array(float_recorders[stage].get_signal())
    x_sig = np.array(fixed_recorders[stage].get_signal()) / FIXED_POINT_SCALE
    min_len = min(len(f_sig), len(x_sig))
    if min_len == 0:
      continue
    rmse = np.sqrt(np.mean((f_sig[:min_len] - x_sig[:min_len])**2))
    print(f"  {stage:<12} RMSE: {rmse:.8f}")

# --- Plotting ---

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

# --- Entry point ---

if __name__ == "__main__":
  patient_number = 116
  record_path = f"ecg_data/patient_{patient_number}/{patient_number}"

  float_lead0, float_lead1 = load_data(record_path)
  fixed_lead0 = (float_lead0 * FIXED_POINT_SCALE).astype(np.int32)
  fixed_lead1 = (float_lead1 * FIXED_POINT_SCALE).astype(np.int32)

  actual_bpm = get_patient_bpm(patient_number)

  print("Running Float Simulation (Shared MAC)...")
  (f_rec0, f_rec1, f_thresh0, f_thresh1, float_bpm,
   f_clock, f_units, f_mac, f_fifo0, f_fifo1) = run_simulation(
    float_lead0, float_lead1, is_fixed=False)

  print("Running Fixed Simulation (Shared MAC)...")
  (x_rec0, x_rec1, x_thresh0, x_thresh1, fixed_bpm,
   x_clock, x_units, x_mac, x_fifo0, x_fifo1) = run_simulation(
    fixed_lead0, fixed_lead1, is_fixed=True)

  # Results
  print(f"\nPatient #{patient_number} (Shared MAC — both leads):")
  print(f"  Actual BPM (Database): {actual_bpm}")
  print(f"  Float Mode BPM:        {float_bpm:.1f}")
  print(f"  Fixed Mode BPM:        {fixed_bpm:.1f}")
  print(f"\n  Float total cycles: {f_clock.cycle}")
  print(f"  Fixed total cycles: {x_clock.cycle}")

  print_performance(f_units, f_mac, "Float Shared MAC")
  print_performance(x_units, x_mac, "Fixed Shared MAC")

  print_fifo_stats(f_fifo0, f_fifo1, "Float")
  print_fifo_stats(x_fifo0, x_fifo1, "Fixed")

  compute_rmse(f_rec0, x_rec0)

  plot_recorders(f_rec0, f"Patient {patient_number} - Float Lead 0 (Shared MAC)")
  plot_recorders(x_rec0, f"Patient {patient_number} - Fixed Lead 0 (Shared MAC)")