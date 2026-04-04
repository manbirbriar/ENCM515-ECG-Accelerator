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
  CLOCK_FREQUENCY, CYCLES_PER_SAMPLE, FIFO_SIZE, DATA_RECORDER_CAPACITY,
  MAX_SAMPLES
)

# --- Data loading helpers ---

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

def build_lane(name: str, samples: np.ndarray, is_fixed: bool, cycles_per_sample: int, fifo_size: int):
  """
  Builds one complete processing lane for a single ECG lead.
  Returns (units_list, fifo, mac, threshold, recorders_dict)
  """
  # Recorders
  raw_recorder       = DataRecorder(f"{name}_raw",       DATA_RECORDER_CAPACITY)
  lp_recorder        = DataRecorder(f"{name}_lp",        DATA_RECORDER_CAPACITY)
  hp_recorder        = DataRecorder(f"{name}_hp",        DATA_RECORDER_CAPACITY)
  dv_recorder        = DataRecorder(f"{name}_dv",        DATA_RECORDER_CAPACITY)
  squaring_recorder  = DataRecorder(f"{name}_squaring",  DATA_RECORDER_CAPACITY)
  mwi_recorder       = DataRecorder(f"{name}_mwi",       DATA_RECORDER_CAPACITY)
  threshold_recorder = DataRecorder(f"{name}_threshold", DATA_RECORDER_CAPACITY)

  # Hardware units
  fifo      = InputFIFO(f"{name}_fifo", fifo_size)
  uploader  = DataUploader(f"{name}_uploader", samples, cycles_per_sample, fifo)
  mac       = MACUnit(f"{name}_mac", fifo, is_fixed_point=is_fixed)
  squaring  = SquaringUnit(f"{name}_squaring", is_fixed_point=is_fixed)
  mwi       = MWIUnit(f"{name}_mwi", is_fixed_point=is_fixed)
  threshold = ThresholdUnit(f"{name}_threshold", SAMPLE_RATE, is_fixed_point=is_fixed)

  # Attach recorders
  mac.attach_raw_recorder(raw_recorder)
  mac.attach_lp_recorder(lp_recorder)
  mac.attach_hp_recorder(hp_recorder)
  mac.attach_dv_recorder(dv_recorder)
  squaring.attach_recorder(squaring_recorder)
  mwi.attach_recorder(mwi_recorder)
  threshold.attach_recorder(threshold_recorder)

  # Wire pipeline
  mac.connect(squaring).connect(mwi).connect(threshold)

  units = [uploader, fifo, mac, squaring, mwi, threshold]

  recorders = {
    "raw":       raw_recorder,
    "lp":        lp_recorder,
    "hp":        hp_recorder,
    "dv":        dv_recorder,
    "squaring":  squaring_recorder,
    "mwi":       mwi_recorder,
    "threshold": threshold_recorder,
  }

  return units, fifo, mac, threshold, recorders

# --- Simulation runner ---

def run_simulation(
  lead0_samples,
  lead1_samples,
  is_fixed: bool,
  clock_frequency_hz: int | None = None,
  fifo_size: int | None = None,
  verbose: bool = True,
):
  mode = "Fixed" if is_fixed else "Float"
  total_samples = len(lead0_samples)

  effective_clock_hz = CLOCK_FREQUENCY if clock_frequency_hz is None else clock_frequency_hz
  cycles_per_sample = max(1, effective_clock_hz // SAMPLE_RATE)
  effective_fifo_size = FIFO_SIZE if fifo_size is None else fifo_size

  units0, fifo0, mac0, threshold0, recorders0 = build_lane("lane0", lead0_samples, is_fixed, cycles_per_sample, effective_fifo_size)
  units1, fifo1, mac1, threshold1, recorders1 = build_lane("lane1", lead1_samples, is_fixed, cycles_per_sample, effective_fifo_size)

  clock = ClockUnit()
  clock.subscribe_many(units0)
  clock.subscribe_many(units1)

  uploader0, uploader1 = units0[0], units1[0]

  # Run until both uploaders are done and all units have drained
  all_units = units0 + units1
  log_interval = max(1, cycles_per_sample * 1000)  # log every 1000 samples worth of cycles
  while True:
    clock.tick()

    # Log progress every 1000 samples
    if verbose and clock.cycle % log_interval == 0:
      samples_sent = uploader0.sample_index
      pct = samples_sent / total_samples * 100
      print(
        f"  [{mode}] cycle={clock.cycle:>8} | "
        f"samples={samples_sent}/{total_samples} ({pct:5.1f}%) | "
        f"fifo0_depth={fifo0.max_depth} fifo1_depth={fifo1.max_depth} | "
        f"peaks0={len(threshold0.peaks)} peaks1={len(threshold1.peaks)}"
      )

    uploaders_done = not uploader0.active and not uploader1.active
    fifo_empty = fifo0.is_empty() and fifo1.is_empty()
    pipeline_drained = all(
      not u.busy and u.output_data is None and u.input_data is None 
      and getattr(u, 'cycles_remaining', 0) == 0 and getattr(u, 'kernel_cycles_remaining', 0) == 0 #ensure all done to stop 1 sample missing
      for u in all_units
      if hasattr(u, 'busy')
    )
    if uploaders_done and fifo_empty and pipeline_drained:
      break

  if verbose:
    print(
      f"  [{mode}] Done — total cycles={clock.cycle} | "
      f"peaks0={len(threshold0.peaks)} peaks1={len(threshold1.peaks)} | "
      f"dropped0={fifo0.dropped_samples} dropped1={fifo1.dropped_samples}"
    )

  bpm0 = threshold0.get_bpm()
  bpm1 = threshold1.get_bpm()
  avg_bpm = (bpm0 + bpm1) / 2 if bpm0 > 0 and bpm1 > 0 else max(bpm0, bpm1)

  return recorders0, recorders1, threshold0, threshold1, avg_bpm, clock, units0, units1

# --- Reporting ---

def print_performance(units: list, label: str):
  print(f"\n{label} Performance:")
  print(f"  {'Unit':<30} {'Busy':>8} {'Idle':>8} {'Stalled':>8} {'Util':>8}")
  print(f"  {'-'*64}")
  for u in units:
    if hasattr(u, 'busy_cycles'):
      print(f"  {u.name:<30} {u.busy_cycles:>8} {u.idle_cycles:>8} {u.stalled_cycles:>8} {u.utilization:>8.1%}")

def print_fifo_stats(fifo: InputFIFO, label: str):
  print(f"\n{label} FIFO Stats:")
  print(f"  Max depth reached: {fifo.max_depth}/{fifo.capacity}")
  print(f"  Dropped samples:   {fifo.dropped_samples}")

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

# --- RMSE analysis ---

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

# --- Entry point ---

if __name__ == "__main__":
  patient_number = 215
  record_path = f"ecg_data/patient_{patient_number}/{patient_number}"

  float_lead0, float_lead1 = load_data(record_path)
  fixed_lead0 = (float_lead0 * FIXED_POINT_SCALE).astype(np.int32)
  fixed_lead1 = (float_lead1 * FIXED_POINT_SCALE).astype(np.int32)

  actual_bpm = get_patient_bpm(patient_number)

  print("Running Floating Point Simulation...")
  f_rec0, f_rec1, f_thresh0, f_thresh1, float_bpm, f_clock, f_units0, f_units1 = \
    run_simulation(float_lead0, float_lead1, is_fixed=False)

  print("Running Fixed Point Simulation...")
  x_rec0, x_rec1, x_thresh0, x_thresh1, fixed_bpm, x_clock, x_units0, x_units1 = \
    run_simulation(fixed_lead0, fixed_lead1, is_fixed=True)

  # Results
  print(f"\nPatient #{patient_number}:")
  print(f"  Actual BPM (Database): {actual_bpm}")
  print(f"  Float Mode BPM:        {float_bpm:.1f}")
  print(f"  Fixed Mode BPM:        {fixed_bpm:.1f}")
  print(f"\n  Float total cycles: {f_clock.cycle}")
  print(f"  Fixed total cycles: {x_clock.cycle}")

  # Performance metrics
  print_performance(f_units0, "Float Lane 0")
  print_performance(f_units1, "Float Lane 1")
  print_performance(x_units0, "Fixed Lane 0")
  print_performance(x_units1, "Fixed Lane 1")

  # FIFO stats
  print_fifo_stats(f_units0[1], "Float Lane 0")
  print_fifo_stats(f_units1[1], "Float Lane 1")
  print_fifo_stats(x_units0[1], "Fixed Lane 0")
  print_fifo_stats(x_units1[1], "Fixed Lane 1")

  # RMSE
  compute_rmse(f_rec0, x_rec0)

  # Plots

  # TODO: Lead 1 is not plotting correctly
  plot_recorders(f_rec0, f"Patient {patient_number} - Float Lane 0")
  plot_recorders(f_rec1, f"Patient {patient_number} - Float Lane 1")

  plot_recorders(x_rec0, f"Patient {patient_number} - Fixed Lane 0")
  plot_recorders(x_rec1, f"Patient {patient_number} - Fixed Lane 1")

# import wfdb
# import numpy as np
# import matplotlib.pyplot as plt

# from clock_unit import ClockUnit
# from data_uploader import DataUploader
# from fifo import InputFIFO
# from mac_unit import MACUnit
# from squaring_unit import SquaringUnit
# from mwi_unit import MWIUnit
# from threshold_unit import ThresholdUnit
# from data_recorder import DataRecorder
# from config import (
#   FIXED_POINT_SCALE, SAMPLE_RATE,
#   CYCLES_PER_SAMPLE, FIFO_SIZE, DATA_RECORDER_CAPACITY,
#   MAX_SAMPLES
# )

# # --- Data loading helpers ---

# def get_patient_bpm(patient_number, data_dir="ecg_data"):
#   record_path = f"{data_dir}/patient_{patient_number}/{patient_number}"
#   record = wfdb.rdrecord(record_path)
#   annotation = wfdb.rdann(record_path, "atr")
#   r_peaks = annotation.sample[np.isin(annotation.symbol, ["N","L","R","B","A","a","J","S","V","F","e","j"])]
#   rr_intervals = np.diff(r_peaks) / record.fs
#   return round(60 / np.median(rr_intervals))

# def load_lead0(record_path: str):
#   record = wfdb.rdrecord(record_path)
#   lead0 = record.p_signal[:, 0].astype(np.float32)[:MAX_SAMPLES]
#   return lead0

# # --- Pipeline builder ---

# def build_lane(name: str, samples: np.ndarray, is_fixed: bool, cycles_per_sample: int):
#   # Recorders
#   raw_recorder       = DataRecorder(f"{name}_raw",       DATA_RECORDER_CAPACITY)
#   lp_recorder        = DataRecorder(f"{name}_lp",        DATA_RECORDER_CAPACITY)
#   hp_recorder        = DataRecorder(f"{name}_hp",        DATA_RECORDER_CAPACITY)
#   dv_recorder        = DataRecorder(f"{name}_dv",        DATA_RECORDER_CAPACITY)
#   squaring_recorder  = DataRecorder(f"{name}_squaring",  DATA_RECORDER_CAPACITY)
#   mwi_recorder       = DataRecorder(f"{name}_mwi",       DATA_RECORDER_CAPACITY)
#   threshold_recorder = DataRecorder(f"{name}_threshold", DATA_RECORDER_CAPACITY)

#   # Hardware units
#   fifo      = InputFIFO(f"{name}_fifo", FIFO_SIZE)
#   uploader  = DataUploader(f"{name}_uploader", samples, cycles_per_sample, fifo)
#   mac       = MACUnit(f"{name}_mac", fifo, is_fixed_point=is_fixed)
#   squaring  = SquaringUnit(f"{name}_squaring", is_fixed_point=is_fixed)
#   mwi       = MWIUnit(f"{name}_mwi", is_fixed_point=is_fixed)
#   threshold = ThresholdUnit(f"{name}_threshold", SAMPLE_RATE, is_fixed_point=is_fixed)

#   # Attach recorders
#   mac.attach_raw_recorder(raw_recorder)
#   mac.attach_lp_recorder(lp_recorder)
#   mac.attach_hp_recorder(hp_recorder)
#   mac.attach_dv_recorder(dv_recorder)
#   squaring.attach_recorder(squaring_recorder)
#   mwi.attach_recorder(mwi_recorder)
#   threshold.attach_recorder(threshold_recorder)

#   # Wire pipeline
#   mac.connect(squaring).connect(mwi).connect(threshold)

#   units = [uploader, fifo, mac, squaring, mwi, threshold]

#   recorders = {
#     "raw":       raw_recorder,
#     "lp":        lp_recorder,
#     "hp":        hp_recorder,
#     "dv":        dv_recorder,
#     "squaring":  squaring_recorder,
#     "mwi":       mwi_recorder,
#     "threshold": threshold_recorder,
#   }

#   return units, fifo, mac, threshold, recorders

# # --- Simulation runner (single lane) ---

# def run_simulation(lead0_samples, is_fixed: bool):
#   mode = "Fixed" if is_fixed else "Float"
#   total_samples = len(lead0_samples)

#   units, fifo, mac, threshold, recorders = build_lane("lane0", lead0_samples, is_fixed, CYCLES_PER_SAMPLE)

#   clock = ClockUnit()
#   clock.subscribe_many(units)

#   uploader = units[0]

#   log_interval = CYCLES_PER_SAMPLE * 1000
#   while True:
#     clock.tick()

#     if clock.cycle % log_interval == 0:
#       samples_sent = uploader.sample_index
#       pct = samples_sent / total_samples * 100
#       print(
#         f"  [{mode}] cycle={clock.cycle:>8} | "
#         f"samples={samples_sent}/{total_samples} ({pct:5.1f}%) | "
#         f"fifo_depth={fifo.max_depth} | "
#         f"peaks={len(threshold.peaks)}"
#       )

#     uploaders_done = not uploader.active
#     fifo_empty = fifo.is_empty()
#     pipeline_drained = all(
#       not u.busy and u.output_data is None and u.input_data is None
#       for u in units
#       if hasattr(u, 'busy')
#     )
#     if uploaders_done and fifo_empty and pipeline_drained:
#       break

#   print(
#     f"  [{mode}] Done — total cycles={clock.cycle} | "
#     f"peaks={len(threshold.peaks)} | "
#     f"dropped={fifo.dropped_samples}"
#   )

#   bpm = threshold.get_bpm()
#   return recorders, threshold, bpm, clock, units

# # --- Reporting ---

# def print_performance(units: list, label: str):
#   print(f"\n{label} Performance:")
#   print(f"  {'Unit':<30} {'Busy':>8} {'Idle':>8} {'Stalled':>8} {'Util':>8}")
#   print(f"  {'-'*64}")
#   for u in units:
#     if hasattr(u, 'busy_cycles'):
#       print(f"  {u.name:<30} {u.busy_cycles:>8} {u.idle_cycles:>8} {u.stalled_cycles:>8} {u.utilization:>8.1%}")

# def print_fifo_stats(fifo: InputFIFO, label: str):
#   print(f"\n{label} FIFO Stats:")
#   print(f"  Max depth reached: {fifo.max_depth}/{fifo.capacity}")
#   print(f"  Dropped samples:   {fifo.dropped_samples}")

# # --- Plotting ---

# def plot_recorders(recorders: dict, label: str):
#   stages = ["raw", "lp", "hp", "dv", "squaring", "mwi", "threshold"]
#   n = len(stages)
#   fig, axes = plt.subplots(n, 1, figsize=(12, 3*n), sharex=True)

#   for ax, stage in zip(axes, stages):
#     signal = recorders[stage].get_signal()
#     ax.plot(signal)
#     ax.set_title(recorders[stage].name)
#     ax.set_ylabel("Amplitude")
#     ax.grid(True, alpha=0.3)

#   axes[-1].set_xlabel("Sample")
#   plt.suptitle(label, fontsize=16)
#   plt.tight_layout()
#   plt.show()

# # --- RMSE analysis ---

# def compute_rmse(float_recorders, fixed_recorders):
#   stages = ["raw", "lp", "hp", "dv", "squaring", "mwi"]
#   print("\nFloat vs Fixed RMSE:")
#   for stage in stages:
#     f_sig = np.array(float_recorders[stage].get_signal())
#     x_sig = np.array(fixed_recorders[stage].get_signal()) / FIXED_POINT_SCALE
#     min_len = min(len(f_sig), len(x_sig))
#     if min_len == 0:
#       continue
#     rmse = np.sqrt(np.mean((f_sig[:min_len] - x_sig[:min_len])**2))
#     print(f"  {stage:<12} RMSE: {rmse:.8f}")

# # --- Entry point ---

# if __name__ == "__main__":
#   patient_number = 116
#   record_path = f"ecg_data/patient_{patient_number}/{patient_number}"

#   float_lead0 = load_lead0(record_path)
#   fixed_lead0 = (float_lead0 * FIXED_POINT_SCALE).astype(np.int32)

#   actual_bpm = get_patient_bpm(patient_number)

#   print("Running Floating Point Simulation (Lead 0)...")
#   f_rec, f_thresh, float_bpm, f_clock, f_units = run_simulation(float_lead0, is_fixed=False)

#   print("Running Fixed Point Simulation (Lead 0)...")
#   x_rec, x_thresh, fixed_bpm, x_clock, x_units = run_simulation(fixed_lead0, is_fixed=True)

#   # Results
#   print(f"\nPatient #{patient_number} (Lead 0):")
#   print(f"  Actual BPM (Database): {actual_bpm}")
#   print(f"  Float Mode BPM:        {float_bpm:.1f}")
#   print(f"  Fixed Mode BPM:        {fixed_bpm:.1f}")
#   print(f"\n  Float total cycles: {f_clock.cycle}")
#   print(f"  Fixed total cycles: {x_clock.cycle}")

#   # Performance metrics
#   print_performance(f_units, "Float Lane 0")
#   print_performance(x_units, "Fixed Lane 0")

#   # FIFO stats
#   print_fifo_stats(f_units[1], "Float Lane 0")
#   print_fifo_stats(x_units[1], "Fixed Lane 0")

#   # RMSE
#   compute_rmse(f_rec, x_rec)

#   # Plots
#   plot_recorders(f_rec, f"Patient {patient_number} - Float Lane 0")
#   plot_recorders(x_rec, f"Patient {patient_number} - Fixed Lane 0")

# import wfdb
# import numpy as np
# import matplotlib.pyplot as plt

# from mac_unit import MACUnit
# from squaring_unit import SquaringUnit
# from mwi_unit import MWIUnit
# from threshold_unit import ThresholdUnit
# from data_recorder import DataRecorder
# from fifo import InputFIFO
# from config import (
#   FIXED_POINT_SCALE, SAMPLE_RATE,
#   CYCLES_PER_SAMPLE, FIFO_SIZE, DATA_RECORDER_CAPACITY,
#   MAX_SAMPLES
# )

# # --- Data loading helpers ---

# def get_patient_bpm(patient_number, data_dir="ecg_data"):
#   record_path = f"{data_dir}/patient_{patient_number}/{patient_number}"
#   record = wfdb.rdrecord(record_path)
#   annotation = wfdb.rdann(record_path, "atr")
#   r_peaks = annotation.sample[np.isin(annotation.symbol, ["N","L","R","B","A","a","J","S","V","F","e","j"])]
#   rr_intervals = np.diff(r_peaks) / record.fs
#   return round(60 / np.median(rr_intervals))

# def load_lead0(record_path: str):
#   record = wfdb.rdrecord(record_path)
#   lead0 = record.p_signal[:, 0].astype(np.float32)[:MAX_SAMPLES]
#   return lead0

# # --- Fast simulation runner ---
# # Instead of ticking every clock cycle, we skip idle cycles entirely.
# # Each sample arrives every CYCLES_PER_SAMPLE cycles. We just call the
# # kernel functions directly once per sample, then compute cycle counts
# # analytically from the known latency values.

# def run_simulation_fast(samples, is_fixed: bool):
#   mode = "Fixed" if is_fixed else "Float"
#   total_samples = len(samples)

#   # Build units (no clock, no uploader, no FIFO needed)
#   fifo      = InputFIFO("fifo", FIFO_SIZE)  # unused but needed for MACUnit constructor
#   mac       = MACUnit("mac", fifo, is_fixed_point=is_fixed)
#   squaring  = SquaringUnit("squaring", is_fixed_point=is_fixed)
#   mwi       = MWIUnit("mwi", is_fixed_point=is_fixed)
#   threshold = ThresholdUnit("threshold", SAMPLE_RATE, is_fixed_point=is_fixed)

#   # Recorders
#   raw_recorder       = DataRecorder("raw",       DATA_RECORDER_CAPACITY)
#   lp_recorder        = DataRecorder("lp",        DATA_RECORDER_CAPACITY)
#   hp_recorder        = DataRecorder("hp",        DATA_RECORDER_CAPACITY)
#   dv_recorder        = DataRecorder("dv",        DATA_RECORDER_CAPACITY)
#   squaring_recorder  = DataRecorder("squaring",  DATA_RECORDER_CAPACITY)
#   mwi_recorder       = DataRecorder("mwi",       DATA_RECORDER_CAPACITY)
#   threshold_recorder = DataRecorder("threshold", DATA_RECORDER_CAPACITY)

#   mac.attach_raw_recorder(raw_recorder)
#   mac.attach_lp_recorder(lp_recorder)
#   mac.attach_hp_recorder(hp_recorder)
#   mac.attach_dv_recorder(dv_recorder)
#   squaring.attach_recorder(squaring_recorder)
#   mwi.attach_recorder(mwi_recorder)
#   threshold.attach_recorder(threshold_recorder)

#   # Process each sample directly — no cycle-by-cycle ticking
#   log_interval = total_samples // 10
#   for i, sample in enumerate(samples):
#     if log_interval > 0 and i % log_interval == 0:
#       pct = i / total_samples * 100
#       print(f"  [{mode}] sample={i}/{total_samples} ({pct:.0f}%) peaks={len(threshold.peaks)}")

#     if i % 100000 == 0:
#       print(f"  sample={i} threshold={threshold.threshold:.4f} spki={threshold.spki:.4f} npki={threshold.npki:.4f}")

#     raw_recorder.record(sample)

#     lp  = mac._run_lp_kernel(sample)
#     hp  = mac._run_hp_kernel(lp)
#     dv  = mac._run_dv_kernel(hp)

#     sq  = squaring.compute(dv)
#     squaring_recorder.record(sq)

#     m   = mwi.compute(sq)
#     mwi_recorder.record(m)

#     t   = threshold.compute(m)
#     threshold_recorder.record(t)

#   # --- Analytical cycle count ---
#   # Each sample takes exactly CYCLES_PER_SAMPLE cycles (ADC rate dominates)
#   # The pipeline compute latency is:
#   #   mac_latency + squaring_latency + mwi_latency + threshold_latency
#   # But since mac_latency << CYCLES_PER_SAMPLE (for fixed) or ~= CYCLES_PER_SAMPLE (for float),
#   # total cycles is always dominated by the ADC rate.
#   mac_latency       = mac.latency_cycles
#   squaring_latency  = squaring.latency_cycles
#   mwi_latency       = mwi.latency_cycles
#   threshold_latency = threshold.latency_cycles
#   pipeline_latency  = mac_latency + squaring_latency + mwi_latency + threshold_latency

#   # Total cycles = samples * cycles_per_sample + pipeline drain time
#   total_cycles = total_samples * CYCLES_PER_SAMPLE + pipeline_latency

#   # Busy cycles per unit = latency * number of samples processed
#   mac_busy       = mac_latency * total_samples
#   squaring_busy  = squaring_latency * total_samples
#   mwi_busy       = mwi_latency * total_samples
#   threshold_busy = threshold_latency * total_samples

#   # Idle cycles = total - busy (stalls are zero as we established)
#   mac_idle       = total_cycles - mac_busy
#   squaring_idle  = total_cycles - squaring_busy
#   mwi_idle       = total_cycles - mwi_busy
#   threshold_idle = total_cycles - threshold_busy

#   bpm = threshold.get_bpm()

#   recorders = {
#     "raw":       raw_recorder,
#     "lp":        lp_recorder,
#     "hp":        hp_recorder,
#     "dv":        dv_recorder,
#     "squaring":  squaring_recorder,
#     "mwi":       mwi_recorder,
#     "threshold": threshold_recorder,
#   }

#   perf = {
#     "total_cycles":    total_cycles,
#     "pipeline_latency": pipeline_latency,
#     "units": [
#       {"name": "mac",       "latency": mac_latency,       "busy": mac_busy,       "idle": mac_idle},
#       {"name": "squaring",  "latency": squaring_latency,  "busy": squaring_busy,  "idle": squaring_idle},
#       {"name": "mwi",       "latency": mwi_latency,       "busy": mwi_busy,       "idle": mwi_idle},
#       {"name": "threshold", "latency": threshold_latency, "busy": threshold_busy, "idle": threshold_idle},
#     ]
#   }

#   print(f"  [{mode}] Done — total_cycles={total_cycles} | peaks={len(threshold.peaks)}")

#   return recorders, threshold, bpm, perf

# # --- Reporting ---

# def print_performance(perf: dict, label: str):
#   total = perf["total_cycles"]
#   print(f"\n{label} Performance (analytical):")
#   print(f"  Total cycles:     {total}")
#   print(f"  Pipeline latency: {perf['pipeline_latency']} cycles/sample")
#   print(f"\n  {'Unit':<15} {'Latency':>10} {'Busy':>12} {'Idle':>12} {'Util':>8}")
#   print(f"  {'-'*60}")
#   for u in perf["units"]:
#     util = u["busy"] / total if total > 0 else 0
#     print(f"  {u['name']:<15} {u['latency']:>10} {u['busy']:>12} {u['idle']:>12} {util:>8.1%}")

# # --- RMSE ---

# def compute_rmse(float_recorders, fixed_recorders):
#   stages = ["raw", "lp", "hp", "dv", "squaring", "mwi"]
#   print("\nFloat vs Fixed RMSE:")
#   for stage in stages:
#     f_sig = np.array(float_recorders[stage].get_signal())
#     x_sig = np.array(fixed_recorders[stage].get_signal()) / FIXED_POINT_SCALE
#     min_len = min(len(f_sig), len(x_sig))
#     if min_len == 0:
#       continue
#     rmse = np.sqrt(np.mean((f_sig[:min_len] - x_sig[:min_len])**2))
#     print(f"  {stage:<12} RMSE: {rmse:.8f}")

# # --- Plotting ---

# def plot_recorders(recorders: dict, label: str):
#   stages = ["raw", "lp", "hp", "dv", "squaring", "mwi", "threshold"]
#   n = len(stages)
#   fig, axes = plt.subplots(n, 1, figsize=(12, 3*n), sharex=True)

#   for ax, stage in zip(axes, stages):
#     signal = recorders[stage].get_signal()
#     ax.plot(signal)
#     ax.set_title(recorders[stage].name)
#     ax.set_ylabel("Amplitude")
#     ax.grid(True, alpha=0.3)

#   axes[-1].set_xlabel("Sample")
#   plt.suptitle(label, fontsize=16)
#   plt.tight_layout()
#   plt.show()

# # --- Entry point ---

# if __name__ == "__main__":
#   patient_number = 123
#   record_path = f"ecg_data/patient_{patient_number}/{patient_number}"

#   float_lead0 = load_lead0(record_path)
#   fixed_lead0 = (float_lead0 * FIXED_POINT_SCALE).astype(np.int32)

#   actual_bpm = get_patient_bpm(patient_number)

#   print("Running Floating Point Simulation (fast)...")
#   f_rec, f_thresh, float_bpm, f_perf = run_simulation_fast(float_lead0, is_fixed=False)

#   print("Running Fixed Point Simulation (fast)...")
#   x_rec, x_thresh, fixed_bpm, x_perf = run_simulation_fast(fixed_lead0, is_fixed=True)

#   # Results
#   print(f"\nPatient #{patient_number} (Lead 0):")
#   print(f"  Actual BPM (Database): {actual_bpm}")
#   print(f"  Float Mode BPM:        {float_bpm:.1f}")
#   print(f"  Fixed Mode BPM:        {fixed_bpm:.1f}")

#   print_performance(f_perf, "Float Lane 0")
#   print_performance(x_perf, "Fixed Lane 0")

#   compute_rmse(f_rec, x_rec)

#   plot_recorders(f_rec, f"Patient {patient_number} - Float Lane 0")
#   plot_recorders(x_rec, f"Patient {patient_number} - Fixed Lane 0")