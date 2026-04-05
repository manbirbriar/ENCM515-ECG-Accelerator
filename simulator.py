import wfdb
import numpy as np
import matplotlib.pyplot as plt
from clock_unit import ClockUnit
from data_uploader import DataUploader
from fifo_buffer import FIFOBuffer
from mac_unit import MACUnit
from squaring_unit import SquaringUnit
from mwi_unit import MWIUnit
from threshold_unit import ThresholdUnit
from data_recorder import DataRecorder
from config import (
  FIXED_POINT_SCALE, SAMPLE_RATE, DATA_RECORDER_CAPACITY, MAX_SAMPLES,
  BATTERY_CAPACITY_MAH, BATTERY_VOLTAGE, DYNAMIC_POWER_UW_PER_MHZ, SWEEP_FREQUENCIES_HZ
)


def get_patient_bpm(patient_number, data_dir="ecg_data"):
  """
  Gets the patients actual BPM.
  """
  record_path = f"{data_dir}/patient_{patient_number}/{patient_number}"
  record = wfdb.rdrecord(record_path)
  annotation = wfdb.rdann(record_path, "atr")
  r_peaks = annotation.sample[np.isin(annotation.symbol, ["N","L","R","B","A","a","J","S","V","F","e","j"])]
  rr_intervals = np.diff(r_peaks) / record.fs
  return round(60 / np.median(rr_intervals))
  
  
# TODO: Do we want to make this some sort of hardware unit?
def load_data(record_path: str):
  """
  Load both ECG leads and combine them into a single signal before processing.
  """
  record = wfdb.rdrecord(record_path)
  lead0 = record.p_signal[:, 0].astype(np.float32)[:MAX_SAMPLES]
  lead1 = record.p_signal[:, 1].astype(np.float32)[:MAX_SAMPLES]
  combined = (lead0 + lead1) / 2.0
  return combined
  

def build_pipeline(samples: np.ndarray, is_fixed: bool, cycles_per_sample: int, fifo_size: int):
  """
  Builds the single processing pipeline for the combined ECG signal.
  """
  raw_recorder = DataRecorder("raw", DATA_RECORDER_CAPACITY)
  lp_recorder = DataRecorder("lp", DATA_RECORDER_CAPACITY)
  hp_recorder = DataRecorder("hp", DATA_RECORDER_CAPACITY)
  dv_recorder = DataRecorder("dv", DATA_RECORDER_CAPACITY)
  squaring_recorder = DataRecorder("squaring", DATA_RECORDER_CAPACITY)
  mwi_recorder = DataRecorder("mwi", DATA_RECORDER_CAPACITY)
  threshold_recorder = DataRecorder("threshold", DATA_RECORDER_CAPACITY)
  
  fifo = FIFOBuffer("fifo", fifo_size)
  uploader = DataUploader("uploader", samples, cycles_per_sample, fifo)
  mac = MACUnit("mac", fifo, is_fixed_point=is_fixed)
  squaring = SquaringUnit("squaring", is_fixed_point=is_fixed)
  mwi = MWIUnit("mwi", is_fixed_point=is_fixed)
  threshold = ThresholdUnit("threshold", SAMPLE_RATE, is_fixed_point=is_fixed)
  
  mac.attach_raw_recorder(raw_recorder)
  mac.attach_lp_recorder(lp_recorder)
  mac.attach_hp_recorder(hp_recorder)
  mac.attach_dv_recorder(dv_recorder)
  squaring.attach_recorder(squaring_recorder)
  mwi.attach_recorder(mwi_recorder)
  threshold.attach_recorder(threshold_recorder)
  
  mac.connect(squaring).connect(mwi).connect(threshold)
  
  units = [uploader, fifo, mac, squaring, mwi, threshold]
  
  recorders = {"raw": raw_recorder, "lp": lp_recorder, "hp": hp_recorder, "dv": dv_recorder,
    "squaring": squaring_recorder, "mwi": mwi_recorder, "threshold": threshold_recorder,
  }
  
  return units, fifo, mac, threshold, recorders
  
  
def run_simulation(samples, is_fixed: bool, clock_frequency_hz: int):
  """
  Run the single-lane pipeline simulation on the multi-lead ECG signal.
  """
  mode = "Fixed" if is_fixed else "Float"
  
  cycles_per_sample = clock_frequency_hz // SAMPLE_RATE
  fifo_size = cycles_per_sample * 2
  
  units, fifo, mac, threshold, recorders = build_pipeline(samples, is_fixed, cycles_per_sample, fifo_size)
  
  clock = ClockUnit()
  clock.subscribe_many(units)
  
  uploader = units[0]

  while True:
    clock.tick()
  
    uploaders_done = not uploader.active
    fifo_empty = fifo.is_empty()
    pipeline_drained = all(
      not u.busy and u.output_data is None and u.input_data is None
      and getattr(u, "cycles_remaining", 0) == 0
      and getattr(u, "kernel_cycles_remaining", 0) == 0
      for u in units
      if hasattr(u, "busy")
    )
    if uploaders_done and fifo_empty and pipeline_drained:
      break
  
  bpm = threshold.get_bpm()
  return recorders, threshold, bpm, clock, units, fifo
  

def compute_battery_life(clock_frequency_hz: int) -> dict:
  """
  Battery life is calculated using the ARM Cortex-M4 40LP dynamic power model.
  """
  clock_mhz = clock_frequency_hz / 1_000_000
  dynamic_power_uW = DYNAMIC_POWER_UW_PER_MHZ * clock_mhz
  dynamic_power_mW = dynamic_power_uW / 1000.0
  current_ma = dynamic_power_mW / BATTERY_VOLTAGE
  battery_life_hours = BATTERY_CAPACITY_MAH / current_ma
  battery_life_days = battery_life_hours / 24.0
  
  return {
    "clock_mhz": clock_mhz, "dynamic_power_uW": dynamic_power_uW, "current_ma": current_ma,
    "battery_life_hours": battery_life_hours, "battery_life_days": battery_life_days,
  }
  

def run_frequency_sweep(float_samples, fixed_samples):
  """
  Run both fixed and float pipelines at each frequency.
  Battery life differs between fixed and float because each mode has a different minimum viable clock frequency.
  """
  results = []
  
  for f_hz in SWEEP_FREQUENCIES_HZ:
    cycles_per_sample = f_hz // SAMPLE_RATE
    # print(f"\n\nClock: {f_hz} Hz")
    # print(f"Cycles per Sample: {cycles_per_sample}")
  
    battery = compute_battery_life(f_hz)
  
    for label, is_fixed, samples in [("Float", False, float_samples), ("Fixed", True, fixed_samples)]:
      # print(f"\n{label}:")
      recorders, threshold, bpm, clock, units, fifo = run_simulation(samples, is_fixed=is_fixed, clock_frequency_hz=f_hz)
  
      total_cycles = clock.cycle
      dropped = fifo.dropped_samples
      valid = dropped == 0
  
      # print(f"Total cycles: {total_cycles}")
      # print(f"Dropped samples: {dropped}")
      # print(f"BPM detected: {bpm:.4f}")
      # print(f"Power draw: {battery["dynamic_power_uW"]:.4f} µW")
      # print(f"Battery life: {battery["battery_life_hours"]:.4f} h ({battery["battery_life_days"]:.4f} days)")
      # print(f"Validity: {"VALID" if valid else "INVALID (samples dropped)"}")
  
      results.append({
        "label": f"{label} {f_hz//1000}kHz",
        "mode": label,
        "freq_hz": f_hz,
        "freq_khz": f_hz / 1000,
        "total_cycles": total_cycles,
        "dropped": dropped,
        "valid": valid,
        "bpm": bpm,
        "dynamic_power_uW": battery["dynamic_power_uW"],
        "battery_life_hours": battery["battery_life_hours"],
        "battery_life_days": battery["battery_life_days"],
        "recorders": recorders,
        "threshold": threshold,
        "units": units,
        "fifo": fifo,
      })
  
  return results
  

def print_sweep_summary(results: list):
  print("\nFrequency Sweeps")
  print(f"{"Configuration":<22} {"Freq(kHz)":>10} {"Dropped":>9} {"BPM":>7} {"Power(µW)":>11} {"Battery(hours)":>12} {"Battery(days)":>14} {"Valid":>6}")
  for r in results:
    print(
      f"{r["label"]:<22} {r["freq_khz"]:>10.1f} {r["dropped"]:>9} {r["bpm"]:>7.1f} {r["dynamic_power_uW"]:>11.4f} "
      f"{r["battery_life_hours"]:>12.1f} {r["battery_life_days"]:>14.1f} {"Yes" if r["valid"] else "No":>6}"
    )
  

def print_performance(units: list, label: str):
  print(f"\n{label} Performance:")
  print(f"{"Unit":<30} {"Busy":>8} {"Idle":>8} {"Stalled":>8} {"Util":>8}")
  for u in units:
    if hasattr(u, "busy_cycles"):
      print(f"{u.name:<30} {u.busy_cycles:>8} {u.idle_cycles:>8} {u.stalled_cycles:>8} {u.utilization:>8.1%}")
  
  
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
  
  
def compute_rmse(float_recorders, fixed_recorders, label: str):
  stages = ["raw", "lp", "hp", "dv", "squaring", "mwi"]
  print(f"\nFloat vs Fixed RMSE: {label}")
  for stage in stages:
    f_sig = np.array(float_recorders[stage].get_signal())
    x_sig = np.array(fixed_recorders[stage].get_signal()) / FIXED_POINT_SCALE
    min_len = min(len(f_sig), len(x_sig))
    if min_len == 0:
      continue
    rmse = np.sqrt(np.mean((f_sig[:min_len] - x_sig[:min_len])**2))
    print(f"{stage:<12} RMSE: {rmse:.8f}")
  

if __name__ == "__main__":
  patient_number = 215
  record_path = f"ecg_data/patient_{patient_number}/{patient_number}"
  
  float_data = load_data(record_path)
  fixed_data = (float_data * FIXED_POINT_SCALE).astype(np.int32)
  
  actual_bpm = get_patient_bpm(patient_number)
  print(f"Patient #{patient_number}")
  print(f"Actual BPM (Database): {actual_bpm}")
  
  results = run_frequency_sweep(float_data, fixed_data)
  print_sweep_summary(results)

  for f_hz in SWEEP_FREQUENCIES_HZ:
    float_result = next(r for r in results if r["mode"] == "Float" and r["freq_hz"] == f_hz)
    fixed_result = next(r for r in results if r["mode"] == "Fixed" and r["freq_hz"] == f_hz)

    compute_rmse(float_result["recorders"], fixed_result["recorders"], label=f"{f_hz//1000}kHz")

    print_performance(float_result["units"], f"Float {f_hz//1000}kHz")
    print_performance(fixed_result["units"], f"Fixed {f_hz//1000}kHz")

    plot_recorders(float_result["recorders"], f"Patient {patient_number} Float {f_hz//1000}kHz")
    plot_recorders(fixed_result["recorders"], f"Patient {patient_number} Fixed {f_hz//1000}kHz")