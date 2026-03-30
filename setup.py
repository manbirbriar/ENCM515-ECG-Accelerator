import wfdb
import numpy as np
from clock_unit import ClockUnit
from mwi_unit import MWIUnit
from sample_queue import SampleQueue
from data_uploader import DataUploader
from scheduler import Scheduler
from derivative_unit import DerivativeUnit
from squaring_unit import SquaringUnit
from data_recorder import DataRecorder
from low_pass_unit import LowPassUnit
from config import FIXED_POINT_SCALE, WINDOW_SIZE, HOP_SIZE, QUEUE_SIZE, DATA_RECORDER_CAPACITY, SAMPLE_RATE, VECTOR_WIDTH, MWI_WINDOW_SIZE
import matplotlib.pyplot as plt
from high_pass_unit import HighPassUnit
from threshold_unit import ThresholdUnit

def get_patient_bpm(patient_number, data_dir="ecg_data"):
  record_path = f"{data_dir}/patient_{patient_number}/{patient_number}"
  record = wfdb.rdrecord(record_path)
  annotation = wfdb.rdann(record_path, "atr")
  
  r_peaks = annotation.sample[np.isin(annotation.symbol, ["N", "L", "R", "B", "A", "a", "J", "S", "V", "F", "e", "j"])]
  rr_intervals = np.diff(r_peaks) / record.fs
  median_bpm = round(60 / np.median(rr_intervals))
  
  return median_bpm

def load_data(record_path: str, channel: int = 0):
  record = wfdb.rdrecord(record_path)
  raw = record.p_signal[:, channel]
  return raw.astype(np.float32)

def plot_data_recorders(recorders: list[DataRecorder], patient_number: int) -> None:
  n = len(recorders)

  fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)

  for ax, recorder in zip(axes, recorders):
    signal = recorder.get_signal()
    indices = np.arange(len(signal))
    ax.plot(indices, signal)
    ax.set_title(recorder.name)
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

  axes[-1].set_xlabel("Sample")
  plt.suptitle(f"Patient {patient_number}", fontsize=24)
  plt.tight_layout()
  plt.show()

def run_simulation(patient_samples, is_fixed: bool):
  raw_ecg_data_recorder = DataRecorder("raw_ecg_data_recorder", capacity=DATA_RECORDER_CAPACITY, hop_size=HOP_SIZE)
  low_pass_data_recorder = DataRecorder("low_pass_data_recorder", capacity=DATA_RECORDER_CAPACITY, hop_size=HOP_SIZE)
  high_pass_data_recorder = DataRecorder("high_pass_data_recorder", capacity=DATA_RECORDER_CAPACITY, hop_size=HOP_SIZE)
  derivative_data_recorder = DataRecorder("derivative_data_recorder", capacity=DATA_RECORDER_CAPACITY, hop_size=HOP_SIZE)
  squaring_data_recorder = DataRecorder("squaring_data_recorder", capacity=DATA_RECORDER_CAPACITY, hop_size=HOP_SIZE)
  mwi_data_recorder = DataRecorder("mwi_data_recorder", capacity=DATA_RECORDER_CAPACITY, hop_size=HOP_SIZE)
  threshold_data_recorder = DataRecorder("threshold_data_recorder", capacity=DATA_RECORDER_CAPACITY, hop_size=HOP_SIZE)

  sample_queue = SampleQueue("sample_queue", QUEUE_SIZE, WINDOW_SIZE, HOP_SIZE)
  data_uploader = DataUploader("data_uploader", patient_samples)
  data_uploader.connect(sample_queue)

  low_pass_unit = LowPassUnit("low_pass_unit", WINDOW_SIZE, VECTOR_WIDTH, is_fixed_point=is_fixed)
  low_pass_unit.attach_recorder(low_pass_data_recorder)

  high_pass_unit = HighPassUnit("high_pass_unit", WINDOW_SIZE, VECTOR_WIDTH, is_fixed_point=is_fixed)
  high_pass_unit.attach_recorder(high_pass_data_recorder)
  
  derivative_unit = DerivativeUnit("derivative_unit", WINDOW_SIZE, VECTOR_WIDTH, is_fixed_point=is_fixed)
  derivative_unit.attach_recorder(derivative_data_recorder)

  squaring_unit = SquaringUnit("squaring_unit", WINDOW_SIZE, VECTOR_WIDTH, is_fixed_point=is_fixed)
  squaring_unit.attach_recorder(squaring_data_recorder)

  mwi_unit = MWIUnit("mwi_unit", WINDOW_SIZE, VECTOR_WIDTH, MWI_WINDOW_SIZE, is_fixed_point=is_fixed)
  mwi_unit.attach_recorder(mwi_data_recorder)

  threshold_unit = ThresholdUnit("threshold_unit", WINDOW_SIZE, SAMPLE_RATE, is_fixed_point=is_fixed)
  threshold_unit.attach_recorder(threshold_data_recorder)

  low_pass_unit.connect(high_pass_unit).connect(derivative_unit).connect(squaring_unit).connect(mwi_unit).connect(threshold_unit)
  
  scheduler_unit = Scheduler("sch", sample_queue, lanes=[low_pass_unit])
  scheduler_unit.attach_recorder(raw_ecg_data_recorder)

  units = [sample_queue, data_uploader, scheduler_unit, low_pass_unit, high_pass_unit, derivative_unit, squaring_unit, mwi_unit, threshold_unit]
  clock = ClockUnit()
  clock.subscribe_many(units)

  while True:
    clock.tick()
    if not data_uploader.active and all(not u.busy and not u.output_data for u in units):
      break
  
  return [raw_ecg_data_recorder, low_pass_data_recorder, high_pass_data_recorder, derivative_data_recorder, squaring_data_recorder, mwi_data_recorder, threshold_data_recorder], threshold_unit

if __name__ == "__main__":
  # patient_number = input("Enter the patient number (116, 123, or 215): ")
  patient_number = 116

  record_path = f"ecg_data/patient_{patient_number}/{patient_number}"

  float_samples = load_data(record_path)
  fixed_samples = (float_samples * FIXED_POINT_SCALE).astype(np.int32)

  print("Running Floating Point Simulation...")
  float_recorders, float_unit = run_simulation(float_samples, is_fixed=False)
  float_bpm = float_unit.get_bpm()

  print("Running Fixed Point Simulation...")
  fixed_recorders, fixed_unit = run_simulation(fixed_samples, is_fixed=True)
  fixed_bpm = fixed_unit.get_bpm()

  print(f"\nPatient #{patient_number}:")
  print(f"Actual BPM (Database): {get_patient_bpm(patient_number)}")
  print(f"Float Mode BPM: {float_bpm}")
  print(f"Fixed Mode BPM: {fixed_bpm}")

  print("\nFloat Point vs Fixed Point Results:")
  for i in range(len(float_recorders)-1):
    float_signal = np.array(float_recorders[i].get_signal())
    fixed_signal = np.array(fixed_recorders[i].get_signal()) / FIXED_POINT_SCALE

    min_len = min(len(float_signal), len(fixed_signal))
    
    rmse = np.sqrt(np.mean((float_signal[:min_len] - fixed_signal[:min_len])**2))
    print(f"{float_recorders[i].name} RMSE: {rmse:.8f}")

  plot_data_recorders(fixed_recorders, patient_number=patient_number)
  plot_data_recorders(float_recorders, patient_number=patient_number)