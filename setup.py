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
from config import FIXED_POINT_SCALE, WINDOW_SIZE, HOP_SIZE, QUEUE_SIZE, DATA_RECORDER_CAPACITY, SAMPLE_RATE
import matplotlib.pyplot as plt
from high_pass_unit import HighPassUnit
from threshold_unit import ThresholdUnit

def get_reference_bpm(record_path, start_sample, end_sample):
  # 1. Load the 'atr' (attribute) file
  # This contains the indices of every heartbeat marked by a cardiologist
  annotation = wfdb.rdann(record_path, 'atr', sampfrom=start_sample, sampto=end_sample)
  
  # 2. Filter for actual 'beat' symbols (N, L, R, B, A, a, J, S, V, r, F, e, j, n, E, f, /)
  # Some annotations are just 'comments' or 'noise' markers, not heartbeats
  beat_symbols = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', 'f', '/']
  true_beats = [s for s in annotation.symbol if s in beat_symbols]
  
  num_beats = len(true_beats)
  duration_seconds = (end_sample - start_sample) / 360.0
  
  ref_bpm = (num_beats / duration_seconds) * 60
  return ref_bpm

# TODO: Confirm that this is working correctly
def load_float_samples(record_path: str, channel: int = 0) -> np.ndarray:
  record = wfdb.rdrecord(record_path)
  raw = record.p_signal[:, channel]
  return raw.astype(np.float32)

# TODO: Confirm that this is working correctly
def load_fixed_samples(record_path: str, channel: int = 0) -> np.ndarray:
  record = wfdb.rdrecord(record_path)
  raw = record.p_signal[:, channel]
  normalised = raw / np.max(np.abs(raw))
  return (normalised * FIXED_POINT_SCALE).astype(np.int16)

# TODO: This is temporary (AI)
def plot_data_recorders(recorders: list[DataRecorder], sample_rate: int, patient_number: int) -> None:
  n = len(recorders)

  fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)

  for ax, recorder in zip(axes, recorders):
    signal = recorder.get_signal()
    time = [i / sample_rate for i in range(len(signal))]
    ax.plot(time, signal, linewidth=0.8)
    ax.set_title(recorder.name)
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

  axes[-1].set_xlabel("Time (s)")
  plt.suptitle(f"Patient {patient_number}", fontsize=24, y=0.99)
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  # patient_number = input("Enter the patient number (100, 101, or 102): ")
  patient_number = 102

  record_path = f"ecg_data/patient_{patient_number}/{patient_number}"

  float_samples = load_float_samples(record_path)
  fixed_samples = load_fixed_samples(record_path)

  raw_ecg_data_recorder = DataRecorder("raw_ecg_data_recorder", capacity=DATA_RECORDER_CAPACITY, hop_size=HOP_SIZE)
  low_pass_data_recorder = DataRecorder("low_pass_data_recorder", capacity=DATA_RECORDER_CAPACITY, hop_size=HOP_SIZE)
  high_pass_data_recorder = DataRecorder("high_pass_data_recorder", capacity=DATA_RECORDER_CAPACITY, hop_size=HOP_SIZE)
  derivative_data_recorder = DataRecorder("derivative_data_recorder", capacity=DATA_RECORDER_CAPACITY, hop_size=HOP_SIZE)
  squaring_data_recorder = DataRecorder("squaring_data_recorder", capacity=DATA_RECORDER_CAPACITY, hop_size=HOP_SIZE)
  mwi_data_recorder = DataRecorder("mwi_data_recorder", capacity=DATA_RECORDER_CAPACITY, hop_size=HOP_SIZE)
  threshold_data_recorder = DataRecorder("threshold_data_recorder", capacity=DATA_RECORDER_CAPACITY, hop_size=HOP_SIZE)

  data_recorders = [raw_ecg_data_recorder, low_pass_data_recorder, high_pass_data_recorder, derivative_data_recorder, squaring_data_recorder, mwi_data_recorder, threshold_data_recorder]

  sample_queue = SampleQueue("sample_queue", queue_size=QUEUE_SIZE, window_size=WINDOW_SIZE, hop_size=HOP_SIZE)
  data_uploader = DataUploader("data_uploader", float_samples)
  data_uploader.connect(sample_queue)

  # TODO: Assign accurate latency values
  low_pass_unit = LowPassUnit("low_pass_unit")
  low_pass_unit.attach_recorder(low_pass_data_recorder)

  high_pass_unit = HighPassUnit("high_pass_unit")
  high_pass_unit.attach_recorder(high_pass_data_recorder)
  
  derivative_unit = DerivativeUnit("derivative_unit")
  derivative_unit.attach_recorder(derivative_data_recorder)
  
  squaring_unit = SquaringUnit("squaring_unit")
  squaring_unit.attach_recorder(squaring_data_recorder)

  mwi_unit = MWIUnit("mwi_unit")
  mwi_unit.attach_recorder(mwi_data_recorder)

  threshold_unit = ThresholdUnit("threshold_unit")
  threshold_unit.attach_recorder(threshold_data_recorder)
  
  low_pass_unit.connect(high_pass_unit).connect(derivative_unit).connect(squaring_unit).connect(mwi_unit).connect(threshold_unit)

  scheduler = Scheduler("scheduler", sample_queue, lanes=[low_pass_unit])
  scheduler.attach_recorder(raw_ecg_data_recorder)

  units = [sample_queue, data_uploader, scheduler, low_pass_unit, high_pass_unit, derivative_unit, squaring_unit, mwi_unit, threshold_unit]
  clock_unit = ClockUnit()
  clock_unit.subscribe_many(units)

  # TODO: this 
  for _ in range(1500000):
    clock_unit.tick()

    if clock_unit.cycle % 100000 == 0:
      print(clock_unit)

  plot_data_recorders(data_recorders, sample_rate=SAMPLE_RATE, patient_number=patient_number)

  print(f"Calculated BPM = {threshold_unit.get_bpm()}")
  print(f"Actual BPM = {get_reference_bpm(record_path, 0, 4140)}")

  print(clock_unit)
  print(data_uploader)
  print(sample_queue)
  print(scheduler)
  print(low_pass_unit)
  print(high_pass_unit)
  print(derivative_unit)
  print(squaring_unit)
  print(mwi_unit)
  print(threshold_unit)