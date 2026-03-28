import wfdb
import numpy as np
from clock_unit import ClockUnit
from sample_queue import SampleQueue
from data_uploader import DataUploader
from scheduler import Scheduler
from derivative_unit import DerivativeUnit
from squaring_unit import SquaringUnit
from data_recorder import DataRecorder
from low_pass_unit import LowPassUnit
from config import FIXED_POINT_SCALE, WINDOW_SIZE, HOP_SIZE, QUEUE_SIZE, NUM_LANES, DATA_RECORDER_CAPACITY, SAMPLE_RATE
import matplotlib.pyplot as plt
from high_pass_unit import HighPassUnit

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
def plot_data_recorders(recorders: list[DataRecorder], sample_rate: int = 360) -> None:
  n = len(recorders)
  if n == 0:
    print("No data to plot.")
    return

  fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
  
  # Handle the case where there is only one subplot (axes is not a list)
  if n == 1:
    axes = [axes]

  for ax, recorder in zip(axes, recorders):
    signal = recorder.get_signal()
    time = [i / sample_rate for i in range(len(signal))]
    ax.plot(time, signal, linewidth=0.8)
    ax.set_title(recorder.name)
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

  axes[-1].set_xlabel("Time (s)")
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  # patient_number = input("Enter the patient number: ")
  patient_number = 100

  record_path = f"ecg_data/patient_{patient_number}/{patient_number}"

  float_samples = load_float_samples(record_path)
  fixed_samples = load_fixed_samples(record_path)

  raw_ecg_data_recorder = DataRecorder("raw_ecg_data_recorder", capacity=DATA_RECORDER_CAPACITY, hop_size=HOP_SIZE)
  low_pass_data_recorder = DataRecorder("low_pass_data_recorder", capacity=DATA_RECORDER_CAPACITY, hop_size=HOP_SIZE)
  high_pass_data_recorder = DataRecorder("high_pass_data_recorder", capacity=DATA_RECORDER_CAPACITY, hop_size=HOP_SIZE)
  derivative_data_recorder = DataRecorder("derivative_data_recorder", capacity=DATA_RECORDER_CAPACITY, hop_size=HOP_SIZE)
  squaring_data_recorder = DataRecorder("squaring_data_recorder", capacity=DATA_RECORDER_CAPACITY, hop_size=HOP_SIZE)

  data_recorders = [raw_ecg_data_recorder, low_pass_data_recorder, high_pass_data_recorder, derivative_data_recorder, squaring_data_recorder]

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
  
  # TODO: Fix issue with incorrect amplitude
  squaring_unit = SquaringUnit("squaring_unit")
  squaring_unit.attach_recorder(squaring_data_recorder)
  
  low_pass_unit.connect(high_pass_unit).connect(derivative_unit).connect(squaring_unit)

  scheduler = Scheduler("scheduler", sample_queue, lanes=[low_pass_unit])
  scheduler.attach_recorder(raw_ecg_data_recorder)

  units = [sample_queue, data_uploader, scheduler, low_pass_unit, high_pass_unit, derivative_unit, squaring_unit]
  clock_unit = ClockUnit()
  clock_unit.subscribe_many(units)

  # ensures that the loop doesn't stop running until all data has gone through the system
  # while not data_uploader.is_done() or not sample_queue.is_empty() or not all(unit.is_available() for unit in clock_unit.units):
  for _ in range(1500000):
    clock_unit.tick()

    if clock_unit.cycle % 100000 == 0:
      print(clock_unit)

  plot_data_recorders(data_recorders, sample_rate=SAMPLE_RATE)

  print(clock_unit)
  print(data_uploader)
  print(sample_queue)
  print(scheduler)
  print(low_pass_unit)
  print(high_pass_unit)
  print(derivative_unit)
  print(squaring_unit)