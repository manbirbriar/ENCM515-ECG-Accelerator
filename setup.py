import wfdb
import numpy as np
from clock_unit import ClockUnit
from low_pass_filter import LowPassFilter
from sample_queue import SampleQueue
from data_uploader import DataUploader
from data_visualizer import DataVisualizer
from scheduler import Scheduler

FIXED_POINT_SCALE = 2**15 - 1

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

# TODO: Confirm that this is working correctly
if __name__ == "__main__":
  # patient_number = input("Enter the patient number: ")
  patient_number = 100

  record_path = f"ecg_data/patient_{patient_number}/{patient_number}"

  float_samples = load_float_samples(record_path)
  fixed_samples = load_fixed_samples(record_path)

  # print(f"Float samples: dtype: {float_samples.dtype}, shape: {float_samples.shape}, range: [{float_samples.min()}, {float_samples.max()}]")
  # print(f"Fixed samples: dtype: {fixed_samples.dtype}, shape: {fixed_samples.shape}, range: [{fixed_samples.min()}, {fixed_samples.max()}]")

  sample_queue = SampleQueue("sample_queue", queue_size=216, window_size=72, hop_size=36)
  data_uploader = DataUploader("data_uploader", float_samples)

  low_pass_filter = LowPassFilter("low_pass_filter", is_fixed=False, latency_cycles=4)

  scheduler = Scheduler("scheduler", sample_queue, [low_pass_filter])

  data_uploader.connect(sample_queue)

  clock_unit = ClockUnit()
  clock_unit.subscribe_many([sample_queue, data_uploader, scheduler, low_pass_filter])

  visualization = DataVisualizer()

  while data_uploader.is_available():
    clock_unit.tick()

    if sample_queue.window_ready():
      # window = sample_queue.get_window()
      output = sample_queue.output_data

      # window[:36] to remove ghosting
      visualization.add_snapshot("Raw ECG", output[:36])

      output = low_pass_filter.output_data
      visualization.add_snapshot("Low Pass", output[:36])

      print(clock_unit)

  visualization.plot(title=f"Patient {patient_number} Signal Flow")

  print(data_uploader)
  print(sample_queue)