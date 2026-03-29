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

# TODO: Confirm that these are correct
def get_patient_bpm(patient_number):
  # https://www.researchgate.net/figure/Heart-rates-of-patients-from-the-MIT-BIH-Arrhythmia-Database_tbl4_371898998
  patient_bpms = {100: 76, 101: 62, 102: 73, 103: 70, 104: 77, 105: 90, 106: 70, 107: 71, 108: 61, 109: 85}
  return patient_bpms[patient_number]

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

  low_pass_unit = LowPassUnit("low_pass_unit", window_size=WINDOW_SIZE, vector_width=VECTOR_WIDTH)
  low_pass_unit.attach_recorder(low_pass_data_recorder)

  high_pass_unit = HighPassUnit("high_pass_unit", window_size=WINDOW_SIZE, vector_width=VECTOR_WIDTH)
  high_pass_unit.attach_recorder(high_pass_data_recorder)
  
  derivative_unit = DerivativeUnit("derivative_unit", window_size=WINDOW_SIZE, vector_width=VECTOR_WIDTH)
  derivative_unit.attach_recorder(derivative_data_recorder)
  
  squaring_unit = SquaringUnit("squaring_unit", window_size=WINDOW_SIZE, vector_width=VECTOR_WIDTH)
  squaring_unit.attach_recorder(squaring_data_recorder)

  mwi_unit = MWIUnit("mwi_unit", window_size=WINDOW_SIZE, vector_width=VECTOR_WIDTH, mwi_window_size=MWI_WINDOW_SIZE)
  mwi_unit.attach_recorder(mwi_data_recorder)

  threshold_unit = ThresholdUnit("threshold_unit", window_size=WINDOW_SIZE, sample_rate=SAMPLE_RATE)
  threshold_unit.attach_recorder(threshold_data_recorder)
  
  low_pass_unit.connect(high_pass_unit).connect(derivative_unit).connect(squaring_unit).connect(mwi_unit).connect(threshold_unit)

  scheduler = Scheduler("scheduler", sample_queue, lanes=[low_pass_unit])
  scheduler.attach_recorder(raw_ecg_data_recorder)

  units = [sample_queue, data_uploader, scheduler, low_pass_unit, high_pass_unit, derivative_unit, squaring_unit, mwi_unit, threshold_unit]
  clock_unit = ClockUnit()
  clock_unit.subscribe_many(units)

  # TODO: Should stop running when all samples are processed
  for _ in range(1000000):
    clock_unit.tick()

    if clock_unit.cycle % 100000 == 0:
      print(clock_unit)

  plot_data_recorders(data_recorders, patient_number=patient_number)

  print(f"Calculated BPM = {threshold_unit.get_bpm()}")
  print(f"Actual BPM = {get_patient_bpm(patient_number)}")

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

  # Confirm that the load_float_samples and load_fixed_samples load data as fixed and float.
  # The latency of the pipeline units (low pass, high pass, derivative, squaring, window, and threshold) should change based on fixed and floating point samples.
  # Confirm that the threshold unit is valid. It most likely is since we get the right BPM, but look it over nonetheless.
  # Confirm that the moving window integration unit is valid. It most likely is since we get the right BPM, but look it over nonetheless.
  # Confirm that each of the pipeline units is using some form of parallelization (if possible).
  # Remove the lane functionality from the scheduler (we only have 1 lane now).
  # There is no overlap between windows anymore. AI told me that this was neccessary, but from what I can tell, we are calculating the BPM correctly so I see no point in adding more complexity. Find a reason for this.
  # Look into how we can analyze throughput vs battery life.
  # It seems like the data being plotted is slightly offset from other pipeline stages. Confirm if this is ok.
  # Figure out the issue with the large number of stalled cycles.