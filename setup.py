import wfdb
import numpy as np
from clock_unit import ClockUnit
from sample_queue import SampleQueue
from data_uploader import DataUploader
from data_visualizer import DataVisualizer
from scheduler import Scheduler
from derivative_unit import DerivativeUnit
from squaring_unit import SquaringUnit
from result_sink import ResultSink

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

# helper function to stitch all saved windows together (flattening windows into continuous traces)
def stitch_windows(windows: list[list[float]], hop_size: int) -> list[float]:
  if not windows:
    return []

  stitched = list(windows[0])
  for window in windows[1:]:
    stitched.extend(window[hop_size:])
  return stitched

# TODO: Confirm that this is working correctly
if __name__ == "__main__":
  patient_number = input("Enter the patient number: ")

  record_path = f"ecg_data/patient_{patient_number}/{patient_number}"

  float_samples = load_float_samples(record_path)
  fixed_samples = load_fixed_samples(record_path)

  print(f"Float samples: dtype: {float_samples.dtype}, shape: {float_samples.shape}, range: [{float_samples.min()}, {float_samples.max()}]")
  print(f"Fixed samples: dtype: {fixed_samples.dtype}, shape: {fixed_samples.shape}, range: [{fixed_samples.min()}, {fixed_samples.max()}]")

  sample_queue = SampleQueue("sample_queue", queue_size=216, window_size=72, hop_size=36)
  data_uploader = DataUploader("data_uploader", float_samples)
  data_uploader.connect(sample_queue)


  derivative_unit = DerivativeUnit("derivative_unit", latency_cycles=2)
  squaring_unit = SquaringUnit("squaring_unit", latency_cycles=1)
  sink = ResultSink("sink")

  derivative_unit2 = DerivativeUnit("derivative_unit", latency_cycles=2)
  squaring_unit2 = SquaringUnit("squaring_unit", latency_cycles=1)
  sink2 = ResultSink("sink")

  derivative_unit3 = DerivativeUnit("derivative_unit", latency_cycles=2)
  squaring_unit3 = SquaringUnit("squaring_unit", latency_cycles=1)
  sink3 = ResultSink("sink")

  derivative_unit4 = DerivativeUnit("derivative_unit", latency_cycles=2)
  squaring_unit4 = SquaringUnit("squaring_unit", latency_cycles=1)
  sink4 = ResultSink("sink")

  derivative_unit.connect(squaring_unit).connect(sink)
  derivative_unit2.connect(squaring_unit2).connect(sink2)
  derivative_unit3.connect(squaring_unit3).connect(sink3)
  derivative_unit4.connect(squaring_unit4).connect(sink4)

  scheduler = Scheduler("scheduler", sample_queue, lanes=[derivative_unit, derivative_unit2, derivative_unit3, derivative_unit4])

  clock_unit = ClockUnit()
  clock_unit.subscribe_many([sample_queue, data_uploader, scheduler, derivative_unit, squaring_unit, sink])

  while data_uploader.is_available() or sample_queue.window_ready() or not derivative_unit.is_available() or not squaring_unit.is_available():
    clock_unit.tick()

    # if sample_queue.window_ready():
    #   window = sample_queue.get_window()
    #   # window[:36] to remove ghosting
    #   visualization.add_snapshot("Raw ECG", window[:36])
    if clock_unit.cycle % 10000 == 0:
      print(clock_unit)
  
  visualization = DataVisualizer()
  
  raw_trace = stitch_windows(scheduler.dispatched_windows, sample_queue.hop_size)
  derivative_trace = stitch_windows(derivative_unit.history, sample_queue.hop_size)
  squared_trace = stitch_windows(squaring_unit.history, sample_queue.hop_size)

  visualization.add_snapshot("Raw ECG", raw_trace)
  visualization.add_snapshot("Derivative", derivative_trace)
  visualization.add_snapshot("Squared", squared_trace)

  print(f"Processed windows: {len(sink.results)}")

  visualization.plot(title=f"Patient {patient_number} Signal Flow")

  print(clock_unit)
  print(data_uploader)
  print(sample_queue)
  print(scheduler)
  print(derivative_unit)
  print(squaring_unit)
  print(sink)