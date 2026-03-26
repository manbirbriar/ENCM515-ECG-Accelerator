from hardware_unit import HardwareUnit
import numpy as np

# Loads the entire MIT-BIH record into memory on initialisation, then on each tick pushes exactly one sample into the connected SampleQueue
# MIT-BIH is sampled at 360 Hz. Thus, one sample per tick means each clock cycle represents a sample period of ~2.78ms
# When all samples have been sent, the uploader becomes inactive
# samples will be in either fixed or floating point format
class DataUploader(HardwareUnit):
  def __init__(self, name: str, samples: np.ndarray):
    super().__init__(name, latency_cycles=1)

    self.samples = samples
    self.sample_index: int = 0
    self.total_samples: int = len(self.samples)
    self.active: bool = True

  # Push one sample per tick into the SampleQueue
  def tick(self, current_cycle: int) -> None:
    self.current_cycle = current_cycle

    if not self.active:
      return

    if self.sample_index >= self.total_samples:
      self.active = False
      return

    if not self.output_data:
      self.output_data = self.samples[self.sample_index]
      self.sample_index += 1

    self.push_output()
  
  def push_output(self) -> None:
    if self.next_unit and self.next_unit.is_available():
      self.next_unit.input_data = self.output_data
      self.output_data = []

  # Unused
  def compute(self, data: list) -> list:
    return data

  def is_available(self) -> bool:
    return self.active

  def __repr__(self) -> str:
    return f"<DataUploader name={self.name} index={self.sample_index}/{self.total_samples} active={self.active}>"