from hardware_unit import HardwareUnit
from fifo import InputFIFO
import numpy as np

class DataUploader(HardwareUnit):
  """
  Models the ADC frontend of the ECG system.
  Produces one sample every CYCLES_PER_SAMPLE clock cycles,
  reflecting the fact that the hardware clock runs faster than the sample rate.
  Pushes samples into the InputFIFO rather than directly into the processing pipeline.
  """
  def __init__(self, name: str, samples: np.ndarray, cycles_per_sample: int, fifo: InputFIFO):
    super().__init__(name, latency_cycles=1)

    self.samples = samples
    self.cycles_per_sample = cycles_per_sample
    self.fifo = fifo

    self.sample_index: int = 0
    self.total_samples: int = len(self.samples)
    self.active: bool = True

  def tick(self, current_cycle: int) -> None:
    if not self.active:
      self.idle_cycles += 1
      return

    if self.sample_index >= self.total_samples:
      self.active = False
      return

    # Only push a sample every cycles_per_sample cycles
    # This models the ADC running at 360Hz while the hardware clock runs faster
    if current_cycle % self.cycles_per_sample != 0:
      self.idle_cycles += 1
      return

    self.fifo.push(self.samples[self.sample_index])
    self.sample_index += 1
    self.busy_cycles += 1

  # Unused
  def compute(self, data):
    return data

  def is_available(self) -> bool:
    return self.active

  def is_done(self) -> bool:
    return self.sample_index >= self.total_samples

  def __repr__(self) -> str:
    return (
      f"<DataUploader name={self.name} "
      f"index={self.sample_index}/{self.total_samples} "
      f"active={self.active}>"
    )
