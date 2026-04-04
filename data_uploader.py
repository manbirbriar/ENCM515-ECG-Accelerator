# from hardware_unit import HardwareUnit
# from fifo import InputFIFO
# import numpy as np

# class DataUploader(HardwareUnit):
#   """
#   Models the ADC frontend of the ECG system.
#   Produces one sample every CYCLES_PER_SAMPLE clock cycles,
#   reflecting the fact that the hardware clock runs faster than the sample rate.
#   Pushes samples into the InputFIFO rather than directly into the processing pipeline.
#   """
#   def __init__(self, name: str, samples: np.ndarray, cycles_per_sample: int, fifo: InputFIFO):
#     super().__init__(name, latency_cycles=1)

#     self.samples = samples
#     self.cycles_per_sample = cycles_per_sample
#     self.fifo = fifo

#     self.sample_index: int = 0
#     self.total_samples: int = len(self.samples)
#     self.active: bool = True

#   def tick(self, current_cycle: int) -> None:
#     if not self.active:
#       self.idle_cycles += 1
#       return

#     if self.sample_index >= self.total_samples:
#       self.active = False
#       return

#     # Only push a sample every cycles_per_sample cycles
#     # This models the ADC running at 360Hz while the hardware clock runs faster
#     if current_cycle % self.cycles_per_sample != 0:
#       self.idle_cycles += 1
#       return

#     self.fifo.push(self.samples[self.sample_index])
#     self.sample_index += 1
#     self.busy_cycles += 1

#   # Unused
#   def compute(self, data):
#     return data

#   def is_available(self) -> bool:
#     return self.active

#   def is_done(self) -> bool:
#     return self.sample_index >= self.total_samples

#   def __repr__(self) -> str:
#     return (
#       f"<DataUploader name={self.name} "
#       f"index={self.sample_index}/{self.total_samples} "
#       f"active={self.active}>"
#     )

from hardware_unit import HardwareUnit
from fifo import InputFIFO
import numpy as np

class DataUploader(HardwareUnit):
  """
  Models the ADC frontend of the ECG system.
  Produces one sample every CYCLES_PER_SAMPLE clock cycles,
  reflecting the fact that the hardware clock runs faster than the sample rate.
  Pushes samples into the InputFIFO rather than directly into the processing pipeline.

  phase_offset: cycle offset for this ADC channel.
  In real hardware, two ADC channels on the same chip are never perfectly
  synchronized — they drift slightly relative to each other. A phase_offset
  of 1 means this channel samples 1 cycle later than channel 0, modeling
  realistic ADC clock skew between leads.
  """
  def __init__(self, name: str, samples: np.ndarray, cycles_per_sample: int, fifo: InputFIFO, phase_offset: int = 0):
    super().__init__(name, latency_cycles=1)

    self.samples = samples
    self.cycles_per_sample = cycles_per_sample
    self.fifo = fifo
    self.phase_offset = phase_offset

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

    # Only push a sample every cycles_per_sample cycles, offset by phase_offset
    # phase_offset models ADC clock skew between leads — in real hardware two
    # ADC channels are never perfectly synchronized on the same cycle
    if (current_cycle - self.phase_offset) % self.cycles_per_sample != 0:
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
      f"active={self.active} "
      f"phase_offset={self.phase_offset}>"
    )