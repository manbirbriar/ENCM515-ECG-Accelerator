from __future__ import annotations
import numpy as np
from fifo import InputFIFO
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from clock_unit import ClockUnit
  from mac_unit import MACUnit

PUSH_SAMPLE = 'PUSH_SAMPLE'

class DataUploader:
  """
  Models the ADC frontend. Schedules one sample push every
  CYCLES_PER_SAMPLE cycles, reflecting a hardware clock running
  faster than the 360Hz sample rate.
  """
  def __init__(self, name: str, samples: np.ndarray, cycles_per_sample: int, fifo: InputFIFO, mac: 'MACUnit'):
    self.name = name
    self.samples = samples
    self.cycles_per_sample = cycles_per_sample
    self.fifo = fifo
    self.mac = mac

    self.sample_index: int = 0
    self.total_samples: int = len(samples)
    self.active: bool = True

    # Performance tracking — stalled_cycles always 0, uploader never stalls
    self.busy_cycles: int = 0
    self.idle_cycles: int = 0
    self.stalled_cycles: int = 0

  def start(self, clock: 'ClockUnit') -> None:
    clock.schedule(self.cycles_per_sample, self, PUSH_SAMPLE)

  def handle_event(self, event: str, cycle: int, clock: 'ClockUnit') -> None:
    if event == PUSH_SAMPLE:
      if self.sample_index >= self.total_samples:
        self.active = False
        return

      self.fifo.push(self.samples[self.sample_index])
      self.sample_index += 1
      self.busy_cycles += 1

      # If MAC is idle notify it there is data available
      if self.mac.is_idle():
        clock.schedule(cycle, self.mac, 'SAMPLE_READY')

      # Schedule next sample push
      if self.sample_index < self.total_samples:
        clock.schedule(cycle + self.cycles_per_sample, self, PUSH_SAMPLE)
      else:
        self.active = False

  @property
  def utilization(self) -> float:
    total = self.busy_cycles + self.idle_cycles + self.stalled_cycles
    return self.busy_cycles / total if total > 0 else 0.0

  def __repr__(self) -> str:
    return (
      f"<DataUploader name={self.name} "
      f"index={self.sample_index}/{self.total_samples} "
      f"active={self.active}>"
    )
