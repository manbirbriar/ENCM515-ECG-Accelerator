from __future__ import annotations

import numpy as np

from hardware_unit import HardwareUnit, SampleToken


class DataUploader(HardwareUnit):
  def __init__(self, name: str, samples: np.ndarray, sample_rate_hz: int, accel_clock_hz: int):
    super().__init__(name, latency_cycles=1, initiation_interval=1)
    self.samples = samples
    self.sample_rate_hz = sample_rate_hz
    self.accel_clock_hz = accel_clock_hz
    self.sample_index = 0
    self.total_samples = len(samples)
    self.active = True
    self.dropped_samples = 0

  def next_arrival_cycle(self) -> int | None:
    if self.sample_index >= self.total_samples:
      return None

    numerator = (self.sample_index + 1) * self.accel_clock_hz
    return (numerator + self.sample_rate_hz - 1) // self.sample_rate_hz

  def tick(self, current_cycle: int) -> None:
    if not self.active:
      return

    if self.sample_index >= self.total_samples:
      self.active = False
      return

    if self.next_unit is None:
      self.active = False
      return

    next_cycle = self.next_arrival_cycle()
    while next_cycle is not None and next_cycle <= current_cycle and self.sample_index < self.total_samples:
      sample_value = self.samples[self.sample_index]
      token = SampleToken(
        sample_id=self.sample_index,
        value=sample_value.item() if hasattr(sample_value, "item") else sample_value,
        ingress_cycle=current_cycle,
      )

      if self.next_unit.accept(token, current_cycle):
        if self.recorder is not None:
          self.recorder.record_token(token, current_cycle)
        self.accepted_count += 1
      else:
        self.dropped_samples += 1
        self.stalled_cycles += 1

      self.sample_index += 1
      next_cycle = self.next_arrival_cycle()

    if self.sample_index >= self.total_samples:
      self.active = False

  def process_token(self, token: SampleToken, current_cycle: int) -> SampleToken:
    return token

  def is_done(self) -> bool:
    return self.sample_index >= self.total_samples

  def can_accept(self, current_cycle: int) -> bool:
    return False
