import numpy as np
from hardware_unit import HardwareUnit
from config import (
  FIXED_POINT_SCALE,
  FIXED_ADD_CYCLES, FIXED_SUB_CYCLES, FIXED_MUL_CYCLES, FIXED_COMPARE_CYCLES,
  FLOAT_ADD_CYCLES, FLOAT_SUB_CYCLES, FLOAT_MUL_CYCLES, FLOAT_COMPARE_CYCLES,
  REFRACTORY_SAMPLES,
)

class ThresholdUnit(HardwareUnit):
  """
  Adaptive threshold peak detector implementing Pan-Tompkins detection.

  Detection strategy:
    A peak is confirmed on the falling edge (sample < prev1) when:
      - prev1 was the local maximum above threshold
      - refractory period has elapsed since last peak

  The slope-based falling edge detector is used instead of a windowed max
  since the MWI output is already smoothed — we just need to find when it
  starts descending after crossing threshold.

  Adaptive threshold updates SPKI (signal) and NPKI (noise) estimates
  using exponential weighting (1/8 new, 7/8 old).

  Hardware blocks:
    Comparator:      sample > threshold, sample < prev1
    Counter:         refractory period
    Shift register:  prev1 for slope detection
    2x Multiplier:   SPKI/NPKI adaptive update
    Adder:           threshold update
    Register file:   SPKI, NPKI, threshold, last_peak_sample
  """
  def __init__(self, name: str, sample_rate: int, is_fixed_point: bool):
    compare_cycles   = FIXED_COMPARE_CYCLES if is_fixed_point else FLOAT_COMPARE_CYCLES
    refractory_cycles = (FIXED_SUB_CYCLES + FIXED_COMPARE_CYCLES) if is_fixed_point else (FLOAT_SUB_CYCLES + FLOAT_COMPARE_CYCLES)
    adaptive_cycles  = (2*FIXED_MUL_CYCLES + FIXED_ADD_CYCLES) if is_fixed_point else (2*FLOAT_MUL_CYCLES + FLOAT_ADD_CYCLES)
    threshold_cycles = (FIXED_SUB_CYCLES + FIXED_MUL_CYCLES + FIXED_ADD_CYCLES) if is_fixed_point else (FLOAT_SUB_CYCLES + FLOAT_MUL_CYCLES + FLOAT_ADD_CYCLES)
    count_cycles     = FIXED_ADD_CYCLES if is_fixed_point else FLOAT_ADD_CYCLES

    latency = compare_cycles + refractory_cycles + adaptive_cycles + threshold_cycles + count_cycles
    super().__init__(name, latency_cycles=latency, is_fixed_point=is_fixed_point)

    self.sample_rate = sample_rate

    scale = FIXED_POINT_SCALE if is_fixed_point else 1.0
    self.spki      = 0.0
    self.npki      = 0.0
    self.threshold = 0.0

    # Peak detector state
    self.prev1 = 0.0   # previous sample (shift register)
    self.above_threshold = False  # was prev1 above threshold

    self.sample_count = 0
    self.last_peak_sample = -REFRACTORY_SAMPLES
    self.peaks = []

  def compute(self, sample) -> float:
    refractory_ok = (self.sample_count - self.last_peak_sample) > REFRACTORY_SAMPLES

    # Falling edge: sample dropped below prev1, and prev1 was above threshold
    falling = sample < self.prev1
    is_peak = falling and self.above_threshold and refractory_ok

    if is_peak:
      self.peaks.append(self.sample_count)
      self.last_peak_sample = self.sample_count
      # Update signal peak estimate
      self.spki = 0.125 * self.prev1 + 0.875 * self.spki
      detection_event = 1.0
    else:
      # Update noise estimate
      self.npki = 0.125 * sample + 0.875 * self.npki
      detection_event = 0.0

    # Update adaptive threshold
    # On first samples use signal directly to initialise
    if self.spki == 0.0 and self.npki == 0.0:
      self.threshold = sample * 0.5
    else:
      self.threshold = self.npki + 0.25 * (self.spki - self.npki)

    # Update state for next sample
    self.above_threshold = sample > self.threshold
    self.prev1 = sample
    self.sample_count += 1

    return detection_event

  def get_bpm(self) -> float:
    if len(self.peaks) < 2:
      return 0.0
    intervals = np.diff(self.peaks)
    return (60.0 * self.sample_rate) / np.mean(intervals)
