import numpy as np
from hardware_unit import HardwareUnit
from config import (
  FIXED_POINT_SCALE,
  FIXED_ADD_CYCLES, FIXED_SUB_CYCLES, FIXED_MUL_CYCLES, FIXED_COMPARE_CYCLES,
  FLOAT_ADD_CYCLES, FLOAT_SUB_CYCLES, FLOAT_MUL_CYCLES, FLOAT_COMPARE_CYCLES,
  MIN_PEAK_WIDTH, REFRACTORY_SAMPLES,
)

class ThresholdUnit(HardwareUnit):
  """
  Adaptive threshold peak detector.

  Hardware blocks modeled:
    - Comparator:       sample > threshold
    - Counter:          refractory period tracking
    - Counter:          peak width (consecutive samples above threshold)
    - Shift register:   prev1, prev2 for slope-based peak detection
    - 2x Multiplier:    SPKI/NPKI adaptive update (0.125 * x)
    - Adder:            threshold = NPKI + 0.25*(SPKI - NPKI)
    - Register file:    SPKI, NPKI, threshold, last_peak_sample

  Peak is confirmed when ALL of:
    - slope was rising  (prev1 > prev2)
    - slope now falling (sample < prev1)
    - prev1 was above threshold
    - refractory period has elapsed
    - signal was above threshold for MIN_PEAK_WIDTH consecutive samples
  """
  def __init__(self, name: str, sample_rate: int, is_fixed_point: bool):

    # 1. Comparator: sample > threshold
    compare_cycles = FIXED_COMPARE_CYCLES if is_fixed_point else FLOAT_COMPARE_CYCLES

    # 2. Refractory check: subtraction + comparison
    refractory_cycles = (
      (FIXED_SUB_CYCLES + FIXED_COMPARE_CYCLES) if is_fixed_point
      else (FLOAT_SUB_CYCLES + FLOAT_COMPARE_CYCLES)
    )

    # 3. Adaptive update: 2x mul + add (SPKI or NPKI)
    adaptive_cycles = (
      (2*FIXED_MUL_CYCLES + FIXED_ADD_CYCLES) if is_fixed_point
      else (2*FLOAT_MUL_CYCLES + FLOAT_ADD_CYCLES)
    )

    # 4. Threshold update: sub + mul + add
    threshold_cycles = (
      (FIXED_SUB_CYCLES + FIXED_MUL_CYCLES + FIXED_ADD_CYCLES) if is_fixed_point
      else (FLOAT_SUB_CYCLES + FLOAT_MUL_CYCLES + FLOAT_ADD_CYCLES)
    )

    # 5. Sample count increment: add
    count_cycles = FIXED_ADD_CYCLES if is_fixed_point else FLOAT_ADD_CYCLES

    latency = compare_cycles + refractory_cycles + adaptive_cycles + threshold_cycles + count_cycles
    super().__init__(name, latency_cycles=latency, is_fixed_point=is_fixed_point)

    self.sample_rate = sample_rate

    # Scale initial estimates to fixed-point range if needed
    scale = FIXED_POINT_SCALE if is_fixed_point else 1.0
    self.spki      = 2.0 * scale
    self.npki      = 0.5 * scale
    self.threshold = 1.0 * scale

    # Peak detector state (shift register in hardware)
    self.prev1 = 0.0
    self.prev2 = 0.0
    self.above_threshold_count = 0

    # Detection tracking
    self.sample_count = 0
    self.last_peak_sample = -REFRACTORY_SAMPLES
    self.peaks = []

  def compute(self, sample) -> float:
    # --- Peak detector logic ---
    rising       = self.prev1 > self.prev2
    falling      = sample < self.prev1
    above        = self.prev1 > self.threshold
    refractory_ok = (self.sample_count - self.last_peak_sample) > REFRACTORY_SAMPLES

    # Track consecutive samples above threshold for noise rejection
    if sample > self.threshold:
      self.above_threshold_count += 1
    else:
      self.above_threshold_count = 0

    width_ok = self.above_threshold_count >= MIN_PEAK_WIDTH

    if rising and falling and above and refractory_ok and width_ok:
      # Confirmed peak — update signal estimate
      self.peaks.append(self.sample_count)
      self.last_peak_sample = self.sample_count
      self.spki = 0.125 * self.prev1 + 0.875 * self.spki
      detection_event = 1.0
    else:
      # Not a peak — update noise estimate
      self.npki = 0.125 * sample + 0.875 * self.npki
      detection_event = 0.0

    # Update adaptive threshold
    self.threshold = self.npki + 0.25 * (self.spki - self.npki)

    # Advance internal sample clock and shift peak detector registers
    self.sample_count += 1
    self.prev2 = self.prev1
    self.prev1 = sample

    return detection_event

  def get_bpm(self) -> float:
    if len(self.peaks) < 2:
      return 0.0
    intervals = np.diff(self.peaks)
    avg_interval_samples = np.mean(intervals)
    return (60.0 * self.sample_rate) / avg_interval_samples
