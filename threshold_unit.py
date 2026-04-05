import numpy as np
from hardware_unit import HardwareUnit
from config import ( FIXED_POINT_SCALE,FIXED_ADD_CYCLES, FIXED_SUB_CYCLES, FIXED_MUL_CYCLES, FIXED_COMPARE_CYCLES,
  FLOAT_ADD_CYCLES, FLOAT_SUB_CYCLES, FLOAT_MUL_CYCLES, FLOAT_COMPARE_CYCLES,
  MIN_PEAK_WIDTH, REFRACTORY_SAMPLES,
)

class ThresholdUnit(HardwareUnit):
  """
  Adaptive threshold peak detector

  Hardware-style pieces this model keeps track of:
    - Comparator: checks whether sample > threshold
    - Refractory counter: enforces a minimum gap between peaks
    - Width counter: counts how long the signal stays above threshold
    - Shift register: keeps prev1 and prev2 for slope checks
    - Multipliers/adders: update SPKI, NPKI, and threshold
    - Registers: store SPKI, NPKI, threshold, and last_peak_sample

  A peak is accepted only if all checks pass:
    - slope was rising (prev1 > prev2)
    - slope just turned downward (sample < prev1)
    - prev1 is above the current threshold
    - refractory period has passed
    - signal stayed above threshold for at least MIN_PEAK_WIDTH samples

  NPKI follows the original Pan-Tompkins idea:
    - update NPKI only at local maxima that are not real peaks
    - this avoids driving NPKI toward zero during flat regions
  """
  def __init__(self, name: str, sample_rate: int, is_fixed_point: bool):

    add_cycles = FIXED_ADD_CYCLES if is_fixed_point else FLOAT_ADD_CYCLES
    sub_cycles = FIXED_SUB_CYCLES if is_fixed_point else FLOAT_SUB_CYCLES
    mul_cycles = FIXED_MUL_CYCLES if is_fixed_point else FLOAT_MUL_CYCLES
    compare_cycles = FIXED_COMPARE_CYCLES if is_fixed_point else FLOAT_COMPARE_CYCLES

    # Build latency from the five logical stages of this unit
    refractory_cycles = sub_cycles + compare_cycles
    adaptive_cycles = 2 * mul_cycles + add_cycles
    threshold_cycles = sub_cycles + mul_cycles + add_cycles
    count_cycles = add_cycles

    latency = compare_cycles + refractory_cycles + adaptive_cycles + threshold_cycles + count_cycles
    super().__init__(name, latency_cycles=latency, is_fixed_point=is_fixed_point)

    self.sample_rate = sample_rate

    # In fixed-point mode, scale initial estimates into integer range
    if is_fixed_point:
      scale = FIXED_POINT_SCALE
    else:
      scale = 1.0

    self.spki = 2.0 * scale
    self.npki = 0.5 * scale
    self.threshold = 1.0 * scale

    self.prev1 = 0.0
    self.prev2 = 0.0
    self.above_threshold_count = 0

    self.sample_count = 0
    self.last_peak_sample = -REFRACTORY_SAMPLES
    self.peaks = []

  def compute(self, sample) -> float:
    # Decide whether prev1 is a valid local peak candidate
    rising = self.prev1 > self.prev2
    falling = sample < self.prev1
    local_max = rising and falling   # prev1 sits at a local maximum
    above = self.prev1 > self.threshold
    refractory_ok = (self.sample_count - self.last_peak_sample) > REFRACTORY_SAMPLES

    # Require a short run above threshold to reduce spurious detections
    if sample > self.threshold:
      self.above_threshold_count += 1
    else:
      self.above_threshold_count = 0

    width_ok = self.above_threshold_count >= MIN_PEAK_WIDTH

    if local_max and above and refractory_ok and width_ok:
      # Real QRS detection: update signal estimate using the peak value
      self.peaks.append(self.sample_count)
      self.last_peak_sample = self.sample_count
      self.spki = 0.125 * self.prev1 + 0.875 * self.spki
      detection_event = 1.0
    elif local_max:
      # Not a QRS peak, so treat it as noise and update NPKI
      # We only do this on local maxima (using prev1), not every sample,
      # which keeps NPKI from collapsing between beats
      self.npki = 0.125 * self.prev1 + 0.875 * self.npki
      detection_event = 0.0
    else:
      detection_event = 0.0

    # Recompute the adaptive threshold from SPKI and NPKI
    self.threshold = self.npki + 0.25 * (self.spki - self.npki)

    # Move the sample clock forward and shift history values
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
