import numpy as np
from config import FIXED_ADD_CYCLES, FIXED_COMPARE_CYCLES, FIXED_MAC_CYCLES, FIXED_POINT_SCALE, FIXED_SUB_CYCLES, FLOAT_ADD_CYCLES, FLOAT_COMPARE_CYCLES, FLOAT_MAC_CYCLES, FLOAT_SUB_CYCLES, MIN_PEAK_WIDTH, REFRACTORY_SAMPLES

from hardware_unit import HardwareUnit

class ThresholdUnit(HardwareUnit):
  """
  https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4122029
  
  Adaptive Threshold Peak Detector

  A peak is accepted only if all of the following are true:
    - Slope was rising
    - Slope just fell
    - Prev1 above Threshold
    - Refractory period elapsed
    - Signal above threshold for MIN_PEAK_WIDTH

  Hardware: 1 ALU, 1 MAC, 8 Registers (SPKI, NPKI, Threshold, Prev1, Prev2, LastPeakSample, SampleCount, AboveThresholdCount)
  """
  def __init__(self, name: str, sample_rate: int, is_fixed_point: bool):
    if is_fixed_point:
      # Slope and threshold comparisons: 3 CMP operations
      slope_threshold_compare = 3 * FIXED_COMPARE_CYCLES
      # Refractory check: 1 SUB + 1 CMP operation
      refractory_check = FIXED_SUB_CYCLES + FIXED_COMPARE_CYCLES
      # Width counter update: 1 CMP + 1 ADD operation
      width_counter_update = FIXED_COMPARE_CYCLES + FIXED_ADD_CYCLES
      # SPKI or NPKI adaptive update: 2 MAC operations (worst case)
      spki_npki_adaptive_update = 2 * FIXED_MAC_CYCLES
      # Threshold recompute: 1 SUB + 1 MAC operation
      threshold_recompute = FIXED_SUB_CYCLES + FIXED_MAC_CYCLES
      # Register updates: 1 ADD operation
      register_update = FIXED_ADD_CYCLES
    else:
      # Slope and threshold comparisons: 3 CMP operations
      slope_threshold_compare = 3 * FLOAT_COMPARE_CYCLES
      # Refractory check: 1 SUB and 1 CMP operation
      refractory_check = FLOAT_SUB_CYCLES + FLOAT_COMPARE_CYCLES
      # Width counter update: 1 CMP and 1 ADD operation
      width_counter_update = FLOAT_COMPARE_CYCLES + FLOAT_ADD_CYCLES
      # SPKI or NPKI adaptive update: 2 MAC operations (worst case)
      spki_npki_adaptive_update = 2 * FLOAT_MAC_CYCLES
      # Threshold recompute: 1 SUB and 1 MAC operation
      threshold_recompute = FLOAT_SUB_CYCLES + FLOAT_MAC_CYCLES
      # Register updates: 1 ADD operation
      register_update = FLOAT_ADD_CYCLES

    latency = slope_threshold_compare + refractory_check + width_counter_update + \
      spki_npki_adaptive_update + threshold_recompute + register_update
    
    super().__init__(name, latency_cycles=latency, is_fixed_point=is_fixed_point)

    self.sample_rate = sample_rate

    scale = FIXED_POINT_SCALE if is_fixed_point else 1.0

    # Registers
    self.spki = 2.0 * scale
    self.npki = 0.5 * scale
    self.threshold = 1.0 * scale

    self.prev1 = 0.0
    self.prev2 = 0.0
    
    self.sample_count = 0
    self.above_threshold_count = 0
    self.last_peak_sample = -REFRACTORY_SAMPLES

    self.peaks = []

  def compute(self, sample) -> float:
    # Slope and Threshold Comparisons
    rising = self.prev1 > self.prev2
    falling = sample < self.prev1
    local_max = rising and falling
    above = self.prev1 > self.threshold

    # Refractory Check
    refractory_ok = (self.sample_count - self.last_peak_sample) > REFRACTORY_SAMPLES

    # Width Counter Update
    if sample > self.threshold:
      self.above_threshold_count += 1
    else:
      self.above_threshold_count = 0
    width_ok = self.above_threshold_count >= MIN_PEAK_WIDTH

    # Adaptive SPKI or NPKI Update
    if local_max and above and refractory_ok and width_ok:
      self.peaks.append(self.sample_count)
      self.last_peak_sample = self.sample_count
      self.spki = 0.125 * self.prev1 + 0.875 * self.spki
      detection_event = 1.0
    elif local_max:
      self.npki = 0.125 * self.prev1 + 0.875 * self.npki
      detection_event = 0.0
    else:
      detection_event = 0.0

    # Threshold Recompute
    self.threshold = self.npki + 0.25 * (self.spki - self.npki)

    # Register Update
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