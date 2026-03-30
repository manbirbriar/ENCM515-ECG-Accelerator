import numpy as np
from hardware_unit import HardwareUnit
from config import (
  FIXED_POINT_SCALE,
  FIXED_ADD_CYCLES,
  FIXED_SUB_CYCLES,
  FIXED_MUL_CYCLES,
  FIXED_COMPARE_CYCLES,
  FLOAT_ADD_CYCLES,
  FLOAT_SUB_CYCLES,
  FLOAT_MUL_CYCLES,
  FLOAT_COMPARE_CYCLES,
)

class ThresholdUnit(HardwareUnit):
  def __init__(self, name: str, window_size: int, sample_rate: int, is_fixed_point: bool):
    # 1) Peak candidate extraction via max(data): model as a compare reduction.
    max_reduce_float_cycles = max(window_size - 1, 0) * FLOAT_COMPARE_CYCLES
    max_reduce_fixed_cycles = max(window_size - 1, 0) * FIXED_COMPARE_CYCLES

    # 2) Gating condition: current_max > threshold and refractory-period check.
    gate_float_cycles = FLOAT_COMPARE_CYCLES + FLOAT_SUB_CYCLES + FLOAT_COMPARE_CYCLES
    gate_fixed_cycles = FIXED_COMPARE_CYCLES + FIXED_SUB_CYCLES + FIXED_COMPARE_CYCLES

    # 3) Adaptive SPKI/NPKI update (one branch executes each window):
    # value = alpha * current_max + beta * previous
    adaptive_update_float_cycles = (2 * FLOAT_MUL_CYCLES) + FLOAT_ADD_CYCLES
    adaptive_update_fixed_cycles = (2 * FIXED_MUL_CYCLES) + FIXED_ADD_CYCLES

    # 4) Threshold update: npki + 0.25 * (spki - npki)
    threshold_update_float_cycles = FLOAT_SUB_CYCLES + FLOAT_MUL_CYCLES + FLOAT_ADD_CYCLES
    threshold_update_fixed_cycles = FIXED_SUB_CYCLES + FIXED_MUL_CYCLES + FIXED_ADD_CYCLES

    # 5) sample_count += num_samples
    sample_count_update_float_cycles = FLOAT_ADD_CYCLES
    sample_count_update_fixed_cycles = FIXED_ADD_CYCLES

    float_latency = (
      max_reduce_float_cycles
      + gate_float_cycles
      + adaptive_update_float_cycles
      + threshold_update_float_cycles
      + sample_count_update_float_cycles
    )
    fixed_latency = (
      max_reduce_fixed_cycles
      + gate_fixed_cycles
      + adaptive_update_fixed_cycles
      + threshold_update_fixed_cycles
      + sample_count_update_fixed_cycles
    )

    if is_fixed_point:
      super().__init__(name, latency_cycles=fixed_latency, is_fixed_point=is_fixed_point)
    else:
      super().__init__(name, latency_cycles=float_latency, is_fixed_point=is_fixed_point)
    self.window_size = window_size
    self.sample_rate = sample_rate

    scale = FIXED_POINT_SCALE if self.is_fixed_point else 1.0
    
    self.spki = 2.0 * scale
    self.npki = 0.5 * scale
    self.threshold = 1.0 * scale
    
    self.sample_count = 0
    self.peaks = []  # Stores the sample_count of each detection
    self.last_peak_sample = -500 # Ensure we can detect the very first beat

  def compute(self, data: list) -> list:
    num_samples = len(data)
    current_max = np.max(data)
    
    # 360Hz * 0.2s = 72 samples (Refractory Period/Cooldown)
    if current_max > self.threshold and (self.sample_count - self.last_peak_sample) > self.window_size:
      self.peaks.append(self.sample_count)
      self.last_peak_sample = self.sample_count
      
      # Update Signal Estimate (1/8th weighting)
      self.spki = 0.125 * current_max + 0.875 * self.spki
      detection_event = 1.0
    else:
      # Update Noise Estimate
      self.npki = 0.125 * current_max + 0.875 * self.npki
      detection_event = 0.0

    # 3. Update Adaptive Threshold
    self.threshold = self.npki + 0.25 * (self.spki - self.npki)
    
    # 4. Advance our internal 'clock'
    self.sample_count += num_samples
    
    # Return a list for the recorder
    return [detection_event] * num_samples

  def get_bpm(self) -> float:
    if len(self.peaks) < 2:
      return 0.0
    
    # Calculate distance between peaks in samples
    intervals = np.diff(self.peaks)
    avg_interval_samples = np.mean(intervals)
    
    # Math: (60 sec / (avg_samples / samples_per_sec))
    return (60.0 * self.sample_rate) / avg_interval_samples