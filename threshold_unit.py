import numpy as np
from hardware_unit import HardwareUnit
from config import FIXED_POINT_SCALE

class ThresholdUnit(HardwareUnit):
  def __init__(self, name: str, window_size: int, sample_rate: int, is_fixed_point: bool):
    super().__init__(name, latency_cycles=1, is_fixed_point=is_fixed_point)
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