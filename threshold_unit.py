import numpy as np
from hardware_unit import HardwareUnit

# TODO: This is temporary (AI)
class ThresholdUnit(HardwareUnit):
  def __init__(self, name: str, sample_rate: int = 360):
    super().__init__(name, latency_cycles=1)
    self.sample_rate = sample_rate
    
    # --- PRE-SEED VALUES ---
    # Prevents 'Cold Start' where threshold stays at 0
    self.spki = 2.0  # Signal Peak Estimate
    self.npki = 0.5  # Noise Peak Estimate
    self.threshold = 1.0 
    
    # --- INTERNAL TIMELINE ---
    self.sample_count = 0 
    self.peaks = []  # Stores the sample_count of each detection
    self.last_peak_sample = -500 # Ensure we can detect the very first beat

  def compute(self, data: list) -> list:
    # 1. Increment our internal timeline
    # 'data' is usually your HOP_SIZE or WINDOW_SIZE
    num_samples = len(data)
    current_max = np.max(data)
    
    # 2. Check for Heartbeat
    # 360Hz * 0.2s = 72 samples (Refractory Period/Cooldown)
    if current_max > self.threshold and (self.sample_count - self.last_peak_sample) > 72:
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
    """Calculates BPM based on internal peak list."""
    if len(self.peaks) < 2:
      return 0.0
    
    # Calculate distance between peaks in samples
    intervals = np.diff(self.peaks)
    avg_interval_samples = np.mean(intervals)
    
    # Math: (60 sec / (avg_samples / samples_per_sec))
    return (60.0 * self.sample_rate) / avg_interval_samples