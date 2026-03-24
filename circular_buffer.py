from hardware_unit import HardwareUnit
import numpy as np

class CircularBuffer(HardwareUnit):
  # MIT-BIH dataset sample rate is 360Hz
  # TODO: I've gone with a 0.2s wide window, but I cannot justify it
  # window_size = 360 samples/s * 0.2s = 72 samples
  # TODO: Not sure how large the buffer_size should be
  def __init__(self, name: str, buffer_size: int, window_size: int = 72, hop_size: int = 36):
    super().__init__(name, latency_cycles=1)

    self.buffer_size: int = buffer_size
    self.window_size: int = window_size
    self.hop_size: int = hop_size

    self.buffer: np.ndarray = np.zeros(buffer_size, dtype=np.float64)
    self.write_ptr: int = 0
    self.sample_count: int = 0
    self.stalled: bool = False

  # Called by DataUploader each tick to write one sample into the buffer
  # If the buffer is full, the sample is dropped and stalled is set
  def receive_sample(self, sample: float) -> None:
    if self.sample_count >= self.buffer_size:
      self.stalled = True
      return

    self.buffer[self.write_ptr] = sample
    self.write_ptr = (self.write_ptr + 1) % self.buffer_size
    self.sample_count += 1
    self.stalled = False

  # True when enough samples have accumulated to dispatch a full window
  def window_ready(self) -> bool:
    return self.sample_count >= self.window_size

  # Extract the oldest window_size samples from the buffer
  def get_window(self) -> list:
    if not self.window_ready():
      return []

    read_ptr = (self.write_ptr - self.sample_count) % self.buffer_size

    window = []
    for i in range(self.window_size):
      window.append(float(self.buffer[(read_ptr + i) % self.buffer_size]))

    # Advance by 36 samples (50% overlap) to ensure R-peaks falling on window boundaries are fully captured in the subsequent frame
    # TODO: I'm not 100% sure on this, but I am convincing myself that it makes sense
    self.sample_count -= self.hop_size
    return window

  # CircularBuffer has no autonomous work to do each cycle
  # Tick is implemented only to satisfy ClockUnit registration and to clear the stall flag if space has freed up
  def tick(self, current_cycle: int) -> None:
    self.current_cycle = current_cycle
    if self.sample_count < self.buffer_size:
      self.stalled = False

  # Unused
  def compute(self, data: list) -> list:
    return data

  # True when the buffer has room for at least one more sample
  def is_available(self) -> bool:
    return self.sample_count < self.buffer_size

  def is_stalled(self) -> bool:
    return self.stalled

  def __repr__(self) -> str:
    return f"<CircularBuffer name={self.name} count={self.sample_count}/{self.buffer_size} window_ready={self.window_ready()} stalled={self.stalled}>"