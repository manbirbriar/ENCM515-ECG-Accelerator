from collections import deque

class DataRecorder:
  def __init__(self, name: str, capacity: int):
    self.name = name
    self.capacity = capacity
    # Automatically removes oldest sample when capacity is reached
    self.buffer: deque = deque(maxlen=capacity)

  # Record a single scalar sample
  def record(self, sample) -> None:
    self.buffer.append(sample)

  def get_signal(self) -> list:
    return list(self.buffer)