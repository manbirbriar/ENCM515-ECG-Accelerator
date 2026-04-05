from collections import deque

class DataRecorder:
  def __init__(self, name: str, capacity: int):
    self.name = name
    self.capacity = capacity
    # Removes oldest sample when capacity is reached
    self.buffer: deque = deque(maxlen=capacity)

  # Records a single sample
  def record(self, sample) -> None:
    self.buffer.append(sample)

  def get_signal(self) -> list:
    return list(self.buffer)