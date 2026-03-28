from collections import deque

class DataRecorder:
  def __init__(self, name: str, capacity: int, hop_size: int):
    self.name = name
    self.capacity = capacity
    self.hop_size = hop_size
    # automatically removes oldest sample when capacity is reached
    self.buffer: deque = deque(maxlen=capacity)

  def record(self, window: list) -> None:
    # only commit the first non-overlapping samples
    for sample in window[:self.hop_size]:
      self.buffer.append(sample)

  def get_signal(self) -> list:
    return list(self.buffer)