class CircularBuffer:
  """
  Fixed-size circular buffer with pointer indexing.
  """
  def __init__(self, size: int, dtype=float):
    self.size = size
    self.dtype = dtype
    self.buffer = [dtype(0)] * size
    self.write_ptr = 0

  def push(self, value) -> None:
    self.buffer[self.write_ptr] = self.dtype(value)
    self.write_ptr = (self.write_ptr + 1) % self.size

  def __getitem__(self, index: int):
    actual = (self.write_ptr + index) % self.size
    return self.buffer[actual]

  def __repr__(self) -> str:
    return f"<CircularBuffer size={self.size} write_ptr={self.write_ptr}>"