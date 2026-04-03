class CircularBuffer:
  """
  Fixed-size circular buffer with pointer-based indexing.
  Supports negative indexing: buffer[-1] = newest, buffer[-N] = Nth oldest from end.
  Used by MACUnit for sample history and intermediate filter outputs.
  """
  def __init__(self, size: int, dtype=float):
    self.size = size
    self.dtype = dtype
    self.buffer = [dtype(0)] * size
    self.write_ptr = 0

  # Push a new value, overwriting the oldest
  def push(self, value) -> None:
    self.buffer[self.write_ptr] = self.dtype(value)
    self.write_ptr = (self.write_ptr + 1) % self.size

  # Negative indexing: -1 = newest sample, -2 = second newest, etc.
  def __getitem__(self, index: int):
    if index >= 0:
      raise IndexError("CircularBuffer only supports negative indexing")
    if abs(index) > self.size:
      raise IndexError(f"Index {index} out of range for buffer of size {self.size}")
    actual = (self.write_ptr + index) % self.size
    return self.buffer[actual]

  def __repr__(self) -> str:
    return f"<CircularBuffer size={self.size} write_ptr={self.write_ptr}>"
