from hardware_unit import HardwareUnit
from circular_buffer import CircularBuffer

# TODO: LOOK OVER
class Scheduler(HardwareUnit):
  """
  Watches the CircularBuffer and dispatches windows to pipeline lanes.

  Each tick, if a full window is ready and at least one lane is free,
  the scheduler extracts the window from the buffer and sends it to
  the first available lane's input_data.

  If all lanes are busy the scheduler stalls — the window stays in the
  buffer and the clock's stalled_cycles counter is incremented via
  is_stalled(). The scheduler also keeps its own local stall counter
  for per-component analysis.

  Parameters
  ----------
  name           : human-readable label
  circular_buffer: the CircularBuffer instance to poll
  lanes          : list of the first HardwareUnit in each pipeline lane
                  (i.e. the LowPassFilter of each lane)
  """

  def __init__(self, name: str, circular_buffer: CircularBuffer, lanes: list[HardwareUnit]):
    super().__init__(name, latency_cycles=1)

    self.circular_buffer: CircularBuffer = circular_buffer
    self.lanes: list[HardwareUnit] = lanes

    self.stall_count: int = 0       # times scheduler waited for a free lane
    self.dispatched_count: int = 0  # total windows successfully dispatched
    self._stalled: bool = False     # current cycle stall state for ClockUnit

  def tick(self, current_cycle: int) -> None:
    self.current_cycle = current_cycle
    self._stalled = False

    if not self.circular_buffer.window_ready():
      return

    free_lane = self._find_free_lane()

    if free_lane is None:
      # All lanes busy — stall
      self._stalled = True
      self.stall_count += 1
      return

    window = self.circular_buffer.get_window()
    free_lane.input_data = window
    self.dispatched_count += 1

  def _find_free_lane(self) -> HardwareUnit | None:
    """Return the first available lane, or None if all are busy."""
    for lane in self.lanes:
      if lane.is_available():
        return lane
    return None

  def compute(self, data: list) -> list:
    # Scheduler does not use the standard compute path
    return data

  def is_stalled(self) -> bool:
    return self._stalled

  def __repr__(self) -> str:
    return (
      f"<Scheduler name={self.name!r} "
      f"dispatched={self.dispatched_count} "
      f"stall_count={self.stall_count} "
      f"stalled_this_cycle={self._stalled}>"
    )