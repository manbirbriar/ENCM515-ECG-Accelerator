from __future__ import annotations
import heapq
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from hardware_unit import HardwareUnit

class ClockUnit:
  """
  Event-driven clock. Instead of ticking every cycle, units schedule
  future events and the clock jumps directly to the next event.
  This makes simulation speed independent of clock frequency.
  """
  def __init__(self):
    self.cycle: int = 0
    self._event_queue: list = []  # heap of (cycle, seq, unit, event)
    self._seq: int = 0            # tiebreaker to keep heap stable

  def schedule(self, cycle: int, unit: 'HardwareUnit', event: str) -> None:
    heapq.heappush(self._event_queue, (cycle, self._seq, unit, event))
    self._seq += 1

  def run(self, on_progress=None) -> None:
    while self._event_queue:
      cycle, _, unit, event = heapq.heappop(self._event_queue)
      self.cycle = cycle
      unit.handle_event(event, cycle, self)
      if on_progress:
        on_progress(cycle)

  def __repr__(self) -> str:
    return f"<ClockUnit cycle={self.cycle} events_remaining={len(self._event_queue)}>"
