from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from qengine.events import MarketEvent, SignalEvent, Universe


class Strategy(ABC):
    """Consumes MarketEvents, maintains its own indicator state, emits SignalEvents.

    A strategy expresses *views* (direction + conviction). It never sizes
    positions or touches cash — that is the Portfolio's job.
    """

    strategy_id: str = "strategy"

    def __init__(self, universe: Universe) -> None:
        self.universe = universe

    @abstractmethod
    def on_market(self, event: MarketEvent) -> Sequence[SignalEvent]: ...
