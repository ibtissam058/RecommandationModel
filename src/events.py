"""Event calendar: Black Friday, Halloween, Christmas, Valentine's Day, ...

To add or change an event, edit the EVENTS list at the bottom.
`window` = how many days BEFORE the event the shopping peak starts ramping up.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Optional

import pandas as pd


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th given weekday (Mon=0 ... Sun=6) of a month; n=-1 means the last one."""
    if n > 0:
        first = date(year, month, 1)
        shift = (weekday - first.weekday()) % 7
        return first + timedelta(days=shift + 7 * (n - 1))
    last = date(year, month, calendar.monthrange(year, month)[1])
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def black_friday(year: int) -> date:
    """Day after US Thanksgiving (4th Thursday of November)."""
    return nth_weekday(year, 11, 3, 4) + timedelta(days=1)


@dataclass(frozen=True)
class Event:
    name: str
    window: int
    when: Callable[[int], date]

    def date(self, year: int) -> date:
        return self.when(year)


EVENTS = [
    Event("valentines_day", 14, lambda y: date(y, 2, 14)),
    Event("womens_day", 10, lambda y: date(y, 3, 8)),  # big gift day in Eastern Europe
    Event("mothers_day", 10, lambda y: nth_weekday(y, 5, 6, 2)),  # 2nd Sunday of May
    Event("halloween", 7, lambda y: date(y, 10, 31)),
    Event("black_friday", 7, black_friday),
    Event("christmas", 14, lambda y: date(y, 12, 25)),
]


def ramp_for(when) -> tuple[Optional[Event], Optional[int], float]:
    """Seasonal intensity of a date.

    Returns (event, days_to_event, ramp):
      * ramp is 0 outside every event window,
      * grows linearly to 1 on the event day itself.
    If windows overlap, the closest upcoming event wins.
    """
    d = pd.Timestamp(when).date()
    best = None
    for ev in EVENTS:
        for y in (d.year, d.year + 1):
            delta = (ev.date(y) - d).days
            if 0 <= delta <= ev.window:
                ramp = 1.0 - delta / (ev.window + 1)
                if best is None or delta < best[1]:
                    best = (ev, delta, ramp)
    return best if best else (None, None, 0.0)


def event_calendar(first_year: int, last_year: int) -> list[tuple[date, str]]:
    return sorted((ev.date(y), ev.name) for y in range(first_year, last_year + 1) for ev in EVENTS)


def describe_season(when) -> str:
    ev, days, ramp = ramp_for(when)
    if ev is None:
        return "Normal day (no special event nearby)"
    label = ev.name.replace("_", " ").title()
    return f"{label} in {days} day(s) - seasonal intensity {ramp:.2f}"