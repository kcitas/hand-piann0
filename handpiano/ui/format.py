"""Display formatting. Missing measurements always render as "N/A", never as a guess."""

from __future__ import annotations

NA = "N/A"


def fmt(value: float | None, unit: str = "", digits: int = 1) -> str:
    if value is None:
        return NA
    text = f"{value:.{digits}f}"
    return f"{text} {unit}" if unit else text


def fmt_pct(fraction: float | None, digits: int = 0) -> str:
    return NA if fraction is None else f"{fraction * 100:.{digits}f} %"
