"""Databestanden inlezen — gedeeld door app en CLI."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pandas as pd

# Volgorde telt: UTF-8 (met of zonder BOM) faalt hard op andere bytes, Windows-1252
# accepteert vrijwel alles en is de gangbare export uit Excel en SPSS.
_ENCODINGS = ("utf-8-sig", "cp1252")
_DELIMITERS = (",", ";", "\t", "|")
_SNIFF_LINES = 20


def read_table(source: Path | bytes, name: str | None = None) -> pd.DataFrame:
    """Lees een CSV- of Parquet-bestand in.

    *source* is een pad of de ruwe inhoud; *name* is de bestandsnaam als *source*
    bytes is. Bij CSV worden scheidingsteken en tekencodering uit het bestand
    afgeleid, zodat ook een export met puntkomma's of in Windows-1252 goed binnenkomt.
    """
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    if str(name or source).lower().endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(data))

    encoding = _detect_encoding(data)
    head = data[:65536].decode(encoding, errors="replace")
    return pd.read_csv(io.BytesIO(data), sep=_detect_delimiter(head), encoding=encoding)


def _detect_encoding(data: bytes) -> str:
    for encoding in _ENCODINGS[:-1]:
        try:
            data.decode(encoding)
        except UnicodeDecodeError:
            continue
        return encoding
    return _ENCODINGS[-1]


def _detect_delimiter(text: str) -> str:
    """Kies het scheidingsteken dat de eerste regels in de meeste gelijke kolommen splitst."""
    lines = text.splitlines()[:_SNIFF_LINES]
    best, best_width = ",", 1
    for delimiter in _DELIMITERS:
        widths = {len(row) for row in csv.reader(lines, delimiter=delimiter) if row}
        if len(widths) == 1 and (width := widths.pop()) > best_width:
            best, best_width = delimiter, width
    return best
