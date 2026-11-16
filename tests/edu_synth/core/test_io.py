"""Tests voor het inlezen van databestanden."""

import pandas as pd
import pytest

from edu_synth.core.io import read_table

FRAME = pd.DataFrame(
    {"naam": ["José", "Zoë", "Anne"], "opleiding": ["a, b", "c", "d"], "ec": [60, 45, 30]}
)


@pytest.mark.parametrize("sep", [",", ";", "\t", "|"])
def test_read_table_detects_delimiter(sep):
    data = FRAME.to_csv(index=False, sep=sep).encode("utf-8")
    pd.testing.assert_frame_equal(read_table(data, "export.csv"), FRAME)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "cp1252"])
def test_read_table_detects_encoding(encoding):
    data = FRAME.to_csv(index=False, sep=";").encode(encoding)
    pd.testing.assert_frame_equal(read_table(data, "export.csv"), FRAME)


def test_read_table_single_column_csv():
    data = b"opmerking\neen; twee\ndrie\n"
    assert read_table(data, "export.csv").columns.tolist() == ["opmerking"]


def test_read_table_from_path(tmp_path):
    path = tmp_path / "export.csv"
    FRAME.to_csv(path, index=False, sep=";", encoding="cp1252")
    pd.testing.assert_frame_equal(read_table(path), FRAME)


def test_read_table_parquet(tmp_path):
    path = tmp_path / "export.parquet"
    FRAME.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(read_table(path), FRAME)
    pd.testing.assert_frame_equal(read_table(path.read_bytes(), "export.parquet"), FRAME)
