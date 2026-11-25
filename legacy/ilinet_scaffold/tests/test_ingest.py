import pytest

from src import config, ingest

GOOD_CSV = b"region,epiweek,wili,ili,num_ili,num_patients\nnat,202301,3.1,3.0,100,3300\n"


def test_validate_accepts_expected_columns():
    df = ingest.validate(GOOD_CSV, config.SOURCES["ilinet"]["expected_columns"])
    assert len(df) == 1


def test_validate_rejects_missing_columns():
    with pytest.raises(ValueError, match="wili"):
        ingest.validate(b"region,epiweek\nnat,202301\n", ["region", "epiweek", "wili"])


def test_validate_rejects_empty():
    with pytest.raises(ValueError, match="no rows"):
        ingest.validate(b"region,epiweek\n", ["region", "epiweek"])


def test_save_raw_writes_bytes_untouched(tmp_path):
    path = ingest.save_raw(GOOD_CSV, "ilinet", tmp_path)
    assert path.parent == tmp_path
    assert path.name.startswith("ilinet_") and path.suffix == ".csv"
    assert path.read_bytes() == GOOD_CSV


def test_ingest_end_to_end_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "download", lambda source: GOOD_CSV)
    path = ingest.ingest("ilinet", tmp_path)
    assert path.exists()
