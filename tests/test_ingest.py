import json

from epicast import config, ingest


def test_save_raw_writes_bytes_untouched_and_logs_them(tmp_path):
    content = b"geo,time,x\nnga,2024,1\n"
    path = ingest.save_raw(content, "demo", "csv", "https://example.org/x.csv", tmp_path)
    assert path.read_bytes() == content
    assert path.name.startswith("demo__") and path.suffix == ".csv"
    entry = json.loads((tmp_path / "manifest.jsonl").read_text().splitlines()[-1])
    assert entry["bytes"] == len(content) and len(entry["sha256"]) == 64
    assert entry["url"] == "https://example.org/x.csv"


def test_latest_raw_returns_the_newest_file(tmp_path):
    first = ingest.save_raw(b"old", "demo", "csv", "u", tmp_path)
    second = ingest.save_raw(b"new", "demo", "csv", "u", tmp_path)
    assert first != second
    assert ingest.latest_raw("demo", tmp_path).read_bytes() == b"new"


def test_wdi_mirror_url_uses_the_pinned_commit_and_lowercase_code():
    url = ingest.wdi_mirror_url("SH.MLR.INCD.P3")
    assert config.WDI_MIRROR_COMMIT in url
    assert url.endswith("ddf--datapoints--sh_mlr_incd_p3--by--geo--time.csv")


def test_ingest_downloads_every_source_without_touching_the_network(tmp_path, monkeypatch):
    requested = []
    monkeypatch.setattr(ingest, "fetch", lambda url, params=None, retries=3: requested.append(url) or b"x")
    paths = ingest.ingest("mirror", tmp_path)
    assert len(paths) == 8 and all(p.exists() for p in paths)   # 2 annual + 2 Ebola + line list + 3 boundary files
    assert sum("open-numbers" in u for u in requested) == 2
    assert sum(config.BVD_LINELIST_COMMIT in u for u in requested) == 1
    paths = ingest.ingest("who", tmp_path)
    assert sum(u.startswith(config.WHO_GHO_BASE) for u in requested) == 2
    ingest.ingest("mirror", tmp_path, live_outbreak=True)
    assert any("/main/" in u for u in requested)                # --live-outbreak reads the newest line list
