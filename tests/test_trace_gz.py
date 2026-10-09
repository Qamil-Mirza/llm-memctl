"""Old per-step logs are kept gzipped; readers fall back to `name.gz` when the plain file is gone."""

import gzip
import json

from memctl.runlog import open_trace, read_jsonl, trace_exists


def test_reads_plain_then_gz(tmp_path):
    rows = [{"seed": 0, "scored": True}, {"seed": 1}]
    text = "".join(json.dumps(row) + "\n" for row in rows)
    plain = tmp_path / "steps.jsonl"
    plain.write_text(text)
    assert read_jsonl(plain) == rows and trace_exists(plain)
    with gzip.open(tmp_path / "steps.jsonl.gz", "wt") as handle:
        handle.write(text)
    plain.unlink()
    assert trace_exists(plain)
    assert read_jsonl(plain) == rows
    with open_trace(plain) as handle:
        assert [json.loads(line) for line in handle] == rows


def test_missing_file(tmp_path):
    assert not trace_exists(tmp_path / "steps.jsonl")
    assert read_jsonl(tmp_path / "steps.jsonl") == []
