"""The dev tool's batch editor: all or nothing, line endings kept."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "dev", Path(__file__).resolve().parents[1] / "scripts" / "dev.py"
)
dev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dev)


def test_edits_apply_once_each_and_keep_crlf(tmp_path):
    crlf = tmp_path / "a.py"
    crlf.write_bytes(b'x = "a"\r\nprint("\\n".join(x))\r\ny = 1\r\n')
    lf = tmp_path / "b.py"
    lf.write_bytes(b"one\ntwo\n")  # write_text would make it CRLF on Windows
    report = dev.apply_edits(
        [
            {"file": "a.py", "old": 'print("\\n".join(x))', "new": 'print("\\n\\n".join(x))'},
            {"file": "b.py", "old": "one\ntwo", "new": "1\n2"},
        ],
        tmp_path,
    )

    assert len(report) == 2
    assert crlf.read_bytes() == b'x = "a"\r\nprint("\\n\\n".join(x))\r\ny = 1\r\n'
    assert lf.read_bytes() == b"1\n2\n"


def test_a_missing_or_ambiguous_match_writes_nothing(tmp_path):
    path = tmp_path / "a.py"
    path.write_bytes(b"a\na\nb\n")
    for edits in (
        [{"file": "a.py", "old": "b", "new": "B"}, {"file": "a.py", "old": "zzz", "new": "x"}],
        [{"file": "a.py", "old": "b", "new": "B"}, {"file": "a.py", "old": "a", "new": "x"}],
    ):
        with pytest.raises(ValueError):
            dev.apply_edits(edits, tmp_path)
        assert path.read_bytes() == b"a\na\nb\n"  # the first edit wasn't written either
    dev.apply_edits([{"file": "a.py", "old": "a", "new": "x", "all": True}], tmp_path)
    assert path.read_bytes() == b"x\nx\nb\n"


def test_edits_in_one_file_build_on_each_other(tmp_path):
    path = tmp_path / "a.py"
    path.write_bytes(b"one\n")
    dev.apply_edits(
        [
            {"file": "a.py", "old": "one", "new": "two"},
            {"file": "a.py", "old": "two", "new": "three"},
        ],
        tmp_path,
    )
    assert path.read_bytes() == b"three\n"


def test_a_python_spec_needs_no_escaping_for_code(tmp_path):
    target = tmp_path / "a.py"
    target.write_bytes(b'print("\\n".join(x))\n')
    spec_file = tmp_path / "spec.py"
    spec_file.write_text(
        "EDITS = [{'file': 'a.py', 'old': r'''\"\\n\".join''', 'new': r'''\"\\n\\n\".join'''}]\n"
    )
    assert dev.apply_edits(dev.load_spec(str(spec_file)), tmp_path)
    assert target.read_bytes() == b'print("\\n\\n".join(x))\n'


def test_the_command_line_reports_a_bad_spec_without_writing(tmp_path, capsys):
    spec_file = tmp_path / "spec.json"
    spec_file.write_text('[{"file": "missing.py", "old": "a", "new": "b"}]')
    assert dev.sub(str(spec_file)) == 1
    assert "nothing written" in capsys.readouterr().err
    assert dev.main(["nonsense"]) == 2
