"""The sandbox protocol: a job goes in, a checked result comes out."""

from __future__ import annotations

import json
import os
import sys

import pytest

from watchtower import gates, sandbox

PY = (sys.executable, "-c")  # the steps below are Python: the same on every OS


def job(steps, **extra) -> sandbox.Job:
    fields = {"id": "job-1", "repo": "owner/repo", "number": 7, "sha": "e" * 40}
    return sandbox.Job(steps=tuple(steps), **(fields | extra))


def tree(tmp_path):
    src = tmp_path / "src"
    (src / "web").mkdir(parents=True, exist_ok=True)
    (src / "README.md").write_text("hi")
    return src


def submit(tmp_path, steps, **extra):
    root = tmp_path / "sandbox"
    return sandbox.write_job(root, job(steps, **extra), tree(tmp_path))


def step(script, name="check", workdir="."):
    return gates.Step(name, script, workdir)


def test_a_job_is_written_tree_first_and_job_file_last(tmp_path):
    where = submit(tmp_path, [step("pass")])

    assert (where / "tree" / "README.md").read_text() == "hi"
    assert sandbox.Job.from_json((where / "job.json").read_text()) == job([step("pass")])
    assert sandbox.next_job(tmp_path / "sandbox") == where
    assert not (where / "job.json.new").exists()


def test_the_runner_takes_the_oldest_job_without_a_result(tmp_path):
    root = tmp_path / "sandbox"
    assert sandbox.next_job(root) is None
    first = sandbox.write_job(root, job([step("pass")], id="a"), tree(tmp_path))
    os.utime(first / "job.json", (1, 1))
    second = sandbox.write_job(root, job([step("pass")], id="b"), tree(tmp_path))
    assert sandbox.next_job(root) == first
    (first / "result.json").write_text("{}")  # done
    assert sandbox.next_job(root) == second
    (second / "job.json").unlink()  # not fully written yet
    assert sandbox.next_job(root) is None


def test_steps_run_in_the_tree_and_the_result_says_how_each_went(tmp_path):
    where = submit(
        tmp_path,
        [
            step("print(open('README.md').read())", "reads the tree"),
            step("import sys; print('boom'); sys.exit(3)", "fails"),
            step("print(1)", "in a folder", "web"),
            step("print(1)", "no such folder", "nowhere"),
        ],
    )
    result = sandbox.execute(where, shell=PY)

    got = [(s.name, s.status, s.code) for s in result.steps]
    assert got == [
        ("reads the tree", "passed", 0),
        ("fails", "failed", 3),
        ("in a folder", "passed", 0),
        ("no such folder", "skipped", None),
    ]
    assert result.steps[0].output.strip() == "hi" and "boom" in result.steps[1].output
    assert not result.green
    assert sandbox.read_result(tmp_path / "sandbox", "job-1") == result  # what the watcher reads


def test_an_all_passed_job_is_green_and_lists_what_could_not_run(tmp_path):
    where = submit(tmp_path, [step("pass")], not_run=(("Hassfest", "it uses the action x"),))
    result = sandbox.execute(where, shell=PY)
    assert result.green and result.not_run == (("Hassfest", "it uses the action x"),)


def test_a_step_that_takes_too_long_is_killed(tmp_path):
    where = submit(tmp_path, [step("import time; time.sleep(30)", "slow"), step("pass", "after")])
    job_file = where / "job.json"
    data = json.loads(job_file.read_text())
    data["step_seconds"] = 1
    job_file.write_text(json.dumps(data))
    result = sandbox.execute(where, shell=PY)

    assert [(s.name, s.status, s.code) for s in result.steps] == [
        ("slow", "timeout", None),
        ("after", "passed", 0),
    ]
    assert result.steps[0].seconds < 10


def test_the_whole_job_has_a_budget_too(tmp_path):
    where = submit(tmp_path, [step("import time; time.sleep(30)", "slow"), step("pass", "after")])
    data = json.loads((where / "job.json").read_text())
    data["step_seconds"], data["total_seconds"] = 5, 1
    (where / "job.json").write_text(json.dumps(data))
    result = sandbox.execute(where, shell=PY)
    assert [s.status for s in result.steps] == ["timeout", "skipped"]
    assert result.steps[1].output == "out of time"


def test_only_the_end_of_a_long_output_is_kept(tmp_path):
    where = submit(tmp_path, [step("print('x' * 100000); print('the end')")])
    (out,) = sandbox.execute(where, shell=PY).steps
    assert len(out.output) <= sandbox.MAX_OUTPUT and out.output.strip().endswith("the end")


def test_a_step_gets_no_secret_from_the_runner_and_only_the_tools_it_was_given(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    tools = tmp_path / "tools"
    where = submit(
        tmp_path,
        [
            step("import os; print(os.environ.get('GITHUB_TOKEN', 'none'))", "token"),
            step("import os; print(os.environ['PATH'].split(os.pathsep)[0])", "path"),
            step("import os; print(os.environ['CI'], os.environ['HOME'] != '')", "ci"),
        ],
    )
    result = sandbox.execute(where, tools, shell=PY)

    assert result.steps[0].output.strip() == "none"
    assert result.steps[1].output.strip() == str(tools / "bin")
    assert result.steps[2].output.strip() == "true True"


def test_the_result_is_written_after_everything_a_step_started_is_dealt_with(tmp_path):
    where = submit(tmp_path, [step("pass")])
    order: list[str] = []
    real = sandbox._write

    def watch(w, result, finish):
        finish and order.append("sweep-before-write")
        real(w, result, finish)

    sandbox._write = watch
    try:
        sandbox.execute(where, shell=PY, finish=lambda: None)
    finally:
        sandbox._write = real
    assert order == ["sweep-before-write"]
    assert (where / "result.json").is_file() and not (where / "result.json.new").exists()


def test_an_unreadable_job_still_gets_a_result_saying_so(tmp_path):
    root = tmp_path / "sandbox"
    where = root / "jobs" / "job-1"
    where.mkdir(parents=True)
    (where / "job.json").write_text("{not json")
    result = sandbox.execute(where, shell=PY)
    assert result.error == "the job file is unreadable" and not result.green
    assert sandbox.read_result(root, "job-1").error == "the job file is unreadable"


def test_a_job_lives_in_its_own_folder_and_ids_are_plain(tmp_path):
    where = submit(tmp_path, [step("pass")])
    moved = where.parent / "other"
    where.rename(moved)
    assert sandbox.execute(moved, shell=PY).error
    with pytest.raises(ValueError):
        sandbox.job_dir(tmp_path, "../x")


# -- what comes back is untrusted --------------------------------------------------------------


def good(**extra) -> dict:
    data = {
        "job": "job-1",
        "steps": [
            {
                "name": "pytest",
                "script": "pytest",
                "status": "failed",
                "code": 1,
                "seconds": 2.5,
                "output": "1 failed",
            }
        ],
        "not_run": [["Hassfest", "it uses the action x"]],
        "seconds": 3.0,
        "error": "",
    }
    return data | extra


def test_a_good_result_is_read_and_sizes_are_capped():
    got = sandbox.parse_result(json.dumps(good()), "job-1")
    assert got.steps[0].status == "failed" and got.not_run == (
        ("Hassfest", "it uses the action x"),
    )

    long = good()
    long["steps"][0]["name"] = "n" * 1000
    long["steps"][0]["output"] = "o" * 100_000
    assert sandbox.parse_result(json.dumps(long), "job-1") is None  # over the file cap

    long["steps"][0]["output"] = "o" * (sandbox.MAX_OUTPUT + 500)
    got = sandbox.parse_result(json.dumps(long), "job-1")
    assert len(got.steps[0].name) == sandbox.MAX_NAME
    assert len(got.steps[0].output) == sandbox.MAX_OUTPUT


@pytest.mark.parametrize(
    "bad",
    [
        good(job="another-job"),  # not this job's
        good(steps="passed"),
        good(steps=[1]),
        good(steps=[good()["steps"][0] | {"status": "great"}]),
        good(steps=[good()["steps"][0] | {"code": "0"}]),
        good(steps=[good()["steps"][0] | {"code": True}]),
        good(steps=[good()["steps"][0] | {"seconds": "fast"}]),
        good(steps=[good()["steps"][0] | {"output": 5}]),
        good(steps=[good()["steps"][0]] * (sandbox.MAX_STEPS + 1)),
        ["not", "an", "object"],
    ],
)
def test_a_result_outside_the_shape_is_no_result(bad):
    assert sandbox.parse_result(json.dumps(bad), "job-1") is None


def test_garbage_and_missing_results_are_no_result(tmp_path):
    assert sandbox.parse_result("{nope", "job-1") is None
    assert sandbox.parse_result(b"x" * (sandbox.MAX_RESULT + 1), "job-1") is None
    root = tmp_path / "sandbox"
    assert sandbox.read_result(root, "job-1") is None
    where = sandbox.job_dir(root, "job-1")
    where.mkdir(parents=True)
    (where / "result.json").mkdir()  # a directory in its place
    assert sandbox.read_result(root, "job-1") is None
    assert sandbox.finished(root, "job-1")
