"""The toolchain service: installing a repo's dependencies for the runner to use."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import test_checks
import yaml
from test_checks import Src, make_draft, step

from watchtower import checking, gates, reviews, runner, sandbox, toolchain

PY = (sys.executable, "-c")
REPO = "owner/repo"
KEY = "k" * 32


def make_tree(tmp_path, **files):
    tree = tmp_path / "main"
    tree.mkdir(exist_ok=True)
    for name, text in (files or {"requirements_test.txt": "pytest\n"}).items():
        path = tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tree


# -- the key --------------------------------------------------------------------------------


def test_the_key_follows_the_dependency_files_and_the_setup_commands(tmp_path):
    setup = (gates.Step("i", "pip install -r requirements_test.txt"),)
    tree = make_tree(tmp_path, **{"requirements_test.txt": "pytest\n", "src/app.py": "x = 1"})
    key = toolchain.tools_key(tree, setup)
    assert len(key) == 32 and toolchain.tools_key(tree, setup) == key

    (tree / "src" / "app.py").write_text("x = 2")  # code doesn't decide what gets installed
    (tree / "node_modules").mkdir()
    (tree / "node_modules" / "package.json").write_text("{}")
    assert toolchain.tools_key(tree, setup) == key

    (tree / "requirements_test.txt").write_text("pytest\nruff\n")
    changed = toolchain.tools_key(tree, setup)
    assert changed != key
    assert toolchain.tools_key(tree, (gates.Step("i", "pip install pytest"),)) != changed
    (tree / "package-lock.json").write_text("{}")
    assert toolchain.tools_key(tree, setup) != changed


def test_environment_paths_are_owner_name_only(tmp_path):
    assert toolchain.env_path(tmp_path, REPO) == tmp_path / "owner" / "repo"
    assert toolchain.env_path(tmp_path, "../x/y") is None
    assert toolchain.env_path(tmp_path, "a/../b") is None
    assert toolchain.current_key(tmp_path, REPO) is None
    assert toolchain.current_key(tmp_path, "nonsense") is None


# -- installing -----------------------------------------------------------------------------

MAKE_ENV = "import os; os.makedirs(os.environ['VIRTUAL_ENV'] + '/bin')"
INSTALL = "import os; open(os.environ['VIRTUAL_ENV'] + '/bin/ruff', 'w').write('x')"
NPM = "import os; os.makedirs('node_modules/.bin'); open('node_modules/.bin/eslint', 'w')"


def submit(tmp_path, steps, key=KEY):
    root = tmp_path / "toolchain"
    job = sandbox.Job("t1", REPO, 0, key, tuple(steps))
    return sandbox.write_job(root, job, make_tree(tmp_path))


def test_a_green_install_swaps_the_environment_in_with_its_key_and_modules(tmp_path):
    tools = tmp_path / "tools"
    where = submit(
        tmp_path,
        [gates.Step("env", MAKE_ENV), gates.Step("pip", INSTALL), gates.Step("npm", NPM)],
    )
    result = toolchain.install(where, tools, shell=PY, sweep=None)

    final = tools / "owner" / "repo"
    assert result.green and sandbox.read_result(tmp_path / "toolchain", "t1").green
    assert (final / "bin" / "ruff").is_file() and (
        final / "node_modules" / ".bin" / "eslint"
    ).exists()
    assert toolchain.current_key(tools, REPO) == KEY
    assert not (final.parent / "repo.new").exists() and not (final.parent / "repo.old").exists()


def test_a_failed_install_leaves_the_working_environment_alone(tmp_path):
    tools = tmp_path / "tools"
    old = tools / "owner" / "repo"
    (old / "bin").mkdir(parents=True)
    (old / "bin" / "ruff").write_text("old")
    (old / ".key").write_text("old-key")
    where = submit(
        tmp_path,
        [
            gates.Step("env", MAKE_ENV),
            gates.Step("pip", "import sys; print('no such package'); sys.exit(1)"),
        ],
    )
    result = toolchain.install(where, tools, shell=PY, sweep=None)

    assert not result.green
    assert "no such package" in sandbox.read_result(tmp_path / "toolchain", "t1").steps[1].output
    assert (
        toolchain.current_key(tools, REPO) == "old-key"
        and (old / "bin" / "ruff").read_text() == "old"
    )
    assert not (old.parent / "repo.new").exists()


def test_a_newer_install_replaces_the_older_one(tmp_path):
    tools = tmp_path / "tools"
    old = tools / "owner" / "repo"
    old.mkdir(parents=True)
    (old / ".key").write_text("old-key")
    (old / "stale.txt").write_text("stale")
    where = submit(tmp_path, [gates.Step("env", MAKE_ENV)])
    toolchain.install(where, tools, shell=PY, sweep=None)
    assert toolchain.current_key(tools, REPO) == KEY and not (old / "stale.txt").exists()


def test_the_result_is_published_only_after_the_swap(tmp_path, monkeypatch):
    tools = tmp_path / "tools"
    where = submit(tmp_path, [gates.Step("env", MAKE_ENV)])
    seen: list[bool] = []
    real = sandbox.publish

    def watch(w, result):
        seen.append(toolchain.current_key(tools, REPO) == KEY)  # the environment is in place
        real(w, result)

    monkeypatch.setattr(sandbox, "publish", watch)
    toolchain.install(where, tools, shell=PY, sweep=None)
    assert seen == [True]


def test_a_job_for_a_nonsense_repo_name_is_refused(tmp_path):
    root = tmp_path / "toolchain"
    job = sandbox.Job("t1", "../evil", 0, KEY, (gates.Step("env", MAKE_ENV),))
    where = sandbox.write_job(root, job, make_tree(tmp_path))
    result = toolchain.install(where, tmp_path / "tools", shell=PY, sweep=None)
    assert result.error == "not a repository name"
    assert not (tmp_path / "tools").exists()


def test_the_loop_takes_one_setup_job(tmp_path):
    root = tmp_path / "toolchain"
    assert toolchain.run(root, tmp_path / "tools", sleep=lambda _: None, waits=1) is None
    submit(tmp_path, [gates.Step("env", MAKE_ENV)])
    assert toolchain.run(root, tmp_path / "tools", shell=PY, sweep=None) == "t1"


def test_the_setup_job_creates_the_environment_first():
    job = toolchain.setup_job("t2", REPO, KEY, (gates.Step("i", "pip install x"),))
    assert [s.script for s in job.steps] == ['python3 -m venv "$VIRTUAL_ENV"', "pip install x"]
    assert job.sha == KEY and job.repo == REPO


def test_the_runner_shows_the_repos_node_modules_in_the_tree(tmp_path):
    tools = tmp_path / "tools"
    (tools / "node_modules").mkdir(parents=True)
    tree = tmp_path / "tree"
    tree.mkdir()
    try:
        os.symlink(tools, tmp_path / "probe", target_is_directory=True)
    except OSError:
        pytest.skip("this system doesn't let the test make symlinks")
    runner.link_modules(tree, tools)
    assert (tree / "node_modules").is_symlink()

    own = tmp_path / "own"
    (own / "node_modules").mkdir(parents=True)
    runner.link_modules(own, tools)
    assert not (own / "node_modules").is_symlink()  # a tree's own stays
    runner.link_modules(own, None)  # no environment: nothing to link


# -- the watcher's setup states ----------------------------------------------------------------


@pytest.fixture
def setup_world(store, tmp_path):
    root, box = tmp_path / "repos", tmp_path / "sandbox"
    main = root / "owner" / "repo"
    (main / ".github" / "workflows").mkdir(parents=True)
    (main / ".github" / "workflows" / "test.yml").write_text(test_checks.WORKFLOW)
    (main / "requirements_test.txt").write_text("pytest\n")
    box.mkdir()
    toolbox, tools = tmp_path / "toolchain", tmp_path / "tools"
    tools.mkdir()
    source = Src(True)
    test_checks.fetch(root, source)
    return root, box, toolbox, tools, source, main


def settle(draft, world, store, **extra):
    root, box, toolbox, tools, source, _ = world
    return checking.settle(draft, source, store, root, box, toolbox=toolbox, tools=tools, **extra)


def test_stale_dependencies_are_installed_before_the_checks_run(store, setup_world):
    draft = make_draft(store)
    root, box, toolbox, tools, source, main = setup_world

    assert settle(draft, setup_world, store, now=10.0) is False
    stage = checking.state(store, draft)
    assert (stage["state"], stage["setup_job"]) == ("setting-up", f"t{draft.id}")
    (folder,) = (toolbox / "jobs").iterdir()
    job = sandbox.Job.from_json((folder / "job.json").read_text())
    assert [s.script for s in job.steps][1:] == ["pip install -r requirements_test.txt"]
    assert job.sha == toolchain.tools_key(main, gates.plan(main).setup)
    assert (folder / "tree" / "requirements_test.txt").is_file()  # the default branch
    assert not sandbox.busy(box)  # no check job yet

    assert settle(draft, setup_world, store, now=20.0) is False  # still installing
    (folder / "result.json").write_text(
        sandbox.Result(folder.name, (step("create the environment"),)).to_json()
    )
    assert settle(draft, setup_world, store, now=30.0) is False
    assert checking.state(store, draft)["state"] == "running"  # on to the checks
    assert not (toolbox / "jobs" / folder.name).exists() and sandbox.busy(box)


def test_current_dependencies_go_straight_to_the_checks(store, setup_world):
    draft = make_draft(store)
    root, box, toolbox, tools, source, main = setup_world
    key = toolchain.tools_key(main, gates.plan(main).setup)
    (tools / "owner" / "repo").mkdir(parents=True)
    (tools / "owner" / "repo" / ".key").write_text(key)

    settle(draft, setup_world, store, now=1.0)
    assert checking.state(store, draft)["state"] == "running"
    assert not (toolbox / "jobs").exists()


def test_dependencies_that_cannot_be_installed_end_the_checks_with_the_output(store, setup_world):
    draft = make_draft(store)
    settle(draft, setup_world, store, now=1.0)
    toolbox = setup_world[2]
    (folder,) = (toolbox / "jobs").iterdir()
    failed = sandbox.Result(
        folder.name,
        (
            step("create the environment"),
            step("Install", "failed", 1, "ERROR: no matching version"),
        ),
    )
    (folder / "result.json").write_text(failed.to_json())

    assert settle(draft, setup_world, store, now=2.0) is True
    stage = checking.state(store, draft)
    assert stage["state"] == "setup-failed"
    lines, outputs = reviews.gate_facts(stage)
    assert "dependencies couldn't be installed" in lines[0]
    assert "no matching version" in outputs and reviews.gate_report(stage) == ""


def test_an_install_that_never_finishes_times_out_as_failed(store, setup_world):
    draft = make_draft(store)
    settle(draft, setup_world, store, now=1.0)
    late = 1.0 + checking.WAIT_SECONDS + 1
    assert settle(draft, setup_world, store, now=late) is True
    assert checking.state(store, draft)["state"] == "setup-failed"
    assert not sandbox.busy(setup_world[2])
    assert "couldn't be installed" in reviews.gate_facts(checking.state(store, draft))[0][0]


def test_a_second_pr_waits_while_the_toolchain_is_busy(store, setup_world, tmp_path):
    draft = make_draft(store)
    toolbox = setup_world[2]
    (tmp_path / "t").mkdir()
    sandbox.write_job(toolbox, sandbox.Job("other", REPO, 0, "k", ()), tmp_path / "t")

    assert settle(draft, setup_world, store) is False
    assert checking.state(store, draft)["state"] == "waiting"
    sandbox.remove_job(toolbox, "other")
    assert settle(draft, setup_world, store, now=5.0) is False
    assert checking.state(store, draft)["state"] == "setting-up"


def test_without_the_toolchain_volumes_nothing_is_installed(store, setup_world):
    draft = make_draft(store)
    root, box, _, _, source, _ = setup_world
    assert checking.settle(draft, source, store, root, box, now=1.0) is False
    assert checking.state(store, draft)["state"] == "running"


# -- compose ---------------------------------------------------------------------------------


def test_compose_gives_the_toolchain_a_network_and_nothing_else():
    text = (Path(__file__).parents[1] / "compose.yaml").read_text("utf-8")
    services = yaml.safe_load(text)["services"]
    tool = services["toolchain"]
    assert tool["runtime"] == "runsc" and tool["cap_drop"] == ["ALL"] and tool["read_only"] is True
    assert tool["networks"] == ["internet"] and "secrets" not in tool
    assert [v.split(":")[1] for v in tool["volumes"]] == ["/toolchain", "/tools"]
    assert services["runner"]["network_mode"] == "none"
    watcher = {v.split(":")[1]: v for v in services["watcher"]["volumes"]}
    assert watcher["/tools"].endswith(":ro") and "/toolchain" in watcher
    others = [n for n, s in services.items() if "internet" in (s.get("networks") or [])]
    assert others == ["toolchain"]  # nobody else shares its network


def test_the_dockerfile_carries_node_for_the_runner():
    text = (Path(__file__).parents[1] / "Dockerfile").read_text("utf-8")
    assert "COPY --from=node:" in text and "/usr/local/bin/npx" in text
