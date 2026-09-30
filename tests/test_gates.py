"""Reading a repo's checks from its own CI, and the conventions when it has none."""

from __future__ import annotations

from watchtower import gates

# BavarianData's test workflow, trimmed: the shape most repos have.
TESTS = """
name: Tests
on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: "3.14"
      - uses: actions/setup-node@v7
        with:
          node-version: "24"
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements_test.txt
      - name: Lint (ruff)
        run: ruff check custom_components/bavariandata tests
      - name: Format (ruff)
        run: ruff format --check custom_components/bavariandata tests tools
      - name: Install Node dev dependencies
        run: npm ci
      - name: Lint the card (ESLint)
        run: npx eslint custom_components/bavariandata/www
      - name: Run tests
        run: pytest tests/ -q --cov --cov-report=term:skip-covered
"""


def names(steps) -> list[str]:
    return [s.name for s in steps]


def test_the_pull_request_workflow_gives_setup_and_gates():
    plan = gates.from_workflow(TESTS, "test.yml")

    assert names(plan.setup) == ["Install dependencies", "Install Node dev dependencies"]
    assert names(plan.gates) == [
        "Lint (ruff)",
        "Format (ruff)",
        "Lint the card (ESLint)",
        "Run tests",
    ]
    assert plan.gates[0].script == "ruff check custom_components/bavariandata tests"
    assert (plan.python, plan.node, plan.not_run) == ("3.14", "24", ())
    assert plan


def test_a_workflow_that_does_not_run_for_pull_requests_gives_nothing():
    release = "on:\n  push:\n    tags: ['v*']\njobs:\n  x:\n    steps:\n      - run: make release\n"
    target = "on: pull_request_target\njobs:\n  x:\n    steps:\n      - run: make test\n"
    assert gates.from_workflow(release) is None
    assert gates.from_workflow(target) is None  # runs with the base's rights: never
    assert gates.from_workflow("- not\n- a workflow") is None
    assert gates.from_workflow("on: [pull_request\n  broken") is None


def test_steps_this_box_cannot_run_are_named_not_skipped_silently():
    text = """
on: [pull_request]
jobs:
  validate:
    steps:
      - uses: actions/checkout@v7
      - name: Hassfest
        uses: home-assistant/actions/hassfest@master
      - name: Publish
        run: twine upload dist/*
        env:
          TOKEN: ${{ secrets.PYPI }}
      - name: Notify
        run: curl -X POST https://example.invalid/hook
      - name: Tag
        run: echo ${{ github.sha }}
      - name: Test
        run: pytest -q
  deploy:
    environment: production
    steps:
      - run: ./deploy.sh
"""
    plan = gates.from_workflow(text)

    assert names(plan.gates) == ["Test"]
    assert dict(plan.not_run) == {
        "Hassfest": "it uses the action home-assistant/actions/hassfest",
        "Publish": "it needs a secret",
        "Notify": "it reaches outside the sandbox",
        "Tag": "it uses a workflow expression",
        "job deploy": "it deploys to an environment",
    }


def test_a_matrix_stands_for_its_first_value_and_a_working_directory_stays_inside():
    text = """
on: pull_request
jobs:
  test:
    strategy:
      matrix:
        python: ["3.13", "3.14"]
    steps:
      - name: Test
        run: python${{ matrix.python }} -m pytest
        working-directory: ../../etc
      - name: Web
        run: npm test
        working-directory: web
"""
    plan = gates.from_workflow(text)
    assert [(s.script, s.workdir) for s in plan.gates] == [
        ("python3.13 -m pytest", "."),
        ("npm test", "web"),
    ]
    assert gates.safe_workdir("/abs") == "." and gates.safe_workdir("a/b") == "a/b"


def test_the_plan_of_a_repo_merges_its_workflows(tmp_path):
    folder = tmp_path / ".github" / "workflows"
    folder.mkdir(parents=True)
    (folder / "test.yml").write_text(TESTS)
    (folder / "lint.yaml").write_text(
        "on: pull_request\njobs:\n  l:\n    steps:\n      - run: ruff check custom_components/bavariandata tests\n"
        "      - run: codespell\n"
    )
    (folder / "release.yml").write_text("on: push\njobs:\n  r:\n    steps:\n      - run: make\n")
    plan = gates.plan(tmp_path)

    assert plan.source == "the repo's CI workflows"
    # the repeated ruff step is kept once
    assert [s.script for s in plan.gates].count(
        "ruff check custom_components/bavariandata tests"
    ) == 1
    assert "codespell" in [s.script for s in plan.gates] and "make" not in str(plan)


def test_a_repo_without_ci_falls_back_to_its_conventions(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\nline-length = 100\n")
    (tmp_path / "requirements_test.txt").write_text("pytest\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "package.json").write_text('{"scripts": {"lint": "eslint .", "build": "x"}}')
    (tmp_path / "package-lock.json").write_text("{}")
    plan = gates.plan(tmp_path)

    assert plan.source == "the project's conventions"
    assert [s.script for s in plan.setup] == ["pip install -r requirements_test.txt", "npm ci"]
    assert [s.script for s in plan.gates] == [
        "ruff check .",
        "ruff format --check .",
        "python -m pytest -q",
        "npm run lint",
    ]


def test_a_makefile_is_the_last_resort_and_nothing_is_nothing(tmp_path):
    assert not gates.plan(tmp_path)
    (tmp_path / "Makefile").write_text("build:\n\tcc x.c\ntest:\n\t./t\n")
    assert [s.script for s in gates.plan(tmp_path).gates] == ["make test"]


def test_the_override_wins_over_everything(tmp_path):
    (tmp_path / ".watchtower").mkdir()
    (tmp_path / ".watchtower" / "gates.toml").write_text(
        'setup = ["pip install -e .[test]"]\ngates = ["tox -e py", 42]\npython = "3.13"\n'
    )
    (tmp_path / "Makefile").write_text("test:\n\t./t\n")
    plan = gates.plan(tmp_path)

    assert [s.script for s in plan.setup] == ["pip install -e .[test]"]
    assert [s.script for s in plan.gates] == ["tox -e py"]
    assert (plan.python, plan.source) == ("3.13", "the repo's .watchtower/gates.toml")
    (tmp_path / ".watchtower" / "gates.toml").write_text("not = = toml")
    assert gates.plan(tmp_path).source == "the project's conventions"
