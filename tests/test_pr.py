"""Fetching a pull request for review: facts, patches, and the code at its head."""

from __future__ import annotations

import io
import os
import tarfile

import pytest

from watchtower import pr, snapshot

REPO = "owner/repo"
HEAD = "c" * 40
NEXT = "d" * 40


def tarball(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(f"owner-repo-{HEAD[:7]}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class PullSource:
    def __init__(self, files: int = 2) -> None:
        self.head = HEAD
        self.files = [
            {
                "filename": f"src/f{i}.py",
                "status": "modified",
                "additions": 3,
                "deletions": 1,
                "patch": "@@ -1 +1 @@\n-a\n+b",
            }
            for i in range(files)
        ]
        self.calls: list[str] = []
        self.downloads: list[str] = []
        self.fail_download = False

    def get_json(self, path: str):
        self.calls.append(path)
        if path.endswith("/files?per_page=100&page=1"):
            return self.files[:100]
        if path.endswith("/files?per_page=100&page=2"):
            return self.files[100:200]
        if path.endswith("/files?per_page=100&page=3"):
            return self.files[200:300]
        assert path == f"/repos/{REPO}/pulls/7"
        return {
            "number": 7,
            "title": "Fix the thing",
            "user": {"login": "stranger"},
            "author_association": "NONE",
            "body": "Fixes #3",
            "state": "open",
            "html_url": f"https://github.com/{REPO}/pull/7",
            "head": {"sha": self.head, "ref": "fix", "repo": {"full_name": "stranger/repo"}},
            "base": {"sha": "e" * 40, "ref": "main"},
            "maintainer_can_modify": True,
            "changed_files": len(self.files),
            "labels": [{"name": "bug"}],
        }

    def download(self, path, dest, max_bytes):
        self.downloads.append(path)
        if self.fail_download:
            raise snapshot.SnapshotError("no")
        dest.write_bytes(tarball({"src/f0.py": "b\n", "README.md": "hi"}))


def test_fetch_keeps_the_facts_the_patches_and_the_code(tmp_path):
    source = PullSource()
    data = pr.fetch(source, REPO, 7, tmp_path)

    assert (data["author"], data["head_repo"], data["head_sha"]) == (
        "stranger",
        "stranger/repo",
        HEAD,
    )
    assert [f["path"] for f in data["files"]] == ["src/f0.py", "src/f1.py"]
    assert data["files"][0]["patch"].startswith("@@")
    assert data["code"] is True
    assert (pr.code_path(tmp_path, REPO, 7) / "src" / "f0.py").read_text() == "b\n"
    assert pr.load(tmp_path, REPO, 7) == data
    assert source.downloads == [f"/repos/{REPO}/tarball/{HEAD}"]  # by commit id, from the base repo


def test_the_code_is_fetched_again_only_when_the_head_moves(tmp_path):
    source = PullSource()
    pr.fetch(source, REPO, 7, tmp_path)
    pr.fetch(source, REPO, 7, tmp_path)
    assert len(source.downloads) == 1

    source.head = NEXT
    data = pr.fetch(source, REPO, 7, tmp_path)
    assert len(source.downloads) == 2
    assert data["head_sha"] == NEXT


def test_a_failed_code_download_still_gives_the_facts(tmp_path):
    source = PullSource()
    source.fail_download = True
    data = pr.fetch(source, REPO, 7, tmp_path)

    assert data["code"] is False
    assert not pr.code_path(tmp_path, REPO, 7).exists()
    assert pr.load(tmp_path, REPO, 7)["title"] == "Fix the thing"


def test_a_long_patch_is_cut_and_the_file_list_is_capped(tmp_path):
    source = PullSource(files=pr.MAX_FILES + 20)
    source.files[0]["patch"] = "x" * (pr.MAX_PATCH + 5)
    data = pr.fetch(source, REPO, 7, tmp_path)

    assert len(data["files"]) == pr.MAX_FILES
    assert len(data["files"][0]["patch"]) == pr.MAX_PATCH
    assert data["files"][0]["patch_cut"] is True


def test_a_head_that_is_not_a_commit_id_is_refused(tmp_path):
    source = PullSource()
    source.head = "../../etc"
    with pytest.raises(snapshot.SnapshotError):
        pr.fetch(source, REPO, 7, tmp_path)
    assert source.downloads == []


def test_only_the_recently_used_copies_stay(tmp_path):
    for number in range(1, pr.KEEP + 3):
        where = pr.folder(tmp_path, REPO, number)
        where.mkdir(parents=True)
        os.utime(where, (number, number))
    pr.prune(tmp_path, REPO)

    left = sorted(int(p.name) for p in pr.folder(tmp_path, REPO, 1).parent.iterdir())
    assert left == list(range(3, pr.KEEP + 3))


def test_load_of_a_missing_or_broken_file_is_none(tmp_path):
    assert pr.load(tmp_path, REPO, 9) is None
    where = pr.folder(tmp_path, REPO, 9)
    where.mkdir(parents=True)
    (where / "pr.json").write_text("{nope")
    assert pr.load(tmp_path, REPO, 9) is None


# -- the merged tree the checks run on -------------------------------------------------------

MERGE = "e" * 40


class MergeSource(PullSource):
    """GitHub answering ``mergeable`` with null a few times before it knows."""

    def __init__(self, mergeable, nulls: int = 0) -> None:
        super().__init__()
        self.mergeable, self.nulls, self.merge_sha = mergeable, nulls, MERGE

    def get_json(self, path):
        data = super().get_json(path)
        if path.endswith("/pulls/7"):
            data = dict(data)
            data["mergeable"] = None if self.nulls > 0 else self.mergeable
            data["merge_commit_sha"] = self.merge_sha
            self.nulls -= 1
        return data


def test_the_merged_tree_is_fetched_by_its_pinned_commit(tmp_path):
    source = MergeSource(True, nulls=2)
    waits: list[float] = []
    target = pr.fetch_merged(source, REPO, 7, tmp_path, sleep=waits.append)

    assert target == pr.folder(tmp_path, REPO, 7) / "merged"
    assert (target / "README.md").read_text() == "hi"
    assert source.downloads == [f"/repos/{REPO}/tarball/{MERGE}"]
    assert waits == [pr.MERGE_WAIT] * 2  # asked again while GitHub was still working it out


def test_the_merged_tree_is_kept_until_the_merge_commit_changes(tmp_path):
    source = MergeSource(True)
    pr.fetch_merged(source, REPO, 7, tmp_path)
    pr.fetch_merged(source, REPO, 7, tmp_path)
    assert len(source.downloads) == 1

    source.merge_sha = "f" * 40  # the base moved: a new merge commit
    pr.fetch_merged(source, REPO, 7, tmp_path)
    assert len(source.downloads) == 2


@pytest.mark.parametrize(
    ("mergeable", "nulls", "sha"),
    [(False, 0, MERGE), (None, 99, MERGE), (True, 0, ""), (True, 0, "../x")],
)
def test_no_merged_tree_when_it_does_not_merge_or_is_unknown(tmp_path, mergeable, nulls, sha):
    source = MergeSource(mergeable, nulls)
    source.merge_sha = sha
    assert pr.fetch_merged(source, REPO, 7, tmp_path, sleep=lambda _: None) is None
    assert source.downloads == []
