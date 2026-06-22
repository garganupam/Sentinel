import logging
from typing import Literal

from app.github_client import GitHubClient

from .chunker import chunk_file
from .file_filter import should_index
from .store import VectorStore

logger = logging.getLogger("pr-reviewer")

ChangeStatus = Literal["added", "modified", "removed"]


def full_index(repo_full_name: str, ref: str, github: GitHubClient, store: VectorStore) -> None:
    """Index every file in the repo at `ref`, from scratch.

    Used for the first-time index (run once via full_index.py's CLI) and
    the weekly safety-net re-index — same logic either way, just triggered
    differently (manually once, vs on a schedule).

    Args:
        repo_full_name: "owner/repo".
        ref:            Branch, tag, or commit SHA to index at.
        github:         GitHubClient — lists files and fetches their content.
        store:          VectorStore for this repo.
    """
    paths = github.list_repo_files(repo_full_name, ref)
    logger.info("Full index: %d file(s) found in %s@%s", len(paths), repo_full_name, ref)

    indexed = sum(
        _index_one_file(repo_full_name, path, ref, github, store)
        for path in paths
    )
    logger.info("Full index complete: %d/%d file(s) indexed", indexed, len(paths))


def incremental_index(
    repo_full_name: str,
    ref: str,
    changed_paths: list[tuple[str, ChangeStatus]],
    github: GitHubClient,
    store: VectorStore,
) -> None:
    """Re-index only the files that changed in one push.

    Delete-and-re-insert per file, per locked design. Renames are NOT a
    distinct status here — whoever parses the push event into
    `changed_paths` is responsible for decomposing a rename into
    [(old_path, "removed"), (new_path, "added")] before calling this.

    Args:
        repo_full_name: "owner/repo".
        ref:            Commit SHA the push landed at.
        changed_paths:  (path, status) tuples — status is "added",
                        "modified", or "removed".
        github:         GitHubClient — fetches content for non-removed files.
        store:          VectorStore for this repo.
    """
    for path, status in changed_paths:
        if status == "removed":
            store.delete_file(path)
            logger.info("Removed %s from index", path)
        else:
            _index_one_file(repo_full_name, path, ref, github, store)


def _index_one_file(
    repo_full_name: str,
    path: str,
    ref: str,
    github: GitHubClient,
    store: VectorStore,
) -> bool:
    """Filter, fetch, chunk, and store one file. Returns True if indexed.

    Two filter passes: a cheap path-based check first (no API call needed
    — catches excluded dirs/filenames/extensions immediately), then a real
    size check after fetching, since should_index() needs actual byte size.
    A file that fails either check, or isn't readable as text, is skipped
    rather than raising — one bad file must never crash a full index.
    """
    pre_ok, pre_reason = should_index(path, file_size=0)
    if not pre_ok:
        logger.debug("Skipped %s: %s", path, pre_reason)
        return False

    content = github.get_file_content(repo_full_name, path, ref)
    if content is None:
        logger.debug("Skipped %s: unreadable as text", path)
        return False

    ok, reason = should_index(path, file_size=len(content.encode("utf-8")))
    if not ok:
        logger.info("Skipped %s: %s", path, reason)
        return False

    chunks = chunk_file(path, content, commit_sha=ref)
    store.upsert_file(chunks)
    return True