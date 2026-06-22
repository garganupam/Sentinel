from dataclasses import dataclass

# The only PR actions that should trigger a review run.
# "opened"      → brand new PR
# "reopened"    → previously closed PR re-opened
# "synchronize" → new commit pushed to an existing PR
ACTIONABLE_ACTIONS = {"opened", "reopened", "synchronize"}


@dataclass(frozen=True)
class PullRequestEvent:
    """Represents the fields we care about from a GitHub pull_request event.

    Attributes:
        action:         GitHub action string e.g. "opened", "closed", "synchronize"
        repo_full_name: The BASE repo where the PR is opened e.g. "owner/repo"
        pr_number:      Pull request number
        head_sha:       The latest commit SHA on the PR branch (used to post review)
        is_fork:        True if the PR comes from a forked repo
                        (important later: fork PRs can't receive direct commits)
    """
    action: str
    repo_full_name: str
    pr_number: int
    head_sha: str
    is_fork: bool


def parse_pull_request_event(payload: dict) -> PullRequestEvent:
    """Parse a raw GitHub 'pull_request' webhook payload into a PullRequestEvent.

    Args:
        payload: The parsed JSON body of the GitHub webhook request.

    Returns:
        A PullRequestEvent with the fields needed for M1.

    Raises:
        KeyError:   If a required field is missing from the payload.
        TypeError:  If the payload structure is not a well-formed PR event.

    Note:
        The caller should catch these exceptions and return HTTP 400.
    """
    pr = payload["pull_request"]
    head = pr["head"]
    base = pr["base"]

    base_repo = base["repo"]["full_name"]

    # head.repo can be None in rare cases (e.g. the fork was deleted).
    # We treat that as a fork to be safe.
    head_repo = (head.get("repo") or {}).get("full_name")
    is_fork = head_repo != base_repo

    return PullRequestEvent(
        action=payload["action"],
        repo_full_name=base_repo,
        pr_number=int(payload["number"]),
        head_sha=head["sha"],
        is_fork=is_fork,
    )


def is_actionable(action: str) -> bool:
    """Return True if this PR action should trigger a review run."""
    return action in ACTIONABLE_ACTIONS


@dataclass(frozen=True)
class PushEvent:
    """Represents the fields needed from a GitHub 'push' webhook payload.

    Attributes:
        ref:                Full ref pushed to, e.g. "refs/heads/main".
        repo_full_name:     "owner/repo".
        is_default_branch:  True if this push landed on the repo's default
                            branch — locked design: only default-branch
                            pushes trigger incremental indexing. Also False
                            for a branch-deletion push, even if the ref
                            happens to match (defensive — a deleted ref has
                            nothing real to index).
        commit_sha:         The SHA the push landed at (payload["after"]) —
                            stamped onto every re-indexed chunk.
        changed_paths:      (path, status) tuples — status is "added",
                            "modified", or "removed". Net effect across
                            every commit in this push, ready to pass
                            straight into indexer.incremental_index().
    """
    ref: str
    repo_full_name: str
    is_default_branch: bool
    commit_sha: str
    changed_paths: list[tuple[str, str]]


def parse_push_event(payload: dict) -> PushEvent:
    """Parse a raw GitHub 'push' webhook payload into a PushEvent.

    Args:
        payload: The parsed JSON body of the GitHub webhook request.

    Returns:
        A PushEvent with the fields needed to trigger incremental indexing.

    Raises:
        KeyError:  If a required field is missing from the payload.
        TypeError: If the payload structure is not a well-formed push event.

    Note:
        The caller should catch these exceptions and return HTTP 400.
    """
    ref = payload["ref"]
    repo_full_name = payload["repository"]["full_name"]
    default_branch = payload["repository"]["default_branch"]
    is_deleted = bool(payload.get("deleted", False))
    is_default_branch = (ref == f"refs/heads/{default_branch}") and not is_deleted

    # Aggregate file changes across every commit in the push. Iterating in
    # order and overwriting per path means the final status reflects the
    # net effect by the end of the push — e.g. added-then-modified still
    # nets to "needs re-index"; removed-then-re-added nets to "added".
    changed: dict[str, str] = {}
    if not is_deleted:
        for commit in payload.get("commits", []):
            for path in commit.get("added", []):
                changed[path] = "added"
            for path in commit.get("modified", []):
                changed[path] = "modified"
            for path in commit.get("removed", []):
                changed[path] = "removed"

    return PushEvent(
        ref=ref,
        repo_full_name=repo_full_name,
        is_default_branch=is_default_branch,
        commit_sha=payload["after"],
        changed_paths=list(changed.items()),
    )