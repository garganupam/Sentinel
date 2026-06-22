import logging
from dataclasses import dataclass
from typing import Literal, Optional

from github import Auth, Github, GithubException

logger = logging.getLogger("pr-reviewer")


@dataclass(frozen=True)
class ChangedFile:
    """Represents a single file changed in a pull request.

    Attributes:
        filename:  Path of the file relative to the repo root e.g. "src/main.py"
        status:    One of "added" | "modified" | "removed" | "renamed"
        additions: Number of lines added
        deletions: Number of lines removed
        patch:     Unified diff string. None for binary files (images, etc.)
    """

    filename: str
    status: str
    additions: int
    deletions: int
    patch: Optional[str]


class GitHubClient:
    """Minimal GitHub access needed for M1: read PR files and post a comment.

    Usage:
        client = GitHubClient(token="ghp_...")
        files  = client.get_changed_files("owner/repo", pr_number=42)
        client.post_comment("owner/repo", pr_number=42, body="Hello from bot!")
    """

    def __init__(self, token: str) -> None:
        """Set up the GitHub client.

        No network call is made here — PyGithub connects lazily on first use.

        Args:
            token: A GitHub Personal Access Token (PAT) with:
                   - `repo` scope  → read files
                   - `pull_requests: write` → post comments
        """
        self._gh = Github(auth=Auth.Token(token))

    def get_changed_files(
        self,
        repo_full_name: str,
        pr_number: int,
    ) -> list[ChangedFile]:
        """Return a list of files changed in the given pull request.

        Args:
            repo_full_name: Repository in "owner/repo" format e.g. "garganupam/test"
            pr_number:      The pull request number.

        Returns:
            List of ChangedFile objects. Empty list if the PR has no changed files.

        Raises:
            github.GithubException: On API errors (bad token, repo not found, etc.)
        """
        repo = self._gh.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)

        return [
            ChangedFile(
                filename=f.filename,
                status=f.status,
                additions=f.additions,
                deletions=f.deletions,
                # Binary files (images, compiled artifacts) have no patch text.
                patch=getattr(f, "patch", None),
            )
            for f in pr.get_files()
        ]

    def post_comment(
        self,
        repo_full_name: str,
        pr_number: int,
        body: str,
    ) -> None:
        """Post a comment in the PR conversation (not a line-level review comment).

        Args:
            repo_full_name: Repository in "owner/repo" format.
            pr_number:      The pull request number.
            body:           Markdown-formatted comment text.

        Raises:
            github.GithubException: On API errors (bad token, no write access, etc.)
        """
        repo = self._gh.get_repo(repo_full_name)
        # GitHub models a PR as an Issue internally.
        # create_issue_comment() posts to the PR conversation timeline.
        repo.get_pull(pr_number).create_issue_comment(body)

    def list_repo_files(self, repo_full_name: str, ref: str) -> list[str]:
        """List every file path in the repo at a given ref.

        Uses the Git Trees API in one recursive call rather than walking
        directories one at a time — far fewer API calls, important for
        larger repos (walking directories individually throttles quickly).

        Args:
            repo_full_name: Repository in "owner/repo" format.
            ref:            Branch name, tag, or commit SHA.

        Returns:
            Repo-relative paths of every FILE in the tree (directories
            excluded — the tree's "blob" entries are files, "tree"
            entries are subdirectories).

        Raises:
            github.GithubException: On API errors (bad ref, repo not found, etc.)
        """
        repo = self._gh.get_repo(repo_full_name)
        tree = repo.get_git_tree(ref, recursive=True)
        return [item.path for item in tree.tree if item.type == "blob"]

    def get_file_content(self, repo_full_name: str, path: str, ref: str) -> Optional[str]:
        """Fetch one file's text content at a specific ref.

        Args:
            repo_full_name: Repository in "owner/repo" format.
            path:           Repo-relative file path.
            ref:            Branch name, tag, or commit SHA.

        Returns:
            Decoded UTF-8 text, or None if the file can't be read as text
            (binary content, or a GitHub API error — e.g. missing file,
            or a file too large for the Contents API). None is treated as
            "skip this file" by callers, not as a fatal error.
        """
        repo = self._gh.get_repo(repo_full_name)
        try:
            content_file = repo.get_contents(path, ref=ref)
            return content_file.decoded_content.decode("utf-8")
        except UnicodeDecodeError as exc:
            logger.debug("Skipping %s at %s: not valid UTF-8 (%s)", path, ref, exc)
            return None
        except GithubException as exc:
            # Catches rate limits, auth errors, file-too-large, missing file.
            # Logged at debug so transient problems (rate limits) are visible
            # when debugging without cluttering normal indexing output.
            logger.debug("Skipping %s at %s: %s", path, ref, exc)
            return None

    def set_commit_status(
        self,
        repo_full_name: str,
        sha: str,
        state: Literal["success", "failure", "error", "pending"],
        description: str,
        context: str = "pr-reviewer/sandbox",
    ) -> None:
        """Set a commit status on a specific SHA — shows in the PR merge box.

        This is the primary M4 reporting mechanism. The status appears
        natively in GitHub's merge box (green check / red X / grey circle)
        tied to the exact commit SHA that was tested, so the author always
        knows which code the verdict applies to.

        Args:
            repo_full_name: "owner/repo".
            sha:            The commit SHA to set the status on — always the
                            PR head SHA that was actually tested. Keying on
                            SHA means a stale status won't satisfy branch
                            protection for a new commit — GitHub handles this.
            state:          One of "success" | "failure" | "error" | "pending".
                              success → green check  (tests passed)
                              failure → red X        (tests failed / build failed)
                              error   → red X        (sandbox infrastructure error)
                              pending → grey circle  (run in progress)
            description:    Short human-readable summary shown in the merge box
                            e.g. "All 12 tests passed (3.2s)". Max ~140 chars.
            context:        Name of the status check. Shown in the merge box
                            and used to identify this check in branch protection
                            rules. Default: "pr-reviewer/sandbox".

        Raises:
            github.GithubException: On API errors (bad token, repo not found).
        """
        repo = self._gh.get_repo(repo_full_name)
        commit = repo.get_commit(sha)
        commit.create_status(
            state=state,
            description=description,
            context=context,
        )

    def approve_pull_request(
        self,
        repo_full_name: str,
        pr_number: int,
        body: str = "Sandbox: all tests passed. ✅",
    ) -> None:
        """Post an APPROVE review on a PR.

        NOT called by the pipeline. Present as a utility only.
        Auto-approval is intentionally disabled — a sandbox passing means
        tests passed, not that the code is correct, secure, or maintainable.
        The human reads the findings and status, then decides to approve
        and merge manually. Bot never approves. Bot never merges.

        Args:
            repo_full_name: "owner/repo".
            pr_number:      The pull request number.
            body:           Text shown with the approval review.

        Raises:
            github.GithubException: On API errors.
        """
        repo = self._gh.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        pr.create_review(body=body, event="APPROVE")