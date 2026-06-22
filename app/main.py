import json
import logging
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, Header, Request, Response, status

from indexing.indexer import incremental_index
from indexing.store import VectorStore
from sandbox.runner import SandboxResult, run_sandbox

from .config import Settings
from .github_client import GitHubClient
from .llm_client import LLMClient
from .reviewer import review_pull_request
from .security import verify_signature
from .webhook import is_actionable, parse_pull_request_event, parse_push_event

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("pr-reviewer")

# ---------------------------------------------------------------------------
# App + startup — fail fast if env vars are missing
# ---------------------------------------------------------------------------

app = FastAPI(title="PR Reviewer", version="0.1.0")

settings = Settings.load()
github  = GitHubClient(settings.github_token)
llm     = LLMClient(settings.gemini_api_key)


# ---------------------------------------------------------------------------
# Sandbox → commit status mapping
# ---------------------------------------------------------------------------

def _commit_status(r: SandboxResult) -> tuple:
    """Map a SandboxResult to (state, description) for set_commit_status().

    GitHub commit status states:
      success → green ✅  failure → red ❌  error → red ❌ (infra)
    Description is capped at 140 chars (GitHub limit).

    Decision table from M4 spec:
      tests passed          → success
      tests failed          → failure
      build/install failed  → failure
      no tests / unsupported→ error  (neutral — unverified)
      timeout / infra error → error  (neutral — sandbox couldn't run)
    """
    dur = f" ({r.duration_seconds:.1f}s)" if r.duration_seconds else ""

    if r.passed:
        return "success", f"All tests passed{dur}"[:140]
    if r.build_failed:
        return "failure", "Build/install failed"
    if r.had_tests:
        return "failure", f"Tests failed{dur}"[:140]
    if r.timed_out:
        return "error", "Sandbox timed out — no verdict"
    if r.error:
        return "error", f"Sandbox error — no verdict"
    return "error", "No tests found — no verdict"


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------

def _process_pull_request(
    repo_full_name: str,
    pr_number: int,
    head_sha: str,
) -> None:
    """Full M4 pipeline for one PR: review → sandbox → report.

    Flow:
      1. Set 'pending' status immediately → merge box shows "in progress"
      2. Fetch changed files
      3. Run Docker sandbox (build + tests at head_sha)
      4. Run advisory review agents (M2 + M3 RAG context)
      5. Post combined comment (findings + sandbox verdict)
      6. Set final commit status (success / failure / error)

    On any unexpected exception: log it, set 'error' status so the merge
    box never stays 'pending' indefinitely, and exit cleanly.
    """
    try:
        logger.info(
            "Processing PR #%d in %s (sha=%s)",
            pr_number, repo_full_name, head_sha[:8],
        )

        # Step 1 — mark in-progress immediately
        github.set_commit_status(
            repo_full_name, head_sha,
            state="pending",
            description="PR Reviewer: review + sandbox in progress…",
        )

        # Step 2 — fetch diff
        files = github.get_changed_files(repo_full_name, pr_number)
        logger.info("Fetched %d changed file(s)", len(files))

        # Step 3 — sandbox (runs before review so its result goes into the comment)
        logger.info("Running sandbox for %s@%s", repo_full_name, head_sha[:8])
        sandbox_result = run_sandbox(repo_full_name, head_sha, settings.github_token)

        # Step 4 — advisory review (M2 agents + M3 RAG)
        store   = VectorStore(settings.chroma_persist_path, repo_full_name)
        comment = review_pull_request(
            files, llm, store,
            rag_debug=settings.rag_debug,
            sandbox_result=sandbox_result,
        )

        # Step 5 — post combined comment
        github.post_comment(repo_full_name, pr_number, comment)
        logger.info("Posted review comment on PR #%d", pr_number)

        # Step 6 — set final commit status
        state, description = _commit_status(sandbox_result)
        github.set_commit_status(repo_full_name, head_sha, state, description)
        logger.info(
            "Commit status set: %s — %s (sha=%s)",
            state, description, head_sha[:8],
        )

    except Exception:
        logger.exception(
            "Unexpected error processing PR #%d in %s", pr_number, repo_full_name
        )
        # Best-effort: clear the 'pending' state so merge box doesn't hang.
        try:
            github.set_commit_status(
                repo_full_name, head_sha,
                state="error",
                description="PR Reviewer: pipeline failed — check server logs",
            )
        except Exception:
            pass


def _process_push(
    repo_full_name: str,
    commit_sha: str,
    changed_paths: list,
) -> None:
    """Incrementally re-index files changed in one push to the default branch."""
    try:
        logger.info(
            "Incremental index: %d file(s) changed in %s@%s",
            len(changed_paths), repo_full_name, commit_sha,
        )
        store = VectorStore(settings.chroma_persist_path, repo_full_name)
        incremental_index(repo_full_name, commit_sha, changed_paths, github, store)
        logger.info("Incremental index complete for %s@%s", repo_full_name, commit_sha)

    except Exception:
        logger.exception(
            "Failed incremental index for %s@%s", repo_full_name, commit_sha
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Simple liveness check."""
    return {"status": "ok"}


@app.post("/webhook")
async def webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: Optional[str] = Header(default=None),
    x_github_event: Optional[str] = Header(default=None),
) -> Response:
    """Receive and process a GitHub webhook event.

    Handles:
      pull_request → advisory review + sandbox + commit status
      push (default branch) → incremental RAG re-index
    Everything else is acknowledged and ignored.
    """
    body = await request.body()

    if not verify_signature(body, settings.webhook_secret, x_hub_signature_256):
        logger.warning("Rejected webhook: invalid or missing signature")
        return Response(content="Invalid signature.", status_code=status.HTTP_401_UNAUTHORIZED)

    if x_github_event == "pull_request":
        return _handle_pull_request_event(body, background_tasks)
    elif x_github_event == "push":
        return _handle_push_event(body, background_tasks)
    else:
        logger.info("Ignored event: %s", x_github_event)
        return Response(content="Event ignored.", status_code=status.HTTP_200_OK)


def _handle_pull_request_event(body: bytes, background_tasks: BackgroundTasks) -> Response:
    """Parse a pull_request payload and queue the full M4 pipeline."""
    try:
        payload = json.loads(body)
        event = parse_pull_request_event(payload)
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Malformed pull_request payload: %s", exc)
        return Response(content="Malformed payload.", status_code=status.HTTP_400_BAD_REQUEST)

    if not is_actionable(event.action):
        logger.info("Ignored PR action: %s", event.action)
        return Response(content="Action ignored.", status_code=status.HTTP_200_OK)

    logger.info(
        "Queuing review for PR #%d in %s (action=%s, fork=%s, sha=%s)",
        event.pr_number, event.repo_full_name,
        event.action, event.is_fork, event.head_sha[:8],
    )
    background_tasks.add_task(
        _process_pull_request,
        repo_full_name=event.repo_full_name,
        pr_number=event.pr_number,
        head_sha=event.head_sha,
    )
    return Response(content="Accepted.", status_code=status.HTTP_202_ACCEPTED)


def _handle_push_event(body: bytes, background_tasks: BackgroundTasks) -> Response:
    """Parse a push payload and queue incremental indexing for default-branch pushes."""
    try:
        payload = json.loads(body)
        event = parse_push_event(payload)
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Malformed push payload: %s", exc)
        return Response(content="Malformed payload.", status_code=status.HTTP_400_BAD_REQUEST)

    if not event.is_default_branch:
        logger.info("Ignored push to non-default ref: %s", event.ref)
        return Response(content="Push ignored (not default branch).", status_code=status.HTTP_200_OK)

    if not event.changed_paths:
        logger.info("No file changes in push — nothing to index")
        return Response(content="No changes to index.", status_code=status.HTTP_200_OK)

    logger.info(
        "Queuing incremental index for %s@%s (%d file(s) changed)",
        event.repo_full_name, event.commit_sha, len(event.changed_paths),
    )
    background_tasks.add_task(
        _process_push,
        repo_full_name=event.repo_full_name,
        commit_sha=event.commit_sha,
        changed_paths=event.changed_paths,
    )
    return Response(content="Accepted.", status_code=status.HTTP_202_ACCEPTED)