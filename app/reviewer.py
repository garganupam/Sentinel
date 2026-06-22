from typing import Optional

from indexing.store import VectorStore
from sandbox.runner import SandboxResult

from .github_client import ChangedFile
from .graph import build_review_graph
from .llm_client import LLMClient
from .models import ReviewIssue, Severity


def review_pull_request(
    files: list[ChangedFile],
    llm: LLMClient,
    store: VectorStore,
    rag_debug: bool = False,
    sandbox_result: Optional[SandboxResult] = None,
) -> str:
    """Run the full review pipeline and return a formatted PR comment.

    Combines two sections:
    1. Advisory review — findings from 4 specialist agents (Logic, Security,
       Style, Tests). Advisory only: findings are informational, they never
       block a merge.
    2. Sandbox verdict — build + test result from the Docker sandbox (M4).
       This is the authority for the commit status check. Optional: when
       sandbox_result is None (M3 and earlier, or sandbox unavailable), only
       the review section is shown.

    Args:
        files:          Changed files from GitHubClient.get_changed_files().
        llm:            Shared Gemini client.
        store:          VectorStore for the repo — used by the retrieval node.
        rag_debug:      When True, logs the full context block sent to agents.
        sandbox_result: Result from sandbox.runner.run_sandbox(). None = omit
                        the sandbox section (advisory-only mode).

    Returns:
        GitHub-ready markdown comment combining both sections.
    """
    graph = build_review_graph(llm, store, rag_debug)
    result = graph.invoke({"files": files, "context": "", "issues": [], "aggregated": []})
    issues = result["aggregated"]

    sections = [_format_review_section(issues)]

    if sandbox_result is not None:
        sections.append(_format_sandbox_section(sandbox_result))

    sections.append(
        "_Bot reports. Human decides — advisory findings are informational only; "
        "the sandbox verdict is the authority for the status check._"
    )
    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Section 1 — Advisory review
# ---------------------------------------------------------------------------

_SEVERITY_EMOJI = {
    Severity.CRITICAL: "🔴",
    Severity.MAJOR:    "🟠",
    Severity.MINOR:    "🟡",
}


def _format_review_section(issues: list[ReviewIssue]) -> str:
    """Format the advisory review findings — grouped by severity."""
    if not issues:
        return (
            "## 🔍 Advisory Review\n\n"
            "No issues found across Logic, Security, Style, and Tests agents. ✅"
        )

    # Group by severity — aggregator already sorted, so order is preserved.
    groups: dict[Severity, list[ReviewIssue]] = {
        Severity.CRITICAL: [],
        Severity.MAJOR:    [],
        Severity.MINOR:    [],
    }
    for issue in issues:
        groups[issue.severity].append(issue)

    lines = [f"## 🔍 Advisory Review — {len(issues)} issue(s) found", ""]

    for severity, severity_issues in groups.items():
        if not severity_issues:
            continue

        emoji = _SEVERITY_EMOJI[severity]
        lines.append(f"### {emoji} {severity.value.capitalize()} ({len(severity_issues)})")
        lines.append("")

        for issue in severity_issues:
            location = f"line {issue.line_number}" if issue.line_number else "whole file"
            lines.append(f"**`{issue.file}`** · {location} · *{issue.agent.value}*")
            lines.append(f"> {issue.comment}")
            if issue.suggestion:
                lines.append(f"```\n{issue.suggestion}\n```")
            lines.append("")

    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Section 2 — Sandbox verdict
# ---------------------------------------------------------------------------

_MAX_LOG_LINES = 50   # last N lines shown in PR comment; full log stays in server logs


def _trim_logs(logs: str) -> str:
    """Keep only the last N lines of sandbox output.

    Failure info (assertion errors, tracebacks) is always at the bottom of
    pytest output. The first N lines are usually install noise. Trimming
    prevents a 5000-line log from making the PR comment unreadable.
    """
    lines = logs.strip().splitlines()
    if not lines:
        return ""
    if len(lines) <= _MAX_LOG_LINES:
        return "\n".join(lines)
    trimmed = lines[-_MAX_LOG_LINES:]
    return (
        f"... (showing last {_MAX_LOG_LINES} of {len(lines)} lines)\n\n"
        + "\n".join(trimmed)
    )


def _format_sandbox_section(r: SandboxResult) -> str:
    """Format the sandbox build+test result — all 5 outcomes from the spec."""
    sha_short = r.tested_sha[:8] if r.tested_sha else "unknown"
    duration = f"{r.duration_seconds:.1f}s" if r.duration_seconds else ""

    if r.error and not r.had_tests and not r.build_failed:
        return (
            f"## 🧪 Sandbox — ⚠️ Unverified\n\n"
            f"Sandbox could not run (SHA `{sha_short}`).\n"
            f"> {r.error}\n\n"
            f"No verdict — author decides whether to merge."
        )

    if r.build_failed:
        log_block = f"\n\n```\n{_trim_logs(r.logs)}\n```" if r.logs.strip() else ""
        return (
            f"## 🧪 Sandbox — ❌ Build Failed\n\n"
            f"Dependency installation failed on SHA `{sha_short}`."
            f"{log_block}"
        )

    if not r.had_tests:
        return (
            f"## 🧪 Sandbox — ⚠️ Unverified\n\n"
            f"No tests found for SHA `{sha_short}`. "
            f"Build succeeded but nothing was verified.\n\n"
            f"No verdict — author decides whether to merge."
        )

    if r.timed_out:
        return (
            f"## 🧪 Sandbox — ⚠️ Timed Out\n\n"
            f"Test run exceeded the time limit on SHA `{sha_short}`. "
            f"No verdict — author decides whether to merge."
        )

    if r.passed:
        timing = f" ({duration})" if duration else ""
        return (
            f"## 🧪 Sandbox — ✅ Passed\n\n"
            f"All tests passed{timing} on SHA `{sha_short}`.\n\n"
            f"The author may now merge at their discretion."
        )

    # Tests ran and failed.
    log_block = f"\n\n```\n{_trim_logs(r.logs)}\n```" if r.logs.strip() else ""
    timing = f" ({duration})" if duration else ""
    return (
        f"## 🧪 Sandbox — ❌ Tests Failed\n\n"
        f"Tests failed{timing} on SHA `{sha_short}`."
        f"{log_block}"
    )