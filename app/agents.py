import logging

from pydantic import BaseModel

from .github_client import ChangedFile
from .llm_client import LLMClient
from .models import AgentType, ReviewIssue, Severity

logger = logging.getLogger("pr-reviewer")

# What each specialist looks for. Injected into the shared prompt below —
# this is the only thing that differs between the four agents.
_AGENT_FOCUS: dict[AgentType, str] = {
    AgentType.LOGIC: (
        "Algorithmic correctness only. Off-by-one errors, wrong operators, "
        "incorrect conditionals, unhandled edge cases, broken control flow. "
        "Ignore style and security — other agents cover those."
    ),
    AgentType.SECURITY: (
        "Security only. Injection risks (SQL / command / template), "
        "hardcoded secrets or credentials, insecure defaults, missing input "
        "validation, unsafe deserialization. Ignore style and pure logic bugs."
    ),
    AgentType.STYLE: (
        "Readability and convention only. Unclear naming, dead code, "
        "inconsistent formatting, missing docstrings on public functions. "
        "Ignore correctness and security — other agents cover those."
    ),
    AgentType.TESTS: (
        "Test coverage only. New or changed functions with no corresponding "
        "test, and existing tests that no longer match the new behaviour. "
        "Ignore implementation details outside test files."
    ),
}

# One-line definition of each severity level. A schema can guarantee the
# *value* is one of critical/major/minor — it can't teach the model *when*
# each applies. This is what actually calibrates judgment quality.
# "critical" is intentionally strict — it requires a traced proof, not a
# pattern match. The false-positive "infinite recursion" case (planning chat)
# was caused by pattern-matching "factorial + negative" without verifying
# whether the base case already handles it. Tightened to prevent recurrence.
_SEVERITY_CRITERIA = """Severity guide:
- critical: you have TRACED the execution path and PROVEN it breaks correctness,
  security, or causes a crash/data loss. A pattern match alone is NOT sufficient
  — you must show the actual code path that produces the failure.
- major: a real bug or risk, but not immediately catastrophic. Should fix.
- minor: style, naming, or a small improvement. Nice to fix, not urgent."""

_PROMPT_TEMPLATE = """You are a specialist code reviewer for a GitHub pull request.

Your ONLY focus for this review: {focus}

{criteria}

=== REFERENCE CONTEXT (background only — do NOT report issues here) ===
{context}
=== END REFERENCE CONTEXT ===

=== CODE UNDER REVIEW (the ONLY code you may report issues in) ===
Read the diff syntax carefully before reviewing:
- Lines starting with `+` are ADDED — this is the NEW code; review it.
- Lines starting with `-` are REMOVED — this is the OLD code being deleted;
  NEVER report an issue in a `-` line (it no longer exists in the codebase).
- Lines with no `+`/`-` prefix are unchanged context lines; they exist for
  orientation only — do not report issues in them unless they interact with
  a `+` line you are specifically flagging.

{diff}
=== END CODE UNDER REVIEW ===

Before reporting any issue, you MUST follow these grounding rules:
1. Only report issues in the NEW code — lines starting with `+` in the CODE
   UNDER REVIEW section. Never cite a `-` line, a context line, or anything
   from the REFERENCE CONTEXT as the location of an issue.
2. Cite the exact line number(s) your claim depends on. Include them in the comment.
3. For ANY claim about runtime behaviour (infinite recursion, crash, exception,
   data loss, infinite loop): trace the full control flow step by step in your
   reasoning before concluding. Check every branch and base case. A single
   pattern match is NOT sufficient evidence.
4. If you cannot point to a specific `+` line that proves the issue exists,
   do not report it. Uncertainty is not a finding.

Find issues only within your focus area above. Return an empty list if you find
nothing — do not invent issues to fill space."""


class _AgentIssue(BaseModel):
    """Shape Gemini must return for one issue.

    Mirrors ReviewIssue minus `agent` — that field is added on the Python
    side after the call, since each agent already knows its own type.
    `severity` reuses the same Severity enum as ReviewIssue, so the schema
    Gemini is constrained to and the type the rest of the app uses can
    never drift apart.
    """
    file: str
    line_number: int
    severity: Severity
    comment: str
    suggestion: str = ""


def run_agent(
    agent_type: AgentType,
    files: list[ChangedFile],
    llm: LLMClient,
    context: str = "",
) -> list[ReviewIssue]:
    """Run one specialist agent over a PR diff and return its findings.

    Each of the four agents (Logic / Security / Style / Tests) calls this
    same function — only `agent_type` changes what they're told to look for.
    Built to be called once per agent type, in parallel (graph.py wires this
    into LangGraph's Send API).

    Gemini's reply is constrained to `list[_AgentIssue]` via structured
    output (response_schema) — not parsed from free text. This guarantees
    valid JSON in the right shape at the API level; it does not guarantee
    good judgment, which is why `_SEVERITY_CRITERIA` is in the prompt too.

    Args:
        agent_type: Which specialist lens to apply.
        files:      Changed files from GitHubClient.get_changed_files().
        llm:        Shared Gemini client used to make the call.
        context:    M3 retrieved-context block from indexing/retriever.py
                    (e.g. "Relevant project context: ..." or "No project
                    context available."). Defaults to "" so this function
                    still works if ever called without it.

    Returns:
        Issues found by this agent. Empty list if none found, if there was
        no reviewable diff text (e.g. only binary files changed), or if the
        Gemini call/parse failed for any reason — every failure mode
        degrades to "no issues" rather than raising, so one bad agent never
        blocks the other three.
    """
    diff = _format_diff(files)
    if not diff:
        return []

    prompt = _PROMPT_TEMPLATE.format(
        focus=_AGENT_FOCUS[agent_type],
        criteria=_SEVERITY_CRITERIA,
        context=context,
        diff=diff,
    )

    try:
        raw_issues = llm.generate_structured(prompt, response_schema=list[_AgentIssue])
    except Exception as exc:
        # Broad on purpose: network/auth failures (APIError) and schema
        # validation failures surface as different exception types, and a
        # single bad agent call must never take down the whole review.
        logger.warning("%s agent: Gemini call failed (%s)", agent_type.value, exc)
        return []

    if not raw_issues:
        return []

    issues = [
        ReviewIssue(
            file=issue.file,
            line_number=issue.line_number,
            severity=issue.severity,
            agent=agent_type,
            comment=issue.comment,
            suggestion=issue.suggestion,
        )
        for issue in raw_issues
    ]

    # Verify-pass filter: drop findings that can't be grounded in the actual
    # diff. This is the structural defence against hallucinated findings —
    # a finding about a file not in the diff, or about a deleted line, is
    # silently dropped here before it can reach the PR comment.
    return _verify_pass_filter(issues, files)


def _format_diff(files: list[ChangedFile]) -> str:
    """Combine each file's patch into one diff block for the prompt.

    Binary files (patch=None) are skipped — there's no text diff to review.
    """
    blocks = [
        f"--- {f.filename} ({f.status}) ---\n{f.patch}"
        for f in files
        if f.patch is not None
    ]
    return "\n\n".join(blocks)


def _verify_pass_filter(
    issues: list[ReviewIssue],
    files: list[ChangedFile],
) -> list[ReviewIssue]:
    """Drop findings that can't be grounded in the actual diff.

    Two checks per finding:
    1. The file the issue cites must be a file actually in the diff.
       Catches: agent attributes a finding to a RAG context file that was
       never part of this PR.
    2. If line_number > 0, that line must be an added (+) line in the patch.
       Catches: agent flags a deleted (-) line as a current bug, or cites
       an unchanged context line as if it were new code.

    line_number == 0 (whole-file finding) passes check 2 automatically —
    "whole file" is a valid scope when the file IS in the diff.

    Findings that fail either check are logged and dropped. This runs after
    the LLM call, so it's a structural safety net, not a prompt instruction.
    """
    diff_files: set[str] = set()
    added_by_file: dict[str, set[int]] = {}

    for f in files:
        if f.patch is None:
            continue
        diff_files.add(f.filename)
        added_by_file[f.filename] = _extract_added_line_numbers(f.patch)

    kept = []
    for issue in issues:
        # Check 1: file must be in the diff
        if issue.file not in diff_files:
            logger.info(
                "verify-pass: dropped %s:%d — file not in diff (RAG context leak?)",
                issue.file, issue.line_number,
            )
            continue

        # Check 2: line_number must be an added line (0 = whole-file, always passes)
        if issue.line_number > 0:
            if issue.line_number not in added_by_file.get(issue.file, set()):
                logger.info(
                    "verify-pass: dropped %s:%d — line not in added lines "
                    "(deleted or unchanged context line)",
                    issue.file, issue.line_number,
                )
                continue

        kept.append(issue)

    if len(kept) < len(issues):
        logger.info(
            "verify-pass: kept %d/%d findings after filter",
            len(kept), len(issues),
        )

    return kept


def _extract_added_line_numbers(patch: str) -> set[int]:
    """Parse a unified diff patch and return the new-file line numbers of + lines.

    Unified diff format:
      @@ -old_start,old_count +new_start,new_count @@
      <space> unchanged context line  (counts in new file)
      - removed line                  (counts in old file only)
      + added line                    (counts in new file — these are what we want)

    We track the running new-file line number by:
      - Resetting at each @@ hunk header to new_start.
      - Incrementing for `+` lines and context lines.
      - Not incrementing for `-` lines (they don't exist in the new file).
    """
    import re
    added: set[int] = set()
    current_line = 0

    for line in patch.splitlines():
        if line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            if m:
                # new_start is where this hunk begins in the new file.
                # We subtract 1 because the first real line will increment it.
                current_line = int(m.group(1)) - 1
        elif line.startswith("+++") or line.startswith("---"):
            # File header lines — skip, don't affect line counting.
            continue
        elif line.startswith("+"):
            current_line += 1
            added.add(current_line)
        elif line.startswith("-"):
            # Removed line — exists in old file only, don't increment new counter.
            pass
        else:
            # Unchanged context line — exists in both files, increment new counter.
            current_line += 1

    return added