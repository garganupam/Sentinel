from .models import ReviewIssue, Severity

# Derived from Severity's own declaration order (Critical, Major, Minor) —
# single source of truth, so a reordered/extended enum updates this too.
_SEVERITY_RANK: dict[Severity, int] = {
    severity: rank for rank, severity in enumerate(Severity)
}


def aggregate_issues(issues: list[ReviewIssue]) -> list[ReviewIssue]:
    """Merge all agents' findings into one deduped, severity-sorted list.

    Locked policy (planning chat): dedupe by (file, line_number), keeping
    only the higher-severity report when two or more agents flag the same
    spot; sort Critical → Major → Minor.

    Args:
        issues: Combined issues from every agent in one flat list. This is
            what graph.py's `operator.add` reducer naturally produces once
            all 4 parallel agent branches finish — each branch contributes
            its own list, concatenated into one.

    Returns:
        Deduplicated issues, sorted Critical → Major → Minor. Empty list
        if every agent found nothing.
    """
    by_location: dict[tuple[str, int], ReviewIssue] = {}

    for issue in issues:
        key = (issue.file, issue.line_number)
        current = by_location.get(key)
        if current is None or _outranks(issue.severity, current.severity):
            by_location[key] = issue

    return sorted(by_location.values(), key=lambda issue: _SEVERITY_RANK[issue.severity])


def _outranks(a: Severity, b: Severity) -> bool:
    """True if severity `a` is more severe than `b` (Critical > Major > Minor)."""
    return _SEVERITY_RANK[a] < _SEVERITY_RANK[b]