from dataclasses import dataclass
from enum import Enum


class Severity(str, Enum):
    """How serious the issue is.

    Used for sorting (Critical first) and deciding whether to enter the fix loop.
    Inherits str so it serialises cleanly to/from JSON.
    """
    CRITICAL = "critical"   # Must fix — correctness or security broken
    MAJOR    = "major"      # Should fix — significant bug or risk
    MINOR    = "minor"      # Nice to fix — style, naming, small improvement


class AgentType(str, Enum):
    """Which specialist agent raised this issue.

    Used in the aggregator to track source and avoid cross-agent dupes.
    """
    LOGIC    = "logic"      # Algorithmic correctness, edge cases, off-by-ones
    SECURITY = "security"   # Injections, hardcoded secrets, insecure defaults
    STYLE    = "style"      # Naming, formatting, dead code, readability
    TESTS    = "tests"      # Missing or weak test coverage


@dataclass
class ReviewIssue:
    """One concrete problem found in the PR diff.

    Every agent produces a list of these.
    The aggregator merges and deduplicates them before posting to GitHub.

    Attributes:
        file:        Repo-relative file path e.g. "app/auth.py"
        line_number: Approximate line in the diff where the issue lives.
                     0 means the whole file (used when a line can't be pinpointed).
        severity:    How serious the problem is (Critical / Major / Minor).
        agent:       Which agent found it.
        comment:     Human-readable description of the problem.
        suggestion:  Concrete fix — what to write instead. Empty string if none.
    """
    file:        str
    line_number: int
    severity:    Severity
    agent:       AgentType
    comment:     str
    suggestion:  str = ""