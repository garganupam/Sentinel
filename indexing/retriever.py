"""RAG retrieval pipeline — §4.2 of the M3 spec.

Full pipeline per query:
  1. Build a rich query from changed function/class code + extracted identifiers.
  2. Retrieve a generous candidate pool (~20) from the vector store.
  3. Remove redundant context by line-range overlap (not file count).
  4. Filter by relevance threshold — drop weak candidates.
  5. Pack highest-relevance chunks until the token budget is exhausted.
"""
import logging
import re
import time
from typing import Optional

from app.github_client import ChangedFile

from .models import QueryMatch
from .store import VectorStore

logger = logging.getLogger("pr-reviewer")

# --- Tunable knobs (§9 — safe to adjust based on eval results) ---
_CANDIDATE_POOL     = 20    # how many candidates to fetch per query
_SIMILARITY_THRESHOLD = 0.40  # drop chunks below this similarity score
_OVERLAP_THRESHOLD  = 0.50  # fraction of shorter chunk's range that triggers dedup
_TOKEN_BUDGET       = 3000  # total tokens across the entire context block
_NO_CONTEXT_MESSAGE = "No project context available."

# Patterns used to extract function/class names from diff lines —
# provides a high-signal semantic anchor alongside the full added code.
_DEF_PATTERNS = [
    re.compile(r"^\s*(?:async\s+)?def\s+(\w+)"),                     # Python function
    re.compile(r"^\s*class\s+(\w+)"),                                 # Python class
    re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)"),  # JS/TS function
    re.compile(r"^\s*(?:pub(?:\s+\w+)?\s+)?fn\s+(\w+)"),             # Rust function
    re.compile(r"^\s*func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)"),         # Go function/method
    re.compile(r"^\s*(?:public|private|protected|static|\s)*\w+\s+(\w+)\s*\("), # Java-ish
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def retrieve_context(
    files: list[ChangedFile],
    store: VectorStore,
    rag_debug: bool = False,
) -> str:
    """Build a shared project-context block for all 4 review agents.

    Locked hard rule (M3 is purely additive): any failure here — empty index,
    zero hits, exception — degrades gracefully to _NO_CONTEXT_MESSAGE.
    The review always continues.

    Args:
        files:     Changed files from GitHubClient.get_changed_files().
        store:     VectorStore for the repo being reviewed.
        rag_debug: When True, additionally logs the exact context block that
                   will be sent to the agents. Set via Settings.rag_debug
                   (RAG_DEBUG=true in .env). Dev only — noisy in production.

    Returns:
        Formatted context text, or _NO_CONTEXT_MESSAGE on any failure.
    """
    try:
        return _run_pipeline(files, store, rag_debug)
    except Exception:
        logger.exception("Context retrieval failed — continuing without project context")
        return _NO_CONTEXT_MESSAGE


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(
    files: list[ChangedFile],
    store: VectorStore,
    rag_debug: bool,
) -> str:
    t_start = time.monotonic()

    # Step 1 — collect candidates across all changed files
    candidates: list[QueryMatch] = []
    for f in files:
        if f.patch is None:
            continue
        query = _build_query(f.patch)
        if not query:
            continue
        batch = store.query(query, n_results=_CANDIDATE_POOL, exclude_file=f.filename)
        candidates.extend(batch)

    total_candidates = len(candidates)

    if not candidates:
        logger.info("[RAG] No candidates retrieved (index empty or no usable diff)")
        return _NO_CONTEXT_MESSAGE

    # Step 2 — deduplicate by line-range content overlap within each file
    candidates = _dedup_by_overlap(candidates)

    # Step 3 — filter by relevance threshold (drop weak candidates)
    candidates = [m for m in candidates if (1 - m.distance) >= _SIMILARITY_THRESHOLD]

    # Step 4 — sort by descending similarity, pack within token budget
    candidates.sort(key=lambda m: m.distance)   # ascending distance = descending similarity
    kept, total_tokens = _pack_to_budget(candidates)

    elapsed_ms = int((time.monotonic() - t_start) * 1000)

    # § 4.3 — Structured metadata log (always, regardless of rag_debug)
    logger.info(
        "[RAG] Retrieved (%d kept / %d candidates, %d ms):",
        len(kept), total_candidates, elapsed_ms,
    )
    for m in kept:
        logger.info(
            "  %-42s lines %d-%d   score=%.2f",
            m.file_path, m.start_line, m.end_line, 1 - m.distance,
        )
    logger.info(
        "[RAG] Context: %d chunks, ~%d tokens (budget %d)",
        len(kept), total_tokens, _TOKEN_BUDGET,
    )

    if not kept:
        return _NO_CONTEXT_MESSAGE

    context = _format_context(kept)

    # § 4.3 RAG_DEBUG — additionally dump the exact context block for dev inspection
    if rag_debug:
        logger.debug(
            "[RAG] DEBUG — exact context block sent to agents:\n%s", context
        )

    return context


# ---------------------------------------------------------------------------
# Step 1 helper — query construction
# ---------------------------------------------------------------------------

def _build_query(patch: str) -> Optional[str]:
    """Build a rich semantic query from one file's diff patch.

    §4.2 requirement: query must represent the *changed code*, not a label.
    Strategy:
      - Extract function/class names from added lines → high-signal identifiers
        that anchor the embedding in semantic space.
      - Append the full added-code text → provides full context.
      - Never return a bare filename, branch name, or label in isolation
        (e.g. "bugg") — that's the exact failure mode the spec calls out.

    Returns:
        A query string, or None if the patch has no usable added code.
    """
    added_lines = [
        line[1:]   # strip the leading "+"
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]

    if not added_lines:
        return None

    added_text = "\n".join(added_lines).strip()
    if not added_text:
        return None

    # Extract function/class identifiers as semantic anchors.
    # Prepend them so the embedding sees the most distinctive signal first.
    identifiers: list[str] = []
    seen: set[str] = set()
    for line in added_lines:
        for pattern in _DEF_PATTERNS:
            m = pattern.match(line)
            if m:
                name = m.group(1)
                if name not in seen:
                    identifiers.append(name)
                    seen.add(name)
                break   # one pattern match per line is enough

    if identifiers:
        # Example result: "factorial compute_result\ndef factorial(n):\n  ..."
        return f"{' '.join(identifiers)}\n{added_text}"

    return added_text


# ---------------------------------------------------------------------------
# Step 2 helper — content-overlap deduplication
# ---------------------------------------------------------------------------

def _dedup_by_overlap(candidates: list[QueryMatch]) -> list[QueryMatch]:
    """Remove redundant chunks based on line-range overlap within the same file.

    §4.2: two chunks from the same file with heavily overlapping line ranges
    carry the same information → keep the higher-scoring one.
    But two chunks from the same file covering *different* line ranges are
    different concepts → keep both if relevant.

    Critically: NO per-file chunk cap. A file with 3 non-overlapping relevant
    sections keeps all 3; a file with 4 near-identical chunks of the same
    section keeps only 1.

    Algorithm: greedy — for each file, sort by ascending distance (best first),
    then add each candidate only if it doesn't overlap significantly with
    any already-kept chunk from that file.
    """
    by_file: dict[str, list[QueryMatch]] = {}
    for m in candidates:
        by_file.setdefault(m.file_path, []).append(m)

    kept: list[QueryMatch] = []
    for file_chunks in by_file.values():
        file_chunks.sort(key=lambda m: m.distance)  # best first within file
        file_kept: list[QueryMatch] = []
        for candidate in file_chunks:
            overlaps = any(
                _overlap_fraction(candidate, existing) >= _OVERLAP_THRESHOLD
                for existing in file_kept
            )
            if not overlaps:
                file_kept.append(candidate)
        kept.extend(file_kept)

    return kept


def _overlap_fraction(a: QueryMatch, b: QueryMatch) -> float:
    """Fraction of the shorter chunk's line range covered by the overlap.

    Uses intersection / min(len_a, len_b) — more conservative than IoU.
    A result of 1.0 means the shorter chunk is fully contained in the longer.
    A result of 0.0 means no overlap at all.
    """
    overlap_start = max(a.start_line, b.start_line)
    overlap_end   = min(a.end_line,   b.end_line)
    if overlap_end < overlap_start:
        return 0.0
    overlap = overlap_end - overlap_start + 1
    len_a   = max(1, a.end_line - a.start_line + 1)
    len_b   = max(1, b.end_line - b.start_line + 1)
    return overlap / min(len_a, len_b)


# ---------------------------------------------------------------------------
# Step 4 helper — token-budget packing
# ---------------------------------------------------------------------------

def _pack_to_budget(
    candidates: list[QueryMatch],
) -> tuple[list[QueryMatch], int]:
    """Pack highest-relevance chunks until the token budget is exhausted.

    §4.2: the token budget — not a fixed chunk count, and not a per-file cap
    — is the sole governing constraint. Small repos: most/all relevant chunks
    fit. Large repos: only the highest-value chunks survive. Same mechanism
    regardless of repo size.

    Candidates must already be sorted by ascending distance (best first).
    Token estimate: len(content) // 4 (rough ~4 chars/token, sufficient for
    a budget gate — not a substitute for a real tokenizer).
    """
    kept: list[QueryMatch] = []
    total_tokens = 0
    for m in candidates:
        chunk_tokens = max(1, len(m.content) // 4)
        if total_tokens + chunk_tokens > _TOKEN_BUDGET:
            break
        kept.append(m)
        total_tokens += chunk_tokens
    return kept, total_tokens


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format_context(chunks: list[QueryMatch]) -> str:
    """Format the kept chunks into one context block for the agent prompts."""
    lines = ["Relevant project context (most similar first):"]
    for m in chunks:
        lines.append(f"\n--- {m.file_path}  (lines {m.start_line}–{m.end_line}) ---")
        lines.append(m.content)
    return "\n".join(lines)