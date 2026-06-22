import ast
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional,Tuple
from typing import List
from .models import Chunk, ChunkType, make_chunk_id

_MARKDOWN_EXTENSIONS = {".md", ".mdx", ".rst"}
_CONFIG_EXTENSIONS = {".yaml", ".yml", ".toml", ".ini", ".cfg"}

_LANGUAGE_BY_EXTENSION = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".jsx": "javascript", ".tsx": "typescript",
    ".go": "go", ".java": "java", ".rb": "ruby", ".rs": "rust",
    ".md": "markdown", ".mdx": "markdown", ".rst": "markdown", ".txt": "text",
    ".yaml": "config", ".yml": "config", ".toml": "config", ".ini": "config", ".cfg": "config",
    ".html": "html", ".css": "css",
}

_MAX_CHUNK_TOKENS = 1500
_MIN_CHUNK_TOKENS = 50

# Generic, loose pattern used only when there's no real parser for the
# language (or Python's AST failed) — catches def/class/function/func
# across most C-like and Python-like syntax.
_GENERIC_DEF_PATTERN = re.compile(r"^\s*(def|class|function|func)\s+\w+")


def chunk_file(file_path: str, content: str, commit_sha: str) -> list[Chunk]:
    """Split one file's content into semantically-meaningful Chunks.

    Dispatches by extension: Python gets AST-based chunking (regex, then
    fixed-size, if ast.parse fails); Markdown/.rst gets heading-based
    chunking; config files are kept whole; everything else gets the same
    regex-then-fixed-size fallback used for broken Python.

    Args:
        file_path:  Repo-relative path — drives language detection and
                    each chunk's deterministic ID.
        content:    Full file text.
        commit_sha: Commit this file's content was read at — stamped onto
                    every chunk for staleness detection later.

    Returns:
        One or more Chunks. Never raises — a file this function can't
        parse falls through to fixed-size chunking rather than blocking
        a full index run.
    """
    ext = Path(file_path).suffix.lower()
    language = _LANGUAGE_BY_EXTENSION.get(ext, "other")

    if ext == ".py":
        pieces = _chunk_python_with_fallback(content)
    elif ext in _MARKDOWN_EXTENSIONS:
        pieces = _chunk_markdown(content)
    elif ext in _CONFIG_EXTENSIONS:
        pieces = [(content, 1, len(content.splitlines()) or 1, ChunkType.CONFIG)]
    else:
        pieces = _chunk_fallback_regex(content) or _chunk_fixed_size(content)

    pieces = _apply_size_guards(pieces)
    now = datetime.now(timezone.utc).isoformat()

    return [
        Chunk(
            chunk_id=make_chunk_id(file_path, i),
            file_path=file_path,
            content=text,
            start_line=start,
            end_line=end,
            chunk_type=chunk_type,
            language=language,
            indexed_commit_sha=commit_sha,
            last_indexed=now,
        )
        for i, (text, start, end, chunk_type) in enumerate(pieces)
    ]


# --- Python: AST-based chunking -------------------------------------------

def _chunk_python_with_fallback(content: str) -> list[tuple[str, int, int, ChunkType]]:
    """Try AST chunking; fall back to regex, then fixed-size, on failure."""
    try:
        return _chunk_python(content)
    except SyntaxError:
        return _chunk_fallback_regex(content) or _chunk_fixed_size(content)


def _chunk_python(content: str) -> list[tuple[str, int, int, ChunkType]]:
    """One chunk per top-level function or class; class docstrings get an
    extra chunk of their own. Raises SyntaxError on invalid Python 3 —
    the caller catches this and falls back."""
    tree = ast.parse(content)
    lines = content.splitlines()
    pieces: list[tuple[str, int, int, ChunkType]] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunk_type = ChunkType.TEST if node.name.startswith("test_") else ChunkType.FUNCTION
            pieces.append(_extract(lines, node.lineno, node.end_lineno, chunk_type))

        elif isinstance(node, ast.ClassDef):
            pieces.append(_extract(lines, node.lineno, node.end_lineno, ChunkType.CLASS))
            docstring_span = _class_docstring_span(node)
            if docstring_span:
                doc_start, doc_end = docstring_span
                pieces.append(_extract(lines, doc_start, doc_end, ChunkType.CLASS))

    if not pieces:
        # No top-level functions/classes — treat the whole file as one
        # module-level chunk rather than returning nothing.
        pieces.append((content, 1, len(lines) or 1, ChunkType.MODULE))

    return pieces


def _extract(lines: list[str], start: int, end: int, chunk_type: ChunkType) -> tuple[str, int, int, ChunkType]:
    """Pull a 1-indexed, inclusive line range out of a file's lines."""
    return ("\n".join(lines[start - 1:end]), start, end, chunk_type)


def _class_docstring_span(node: ast.ClassDef) -> Optional[Tuple[int, int]]:
    """Return (start_line, end_line) of a class's docstring, if it has one."""
    if not node.body:
        return None
    first = node.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first.lineno, first.end_lineno
    return None


# --- Markdown: heading-based chunking --------------------------------------

_HEADING_PATTERN = re.compile(r"^#{1,6}\s")


def _chunk_markdown(content: str) -> list[tuple[str, int, int, ChunkType]]:
    """Split on markdown heading lines (# through ######). Content before
    the first heading becomes its own leading chunk."""
    lines = content.splitlines()
    boundaries = [i for i, line in enumerate(lines) if _HEADING_PATTERN.match(line)]
    if not boundaries or boundaries[0] != 0:
        boundaries = [0] + boundaries
    boundaries.append(len(lines))

    pieces = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1] - 1
        if end < start:
            continue
        pieces.append(("\n".join(lines[start:end + 1]), start + 1, end + 1, ChunkType.HEADING))
    return pieces or [(content, 1, len(lines) or 1, ChunkType.HEADING)]


# --- Fallback chunking (used for broken Python and unparsed languages) ----

def _chunk_fallback_regex(
    content: str,
) -> Optional[List[Tuple[str, int, int, ChunkType]]]:    
    """Best-effort split on lines that look like a function/class definition.

    Loose on purpose — there's no real parser here for every language.
    Returns None if nothing matches, so the caller falls through to
    fixed-size chunking instead.
    """
    lines = content.splitlines()
    boundaries = [i for i, line in enumerate(lines) if _GENERIC_DEF_PATTERN.match(line)]
    if not boundaries:
        return None

    boundaries.append(len(lines))
    pieces = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1] - 1
        if end < start:
            continue
        pieces.append(("\n".join(lines[start:end + 1]), start + 1, end + 1, ChunkType.FALLBACK_REGEX))
    return pieces


def _chunk_fixed_size(
    content: str, lines_per_chunk: int = 60, overlap_lines: int = 5
) -> list[tuple[str, int, int, ChunkType]]:
    """Last resort: fixed-size windows with overlap, no semantic awareness."""
    lines = content.splitlines()
    if not lines:
        return [(content, 1, 1, ChunkType.FALLBACK_FIXED_SIZE)]

    pieces = []
    i = 0
    while i < len(lines):
        end = min(i + lines_per_chunk, len(lines))
        pieces.append(("\n".join(lines[i:end]), i + 1, end, ChunkType.FALLBACK_FIXED_SIZE))
        if end == len(lines):
            break
        i = end - overlap_lines
    return pieces


# --- Size guards: applied after language-specific chunking -----------------

def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) — good enough for a size
    guard, not a substitute for a real tokenizer."""
    return max(1, len(text) // 4)


# Only fallback chunking (regex/fixed-size, used when there's no real
# parser) produces meaningless tiny fragments worth merging away. A short
# AST function, a class docstring, or a brief markdown section are still
# complete, meaningful units — merging them into an unrelated neighbor
# would mix semantics for no benefit.
_MERGEABLE_TYPES = {ChunkType.FALLBACK_REGEX, ChunkType.FALLBACK_FIXED_SIZE}


def _apply_size_guards(
    pieces: list[tuple[str, int, int, ChunkType]],
) -> list[tuple[str, int, int, ChunkType]]:
    """Sub-split anything over ~1500 tokens; merge undersized FALLBACK
    chunks forward into the next one. Required by locked design, scoped
    to where it actually helps — see _MERGEABLE_TYPES above.
    """
    split: list[tuple[str, int, int, ChunkType]] = []
    for text, start, end, chunk_type in pieces:
        if _estimate_tokens(text) > _MAX_CHUNK_TOKENS:
            split.extend(_split_oversized(text, start, end, chunk_type))
        else:
            split.append((text, start, end, chunk_type))

    if not split:
        return split

    merged = [split[0]]
    for text, start, end, chunk_type in split[1:]:
        prev_text, prev_start, _prev_end, prev_type = merged[-1]
        if prev_type in _MERGEABLE_TYPES and _estimate_tokens(prev_text) < _MIN_CHUNK_TOKENS:
            merged[-1] = (prev_text + "\n" + text, prev_start, end, prev_type)
        else:
            merged.append((text, start, end, chunk_type))
    return merged


def _split_oversized(
    text: str, start: int, end: int, chunk_type: ChunkType
) -> list[tuple[str, int, int, ChunkType]]:
    """Break one too-large chunk into evenly-sized pieces by line count."""
    lines = text.splitlines()
    num_pieces = (_estimate_tokens(text) // _MAX_CHUNK_TOKENS) + 1
    lines_per_piece = max(1, len(lines) // num_pieces)

    pieces = []
    for i in range(0, len(lines), lines_per_piece):
        piece_lines = lines[i:i + lines_per_piece]
        piece_start = start + i
        piece_end = min(end, piece_start + len(piece_lines) - 1)
        pieces.append(("\n".join(piece_lines), piece_start, piece_end, chunk_type))
    return pieces