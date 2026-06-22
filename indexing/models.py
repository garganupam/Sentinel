from dataclasses import dataclass
from enum import Enum


class ChunkType(str, Enum):
    """How a chunk was produced / what kind of semantic unit it represents.

    AST-derived chunks (FUNCTION/CLASS/MODULE/TEST) have clean, meaningful
    boundaries. HEADING is markdown-specific. CONFIG is a whole small file
    kept intact. The two FALLBACK_* values exist because ast.parse failed
    (or the language has no parser here) — boundaries are best-effort.
    """
    FUNCTION = "function"
    CLASS = "class"
    HEADING = "heading"
    MODULE = "module"
    CONFIG = "config"
    TEST = "test"
    FALLBACK_REGEX = "fallback_regex"
    FALLBACK_FIXED_SIZE = "fallback_fixed_size"


@dataclass(frozen=True)
class Chunk:
    """One semantically-meaningful piece of source code, ready to embed and store.

    Locked design (planning chat): chunk_id is deterministic — built from
    file_path + an index — so a file's old chunks can always be found and
    deleted before its new ones are inserted (delete-and-re-insert).

    Attributes:
        chunk_id:           Deterministic ID, e.g. "app/models.py::0".
        file_path:          Repo-relative path the chunk came from.
        content:            The actual source text of this chunk.
        start_line:         First line of this chunk in the original file.
        end_line:           Last line of this chunk in the original file.
        chunk_type:         How it was produced — see ChunkType.
        language:           Coarse language label derived from extension,
                            e.g. "python", "markdown" — enables optional
                            language-filtered retrieval later.
        indexed_commit_sha: Commit SHA this chunk's content reflects.
                            Used for staleness detection.
        last_indexed:       ISO 8601 timestamp of when this chunk was
                            written to the store (distinct from the SHA
                            above — this is wall-clock time).
    """
    chunk_id: str
    file_path: str
    content: str
    start_line: int
    end_line: int
    chunk_type: ChunkType
    language: str
    indexed_commit_sha: str
    last_indexed: str


def make_chunk_id(file_path: str, index: int) -> str:
    """Build a deterministic chunk ID from a file path and chunk index.

    Single source of truth for the ID format — chunker.py creates these;
    store.py and indexer.py rely on the file_path prefix to find and
    delete a file's old chunks before inserting new ones.
    """
    return f"{file_path}::{index}"


@dataclass(frozen=True)
class QueryMatch:
    """One result from a vector store similarity search.

    Attributes:
        chunk_id:    The matched chunk's deterministic ID.
        file_path:   Which file this chunk came from.
        content:     The chunk's source text.
        start_line:  First line of the chunk in its original file.
        end_line:    Last line of the chunk in its original file.
        chunk_type:  How the chunk was produced (string value of ChunkType).
        distance:    Raw distance from ChromaDB — smaller is more similar.
                    NOT a similarity score; retriever.py converts this for
                    logging/display, store.py stays a thin pass-through.
    """
    chunk_id: str
    file_path: str
    content: str
    start_line: int
    end_line: int
    chunk_type: str
    distance: float