import chromadb

from .embeddings import get_embedding_function
from .models import Chunk, QueryMatch
from typing import Optional


class VectorStore:
    """Thin wrapper around one repo's ChromaDB collection.

    Locked design: persistent (on-disk, survives restarts), one collection
    per repo, collection name = repo full name (sanitized — Chroma names
    can't contain "/").

    Usage:
        store = VectorStore(persist_path="./chroma_db", repo_full_name="owner/repo")
        store.upsert_file(chunks)             # delete-and-re-insert one file
        store.delete_file("app/old_module.py")  # file removed/renamed away
        matches = store.query("some diff text", n_results=5, exclude_file="app/foo.py")
    """

    def __init__(self, persist_path: str, repo_full_name: str) -> None:
        """Open (or create) the persistent collection for one repo.

        Args:
            persist_path:   On-disk directory for ChromaDB's data.
            repo_full_name: e.g. "owner/repo" — used as the collection name.
        """
        client = chromadb.PersistentClient(path=persist_path)
        collection_name = repo_full_name.replace("/", "__")
        self._collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=get_embedding_function(),
        )

    def upsert_file(self, chunks: list[Chunk]) -> None:
        """Replace all of one file's chunks: delete the old ones, insert the new.

        This is the delete-and-re-insert pattern locked in planning chat —
        simpler and more reliable than updating individual chunks in place.
        Safe to call on a brand-new file too (the delete is just a no-op).

        Args:
            chunks: All chunks for ONE file, freshly produced by chunker.py.
                    Must be non-empty — for a file with no chunks (deleted
                    or filtered out), call delete_file() instead.
        """
        if not chunks:
            return

        file_path = chunks[0].file_path
        self.delete_file(file_path)

        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.content for c in chunks],
            metadatas=[_to_metadata(c) for c in chunks],
        )

    def delete_file(self, file_path: str) -> None:
        """Delete every chunk belonging to one file.

        Used when re-indexing a changed file (paired with upsert_file), a
        deleted file, or the old half of a rename (delete + add new).
        """
        self._collection.delete(where={"file_path": file_path})

    def query(
        self,
        query_text: str,
        n_results: int = 20,
        exclude_file: Optional[str]= None,
    ) -> list[QueryMatch]:
        """Find the most similar chunks to a piece of text.

        Returns a generous candidate pool (default 20) — the retriever's
        overlap-dedup + relevance-threshold + token-budget pipeline filters
        this down to only what's actually useful. Fetching more candidates
        here is cheap; the filtering in retriever.py decides what survives.

        Args:
            query_text:   Text to embed and search against.
            n_results:    Candidate pool size. Default 20 — favour recall
                          here, precision comes from the retriever filter.
            exclude_file: If given, chunks from this file are excluded —
                          used so an agent doesn't get the file's own content
                          back as "related context."

        Returns:
            Up to n_results QueryMatch objects, nearest first. Empty list
            if the collection has no chunks yet, nothing matches, or the
            query itself fails for any reason — M3 is purely additive, so
            a broken retrieval must never break a review.
        """
        kwargs = {"query_texts": [query_text], "n_results": n_results}
        if exclude_file:
            kwargs["where"] = {"file_path": {"$ne": exclude_file}}

        try:
            result = self._collection.query(**kwargs)
        except Exception:
            return []

        ids = result.get("ids") or [[]]
        if not ids or not ids[0]:
            return []

        metadatas = result["metadatas"][0]
        documents = result["documents"][0]
        distances = result["distances"][0]

        return [
            QueryMatch(
                chunk_id=ids[0][i],
                file_path=metadatas[i]["file_path"],
                content=documents[i],
                start_line=metadatas[i]["start_line"],
                end_line=metadatas[i]["end_line"],
                chunk_type=metadatas[i]["chunk_type"],
                distance=distances[i],
            )
            for i in range(len(ids[0]))
        ]


def _to_metadata(chunk: Chunk) -> dict:
    """Flatten a Chunk into the dict Chroma's `metadatas` parameter expects.

    Excludes chunk_id (already the Chroma id) and content (already the
    Chroma document) — metadata is everything else.
    """
    return {
        "file_path": chunk.file_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "chunk_type": chunk.chunk_type.value,
        "language": chunk.language,
        "indexed_commit_sha": chunk.indexed_commit_sha,
        "last_indexed": chunk.last_indexed,
    }