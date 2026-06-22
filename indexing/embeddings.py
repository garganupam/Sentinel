from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Decided: local sentence-transformers model, not Gemini — offline, no API
# cost or network dependency at index/query time. Also Chroma's own
# internal default embedding model.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    """Build the embedding function used for every chunk and every query.

    Single source of truth: store.py imports this rather than constructing
    its own — swapping embedding models later means editing this one
    function, not every caller.

    Important: once a Chroma collection's first embeddings are added, the
    collection is locked to this model's output dimensionality. Every
    future add/query against that collection must use this exact same
    function — swapping models means creating a new collection, not an
    in-place change.

    Returns:
        A configured SentenceTransformerEmbeddingFunction, ready to pass
        into chromadb's create_collection() / get_or_create_collection().
    """
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)