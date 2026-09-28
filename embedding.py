import os
from collections import defaultdict

import numpy as np
from huggingface_hub import InferenceClient

from chunking import TOKENIZER


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_client = None


def get_client():
    """
    Create the Hugging Face inference client lazily.

    The MiniLM model is no longer loaded into Render memory.
    Instead, text is sent to Hugging Face for embedding.
    """
    global _client

    if _client is None:
        token = os.getenv("HF_TOKEN")

        if not token:
            raise RuntimeError(
                "HF_TOKEN environment variable is missing."
            )

        _client = InferenceClient(
            provider="hf-inference",
            api_key=token,
        )

    return _client


def embed_text(text: str):
    """
    Embed one text using remote all-MiniLM-L6-v2.
    Returns a 384-dimensional numpy vector.
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text.")

    vector = get_client().feature_extraction(
        text,
        model=MODEL_NAME,
    )

    vector = np.asarray(
        vector,
        dtype=np.float32,
    )

    if vector.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"Unexpected embedding shape: {vector.shape}"
        )

    return vector


def embed_texts(texts: list[str]):
    """
    Embed multiple texts in one remote API request.

    This is much more efficient than sending one HTTP
    request for every individual chunk.
    """
    if not texts:
        return np.empty(
            (0, EMBEDDING_DIM),
            dtype=np.float32,
        )

    vectors = get_client().feature_extraction(
        texts,
        model=MODEL_NAME,
    )

    vectors = np.asarray(
        vectors,
        dtype=np.float32,
    )

    # Hugging Face may return a 1-D vector
    # when the batch contains only one text.
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    if (
        vectors.ndim != 2
        or vectors.shape[1] != EMBEDDING_DIM
    ):
        raise ValueError(
            f"Unexpected batch embedding shape: "
            f"{vectors.shape}"
        )

    return vectors


def get_token_count(text: str):
    """
    Count tokens using the same MiniLM tokenizer
    without loading the full transformer model.
    """
    encoding = TOKENIZER.encode(
        text,
        add_special_tokens=True,
    )

    return len(encoding.ids)


def aggregate_embeddings(embeddings):
    return np.mean(
        embeddings,
        axis=0,
    )


def aggregate_embeddings_weighted(
    embeddings,
    weights,
):
    embeddings = np.asarray(embeddings)

    weights = np.asarray(
        weights,
        dtype=float,
    )

    if len(embeddings) != len(weights):
        raise ValueError(
            "embeddings and weights must have the same length"
        )

    if len(embeddings) == 0:
        raise ValueError(
            "cannot aggregate empty embeddings"
        )

    return np.average(
        embeddings,
        axis=0,
        weights=weights,
    )


def embed_sections(chunks):
    grouped_chunks = defaultdict(list)

    for chunk in chunks:
        grouped_chunks[
            chunk["section"]
        ].append(chunk)

    section_embeddings = {}

    for (
        section_name,
        section_chunks,
    ) in grouped_chunks.items():

        texts = [
            chunk["text"]
            for chunk in section_chunks
        ]

        weights = [
            chunk["token_count"]
            for chunk in section_chunks
        ]

        chunk_embeddings = embed_texts(
            texts
        )

        section_embedding = (
            aggregate_embeddings_weighted(
                chunk_embeddings,
                weights,
            )
        )

        section_embeddings[
            section_name
        ] = section_embedding

    return section_embeddings