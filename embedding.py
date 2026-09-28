from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict

MODEL_NAME = "all-MiniLM-L6-v2"


_model = SentenceTransformer(MODEL_NAME)


def embed_text(text: str):
    return _model.encode(text)

def embed_texts(texts: list[str]):
    return _model.encode(texts)

def get_token_count(text: str):
    tokenizer = _model.tokenizer

    tokens = tokenizer.encode(
        text,
        add_special_tokens=True,
        truncation=False
    )

    return len(tokens)
def aggregate_embeddings(embeddings):
    return np.mean(embeddings, axis=0)

def aggregate_embeddings_weighted(embeddings, weights):
    embeddings = np.asarray(embeddings)
    weights = np.asarray(weights, dtype=float)

    if len(embeddings) != len(weights):
        raise ValueError("embeddings and weights must have the same length")

    if len(embeddings) == 0:
        raise ValueError("cannot aggregate empty embeddings")

    return np.average(
        embeddings,
        axis=0,
        weights=weights
    )
def embed_sections(chunks):
    grouped_chunks = defaultdict(list)

    for chunk in chunks:
        grouped_chunks[chunk["section"]].append(chunk)

    section_embeddings = {}

    for section_name, section_chunks in grouped_chunks.items():
        texts = [chunk["text"] for chunk in section_chunks]
        weights = [chunk["token_count"] for chunk in section_chunks]

        chunk_embeddings = embed_texts(texts)

        section_embedding = aggregate_embeddings_weighted(
            chunk_embeddings,
            weights
        )

        section_embeddings[section_name] = section_embedding

    return section_embeddings