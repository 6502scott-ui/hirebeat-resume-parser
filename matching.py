import json
import re
from pathlib import Path

from chunking import section_aware_chunk_text
from embedding import embed_text
from similarity import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"


def load_role_configs():
    """
    Load all job matching configs from config/*.json.

    Each config must contain a unique role_id.
    """

    configs = {}

    for config_path in CONFIG_DIR.glob("*.json"):
        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as file:
            config = json.load(file)

        role_id = config.get("role_id")

        if role_id is None:
            continue

        role_id = int(role_id)

        if role_id in configs:
            raise ValueError(
                f"Duplicate matching config for role_id={role_id}"
            )

        configs[role_id] = config

    return configs


ROLE_CONFIGS = load_role_configs()


def get_role_config(role_id: int):
    return ROLE_CONFIGS.get(int(role_id))


def get_supported_role_ids():
    return sorted(ROLE_CONFIGS.keys())


def build_evidence_chunks(resume_chunks):
    """
    Split Experience / Projects into smaller evidence units.

    Education and Skills remain unchanged.
    """

    evidence_chunks = []

    for chunk in resume_chunks:
        section = chunk["section"]
        text = chunk["text"]

        if section in {"experience", "projects"}:
            parts = re.split(
                r"\s*[•▪]\s*|\n+|(?<=[.!?])\s+",
                text,
            )

            for part in parts:
                part = part.strip()

                # Ignore very small fragments.
                if len(part.split()) < 5:
                    continue

                evidence_chunks.append(
                    {
                        "section": section,
                        "text": part,
                    }
                )

        else:
            evidence_chunks.append(chunk)

    return evidence_chunks


def score_criterion(
    criterion,
    resume_chunks,
):
    """
    Find the best matching resume evidence for one
    human-defined criterion.
    """

    criterion_text = criterion["text"]

    allowed_sections = set(
        criterion["allowed_sections"]
    )

    candidate_chunks = [
        chunk
        for chunk in resume_chunks
        if chunk.get("section") in allowed_sections
        and str(chunk.get("text", "")).strip()
    ]

    if not candidate_chunks:
        return 0.0, None

    criterion_vector = embed_text(
        criterion_text
    )

    best_score = -1.0
    best_chunk = None

    for chunk in candidate_chunks:
        chunk_text = str(chunk.get("text", "")).strip()

        if not chunk_text:
            continue

        chunk_vector = embed_text(chunk_text)

        score = float(
            cosine_similarity(
                criterion_vector,
                chunk_vector,
            )
        )

        if score > best_score:
            best_score = score
            best_chunk = chunk

    return float(best_score), best_chunk


def score_resume(
    criteria,
    resume_text: str,
):
    """
    Score one resume using Human-in-the-loop criteria.

    Returns:
        overall_score,
        evidence_rows
    """

    # --------------------------------------------------------
    # 1. Resume section detection
    # --------------------------------------------------------

    resume_chunks = section_aware_chunk_text(
        resume_text
    )

    # Ignore header/contact content that could not be assigned
    # to a meaningful resume section.
    resume_chunks = [
        chunk
        for chunk in resume_chunks
        if chunk["section"] != "other"
    ]

    if not resume_chunks:
        return None, []

    # --------------------------------------------------------
    # 2. Fine-grained evidence chunking
    # --------------------------------------------------------

    resume_chunks = build_evidence_chunks(
        resume_chunks
    )

    if not resume_chunks:
        return None, []

    # --------------------------------------------------------
    # 3. Criterion-level matching
    # --------------------------------------------------------

    weighted_sum = 0.0
    total_weight = 0.0

    evidence_rows = []

    for criterion in criteria:
        score, best_chunk = score_criterion(
            criterion,
            resume_chunks,
        )

        weight = float(
            criterion["weight"]
        )

        weighted_score = (
            float(score) * weight
        )

        weighted_sum += weighted_score
        total_weight += weight

        evidence_rows.append(
            {
                "criterion": criterion["name"],
                "score": float(score),
                "weight": weight,
                "weighted_score": float(
                    weighted_score
                ),
                "matched_section": (
                    best_chunk["section"]
                    if best_chunk is not None
                    else None
                ),
                "matched_evidence": (
                    best_chunk["text"]
                    if best_chunk is not None
                    else None
                ),
            }
        )

    if total_weight == 0:
        return None, evidence_rows

    overall_score = (
        weighted_sum / total_weight
    )

    return float(overall_score), evidence_rows


def score_resume_for_role(
    role_id: int,
    resume_text: str,
):
    """
    Production entry point.

    role_id determines which human-defined job configuration
    should be used.
    """

    config = get_role_config(role_id)

    if config is None:
        raise ValueError(
            f"No matching configuration found for role_id={role_id}"
        )

    criteria = config.get("criteria", [])

    if not criteria:
        raise ValueError(
            f"No matching criteria configured for role_id={role_id}"
        )

    overall_score, evidence = score_resume(
        criteria,
        resume_text,
    )

    return {
        "role_id": role_id,
        "role": config.get("role"),
        "matching_version": config.get(
            "matching_version",
            "unknown",
        ),
        "score": overall_score,
        "criteria_count": len(criteria),
        "evidence": evidence,
    }