import numpy as np


def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    return dot_product / (norm_a * norm_b)

def section_similarity_matrix(resume_sections, jd_sections):
    matrix = {}

    for jd_section, jd_vector in jd_sections.items():
        matrix[jd_section] = {}

        for resume_section, resume_vector in resume_sections.items():
            score = cosine_similarity(
                resume_vector,
                jd_vector
            )

            matrix[jd_section][resume_section] = float(score)

    return matrix