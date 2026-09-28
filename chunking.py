from embedding import _model
import re

CHUNK_SIZE = 220


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE):
    tokenizer = _model.tokenizer

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    )

    chunks = []

    for start in range(0, len(token_ids), chunk_size):
        chunk_token_ids = token_ids[start:start + chunk_size]

        chunk_text = tokenizer.decode(
            chunk_token_ids,
            skip_special_tokens=True,
        )

        chunks.append(chunk_text)

    return chunks
SECTION_ALIASES = {
    "education": [
        "education",
        "academic background",
        "education background",
        "educational background",
        "academic experience",
        "academic experiences",
    ],
    "experience": [
        "experience",
        "experiences",
        "work experience",
        "work experiences",
        "professional experience",
        "professional experiences",
        "internship experience",
        "internship experiences",
        "relevant experience",
        "relevant experiences",
        "employment history",
        "employment experience",
        "research experience",
        "research experiences",
        "leadership experience",
        "leadership experiences",
        "leadership experience and activities",
        "professional experiences",
        "research",
        "leadership",
        "leadership & activities",
        "leadership and activities",
        "internship",
        "internships",
    ],
    
    "skills": [
        "skills",
        "technical skills",
        "skills & certificates",
        "skills and certificates",
        "technologies",
        "skills & interests",
        "skills and interests",
        "skills, activities & interests",
        'tools',
        'tool',
        "skill",
        "technical skill",
        "technical skills and interests",
        "technical skills & interests",
        "skills & certifications",
        "skills and certifications",
        "professional skills",
        "relevant skills",
    ],
    "projects": [
        "project",
        "projects",
        "academic project",
        "academic projects",
        "project experience",
        "project experiences",
        "selected project",
        "selected projects",
        "personal project",
        "personal projects",
        "technical project",
        "technical projects",
        "course project",
        "course projects",
        "research project",
        "research projects",
    ],
}
JD_SECTION_ALIASES = {
    "description": [
        "job description",
        "about the role",
        "about this role",
        "position description",
    ],

    "responsibilities": [
        "primary responsibilities",
        "key responsibilities",
        "responsibilities",
        "what you'll do",
        "what you will do",
        "duties",
    ],

    "requirements": [
        "basic qualifications",
        "basic requirements",
        "qualifications",
        "requirements",
        "minimum qualifications",
        "required qualifications",
        "required skills and experience",
    ],

    "preferred": [
        "preferred qualifications",
        "preferred requirements",
        "preferred skills",
        "preferred skills and experience",
        "nice to have",
        "nice-to-have",
    ],

    # These sections mainly act as boundaries.
    "benefits": [
        "benefits",
        "what we offer",
        "perks",
        "perks and benefits",
        "compensation and benefits",
    ],

    "company": [
        "about us",
        "about the company",
        "who we are",
    ],

    "other": [
        "location",
        "position location",
        "compensation",
        "salary",
        "pay range",
        "equal opportunity",
        "equal employment opportunity",
        "eeo",
    ],
}
def normalize_section_heading(line: str):
    normalized = line.strip().lower()

    for section, aliases in SECTION_ALIASES.items():
        if normalized in aliases:
            return section

    return None
def restore_inline_resume_headings(text: str):
    """
    Restore section boundaries that were lost during PDF text extraction.

    Example:
    "... technical skills : python / sql ... internships company A ..."

    becomes:

    "...
    technical skills
    python / sql ...
    internships
    company A ..."
    """

    strong_headings = [
        "technical skills",
        "professional experience",
        "work experience",
        "internship experience",
        "research experience",
        "leadership experience",
        "selected projects",
        "academic projects",
        "personal projects",
        "technical projects",
        "research projects",
    ]

    # Multi-word headings are relatively safe to detect inline.
    for heading in sorted(
        strong_headings,
        key=len,
        reverse=True,
    ):
        pattern = rf"(?i)(?<!\w){re.escape(heading)}\s*:?\s*"

        text = re.sub(
            pattern,
            f"\n{heading}\n",
            text,
        )

    # Some resumes use simply "INTERNSHIPS" as a heading.
    text = re.sub(
        r"(?i)(?<!\w)internships\s+",
        "\ninternships\n",
        text,
    )

    return text
def split_sections(text: str):
    text = restore_inline_resume_headings(text)
    sections = []
    current_section = "other"
    current_lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        section = normalize_section_heading(line)

        if section is not None:
            if current_lines:
                sections.append({
                    "section": current_section,
                    "text": "\n".join(current_lines)
                })

            current_section = section
            current_lines = []

        else:
            current_lines.append(line)

    if current_lines:
        sections.append({
            "section": current_section,
            "text": "\n".join(current_lines)
        })

    return sections
def section_aware_chunk_text(text: str, chunk_size: int = CHUNK_SIZE):
    """
    Split resume into semantic sections first, then chunk each section
    independently by token length.

    Returns:
        [
            {
                "section": "education",
                "chunk_index": 0,
                "text": "..."
            },
            ...
        ]
    """

    sections = split_sections(text)

    output_chunks = []

    for section_data in sections:
        section_name = section_data["section"]
        section_text = section_data["text"].strip()

        if not section_text:
            continue

        # Reuse our existing token-based chunking function
        text_chunks = chunk_text(
            section_text,
            chunk_size=chunk_size
        )

        for chunk_index, chunk in enumerate(text_chunks):
            token_count = len(
                _model.tokenizer.encode(
                    chunk,
                    add_special_tokens=False,
                    truncation=False,
                )
            )
            max_model_length = _model.max_seq_length

            special_token_count = _model.tokenizer.num_special_tokens_to_add(
                pair=False
            )

            max_content_length = (
                max_model_length - special_token_count
            )

            assert token_count <= max_content_length, (
                f"Chunk too long for model: "
                f"{token_count} content tokens + "
                f"{special_token_count} special tokens "
                f"> {max_model_length} "
                f"in section {section_name}"
            )
            output_chunks.append({
                "section": section_name,
                "chunk_index": chunk_index,
                "token_count": token_count,
                "text": chunk,
            })

    return output_chunks

def normalize_jd_section_heading(line: str):
    normalized = line.strip().lower().rstrip(":").strip()

    for section, aliases in JD_SECTION_ALIASES.items():
        if normalized in aliases:
            return section

    return None

def split_jd_sections(text: str):
    sections = []
    current_section = "other"
    current_lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        section = normalize_jd_section_heading(line)
        # Detect metadata-style section boundaries such as:
        # "Position Location: New York City / Remote"
        if section is None and ":" in line:
            possible_heading = line.split(":", 1)[0].strip()
            section = normalize_jd_section_heading(possible_heading)

        if section is not None:
            if current_lines:
                sections.append({
                    "section": current_section,
                    "text": "\n".join(current_lines),
                })

            current_section = section
            current_lines = []

        else:
            current_lines.append(line)

    if current_lines:
        sections.append({
            "section": current_section,
            "text": "\n".join(current_lines),
        })

    return sections

def jd_section_aware_chunk_text(text: str, chunk_size: int = CHUNK_SIZE):
    sections = split_jd_sections(text)

    output_chunks = []

    for section in sections:
        section_name = section["section"]
        section_text = section["text"]

        text_chunks = chunk_text(
            section_text,
            chunk_size=chunk_size
        )

        for chunk_index, chunk in enumerate(text_chunks):
            token_count = len(
                _model.tokenizer.encode(
                    chunk,
                    add_special_tokens=False,
                    truncation=False,
                )
            )

            output_chunks.append({
                "section": section_name,
                "chunk_index": chunk_index,
                "token_count": token_count,
                "text": chunk,
            })

    return output_chunks
JD_FOOTER_PREFIXES = [
    "you'll be rewarded",
    "you will be rewarded",
    "equal opportunity employer",
    "we are an equal opportunity",
]
def split_requirements(requirements_text: str):
    requirements = []
    current_requirement = []

    for line in requirements_text.splitlines():
        line = line.strip()

        if not line:
            continue

        normalized = line.lower().replace("’", "'")
        footer_position = None

        for prefix in JD_FOOTER_PREFIXES:
            position = normalized.find(prefix)

            if position != -1:
                footer_position = position
                break

        if footer_position is not None:
            valid_text = line[:footer_position].strip()

            if valid_text and current_requirement:
                current_requirement.append(valid_text)

            break

        # A new bullet starts a new requirement
        if line.startswith("•"):
            if current_requirement:
                requirements.append(" ".join(current_requirement))

            current_requirement = [
                line.lstrip("•").strip()
            ]

        elif current_requirement:
            current_requirement.append(line)
    # Save the final requirement
    if current_requirement:
        requirements.append(" ".join(current_requirement))

    return requirements