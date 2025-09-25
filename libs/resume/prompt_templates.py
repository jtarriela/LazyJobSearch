"""
Prompt templates for resume parsing + retries + PDF-requirements inference.

This module centralizes compact, token-efficient system/user prompts used when
calling an LLM to extract structured resume data or to infer fields a PDF is
requesting. It aligns with the mock/real providers that look for the sentinel
'RESUME TEXT:' (or 'PDF TEXT:') in the user prompt.

Exports:
- build_parse_prompt(text, hints=None, schema_hint=None) -> (system, user)
- build_retry_prompt(text, missing_fields, current_json=None) -> (system, user)
- build_requirements_prompt(pdf_text) -> (system, user)
"""

from __future__ import annotations
from typing import Dict, Any, Iterable, Tuple, Optional
import json


# -------------------------
# Internal helpers
# -------------------------

_DEFAULT_SCHEMA = """{
  "full_name": "<string|null>",
  "email": "<string|null>",
  "phone": "<string|null>",
  "skills": ["<string>", "..."],
  "experience": [
    {
      "title": "<string|null>",
      "company": "<string|null>",
      "duration": "<string|null>",
      "description": "<string|null>"
    }
  ],
  "education": [
    {
      "degree": "<string|null>",
      "field": "<string|null>",
      "institution": "<string|null>",
      "year": "<string|null>"
    }
  ],
  "full_text": "<the ENTIRE resume text you received>",
  "summary": "<string|null>",
  "certifications": ["<string>", "..."],
  "years_of_experience": <number|null>,
  "education_level": "<high_school|associates|bachelors|masters|phd|null>",
  "links": {"linkedin":"<string|null>", "github":"<string|null>", "portfolio":"<string|null>"}
}"""


def _compact_hints(hints: Optional[Dict[str, Any]]) -> str:
    if not hints:
        return ""
    parts = []
    if hints.get("email"):
        parts.append(f"Email={hints['email']}")
    if hints.get("phone"):
        parts.append(f"Phone={hints['phone']}")
    if hints.get("name"):
        parts.append(f"Name={hints['name']}")
    return "; ".join(parts)


# -------------------------
# Public builders
# -------------------------

def build_parse_prompt(
    text: str,
    hints: Optional[Dict[str, Any]] = None,
    schema_hint: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Return (system_prompt, user_prompt) for FIRST-PASS parsing.

    - Uses 'RESUME TEXT:' sentinel so providers can reliably slice content.
    - Demands ONLY raw JSON, no markdown fences or prose.
    - If schema_hint is not provided, uses an informal minimal schema.
    - Keeps tokens lean; avoids over-instruction.
    """
    system = (
        "You are a precise resume extraction engine. "
        "Return ONLY one valid JSON object matching the schema. "
        "Do NOT invent information that is not present. "
        "If a field is missing, use null for scalars or [] for lists."
    )

    hints_line = _compact_hints(hints)
    schema = schema_hint.strip() if schema_hint else _DEFAULT_SCHEMA

    user_sections = []
    user_sections.append("# TASK\nExtract structured resume data as JSON ONLY.\n")
    if hints_line:
        user_sections.append(f"# HINTS\n{hints_line}\n")
    user_sections.append("# SCHEMA (informal, types in angle brackets)\n")
    user_sections.append(schema + "\n")
    user_sections.append("RESUME TEXT:\n")
    user_sections.append(text.rstrip() + "\n")
    user_sections.append("IMPORTANT:\n- Output MUST be a single JSON object with the exact keys from the schema.\n"
                         "- No markdown, no commentary, no code fences.\n")

    user = "\n".join(user_sections)
    return system, user


def build_retry_prompt(
    text: str,
    missing_fields: Iterable[str],
    current_json: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """
    Return (system_prompt, user_prompt) for FOLLOW-UP passes when some fields
    were missing. The model must ONLY return a JSON object containing the
    missing fields filled (others omitted).
    """
    system = (
        "You are a precise resume extraction engine. "
        "Return ONLY one JSON object. "
        "Do NOT invent information. If still missing, set null/[]."
    )

    missing_str = ", ".join(str(f) for f in missing_fields) or "(none)"
    current_str = json.dumps(current_json or {}, ensure_ascii=False, separators=(",", ":"), indent=2)

    user_sections = []
    user_sections.append("# TASK\nSome fields were missing; return ONLY those fields in JSON.\n")
    user_sections.append(f"# MISSING FIELDS\n{missing_str}\n")
    user_sections.append("# CURRENT DATA (already extracted; do not repeat fields that are not missing)\n")
    user_sections.append(current_str + "\n")
    user_sections.append("RESUME TEXT:\n")
    user_sections.append(text.rstrip() + "\n")
    user_sections.append("IMPORTANT:\n- Output MUST be a single JSON object containing ONLY the missing fields.\n"
                         "- No markdown, no commentary, no code fences.\n")

    user = "\n".join(user_sections)
    return system, user


def build_requirements_prompt(pdf_text: str) -> Tuple[str, str]:
    """
    Return (system_prompt, user_prompt) for PDF requirements inference
    (extracting the fields the PDF is asking the user to provide).

    Output MUST be a JSON array of objects:
      [{"name": "...", "type": "string|number|date|email|phone|url|enum|address|textarea",
        "required": true/false, "description": "<string|null>",
        "options": ["opt1", "..."]|null, "regex_hint": "<string|null>", "example": "<string|null>"}]
    """
    system = (
        "You are a form field inference engine. "
        "Given PDF text (labels/instructions), return ONLY a JSON array describing input fields. "
        "Do NOT include any text outside JSON."
    )

    user_sections = []
    user_sections.append("# TASK\nInfer the input fields the PDF requests. Return ONLY a JSON array.\n")
    user_sections.append("# SCHEMA\n"
                         '[{"name":"<string>", "type":"string|number|date|email|phone|url|enum|address|textarea", '
                         '"required":<boolean>, "description":"<string|null>", "options":["<string>", "..."]|null, '
                         '"regex_hint":"<string|null>", "example":"<string|null>"}]\n')
    user_sections.append("PDF TEXT:\n")
    user_sections.append(pdf_text.rstrip() + "\n")
    user_sections.append("IMPORTANT:\n- Output MUST be a JSON array only. No markdown, no prose.\n")

    user = "\n".join(user_sections)
    return system, user
