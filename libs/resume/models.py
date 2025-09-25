from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class Experience(BaseModel):
    company: Optional[str] = None
    title: Optional[str] = None
    location: Optional[str] = None
    employment_type: Optional[str] = None
    start_date: Optional[str] = None  # normalized to 'YYYY-MM' or None
    end_date: Optional[str] = None    # normalized to 'YYYY-MM' or None
    current: bool = False
    bullets: List[str] = Field(default_factory=list)
    tech: List[str] = Field(default_factory=list)

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def _normalize_date(cls, v: Any) -> Optional[str]:
        """
        Accepts None, '', 'YYYY', 'YYYY-MM', or common 'present/current' tokens.
        - 'YYYY' -> 'YYYY-01'
        - 'YYYY-MM' passthrough
        - '', 'present', 'current' -> None (handled in model_validator for 'current' flag)
        """
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            low = s.lower()
            if low in {"present", "current", "now"}:
                return None
            # YYYY
            if len(s) == 4 and s.isdigit():
                return f"{s}-01"
            # YYYY-MM
            if (
                len(s) == 7
                and s[:4].isdigit()
                and s[4] == "-"
                and s[5:7].isdigit()
                and 1 <= int(s[5:7]) <= 12
            ):
                return s
        raise ValueError("Invalid date format (expected YYYY or YYYY-MM, or present/current)")

    @model_validator(mode="after")
    def _infer_current_from_end_date(self) -> "Experience":
        """
        If end_date text looked like 'present/current/now' (normalized to None above)
        and there's a start_date, mark current=True. If end_date is None but current
        already set, keep it as is.
        """
        if self.end_date is None and self.start_date and not self.current:
            # Heuristic: if there's a start_date and no end_date, assume current role
            self.current = True
        return self


class ParsedResume(BaseModel):
    full_name: Optional[str] = None
    contact: Dict[str, Any] = Field(default_factory=dict)          # {email, phone, linkedin, ...}
    summary: Optional[str] = None
    skills: Dict[str, Any] = Field(default_factory=dict)           # e.g. {"primary": [...], "secondary": [...]}
    experience: List[Experience] = Field(default_factory=list)
    education: List[Dict[str, Any]] = Field(default_factory=list)  # flexible until you formalize a model
    certifications: List[Dict[str, Any]] = Field(default_factory=list)
    projects: List[Dict[str, Any]] = Field(default_factory=list)
    awards: List[Dict[str, Any]] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    derived: Dict[str, Any] = Field(default_factory=dict)          # computed fields (yoe, seniority, etc.)
    parsing_method: str = "heuristic"
    version: int = 1

    @field_validator("full_name", mode="before")
    @classmethod
    def _normalize_full_name(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s or None
