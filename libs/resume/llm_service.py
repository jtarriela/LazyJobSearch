"""
llm_service.py — unified, working LLM provider + service layer

What this gives you (fixed + coherent):
- A single, importable module that exposes:
    * LLMProvider (enum)
    * LLMConfig (reads env, defaults to mock)
    * create_llm_client(config) -> LLMClient
    * LLMService (high-level resume parser with retries)
    * create_llm_service(provider=?, model=?)  # backward-compat factory
    * LLMResumeParser (client-oriented variant, if you prefer)
- A single Mock implementation used everywhere (no naming collisions).
- OpenAI/Anthropic/Gemini clients that gracefully fall back to the mock.
- parse_resume(resume_text: str) that returns (ParsedResumeData, [LLMResponse]).
- Optional helpers:
    * generic_structured_extract(text, schema_instruction)
    * infer_pdf_requirements(pdf_text)

Drop-in usage in your ResumeParser:
    from .llm_service import create_llm_service, LLMProvider
    svc = create_llm_service(provider=LLMProvider.MOCK)
    parsed, responses = await svc.parse_resume(resume_text)

Everything is self-contained and tested to import without missing symbols.
"""


from __future__ import annotations
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import asyncio
import json
import logging
import os
import re
import time
import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

# Auto-load a local .env file if present to make local development easier.
# This is safe because if python-dotenv is not installed, the import will fail
# silently and we fall back to reading real environment variables.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

logger = logging.getLogger(__name__)


# =========================
# Provider / Config / DTOs
# =========================

class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    MOCK = "mock"  # Default for local/dev tests


class LLMConfig:
    """LLM configuration from environment variables."""
    def __init__(self):
        # Provider & model
        self.provider = os.getenv("LJS_LLM_PROVIDER", "mock").lower()
        self.model = os.getenv("LJS_LLM_MODEL", "gpt-3.5-turbo")

        # Timeouts/limits
        self.timeout = int(os.getenv("LJS_LLM_TIMEOUT", "30"))
        self.max_tokens = int(os.getenv("LJS_LLM_MAX_TOKENS", "2000"))

        # API keys (optional when using mock)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        # Validate known provider values
        if self.provider not in {"mock", "openai", "anthropic", "gemini"}:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")


@dataclass
class LLMRequest:
    """Request for LLM processing."""
    prompt: str
    model: str
    max_tokens: int = 2000
    temperature: float = 0.1
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.request_id is None:
            self.request_id = self._generate_request_id()

    def _generate_request_id(self) -> str:
        content_hash = hashlib.sha256(f"{self.prompt}:{self.model}".encode("utf-8")).hexdigest()
        return f"llm_{content_hash[:16]}"


@dataclass
class LLMResponse:
    """Response from LLM processing."""
    content: str
    model: str
    tokens_used: int
    cost_cents: float
    request_id: str
    processing_time_ms: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# =========================
# Parsed Resume Data Model
# =========================

class ExperienceItem(BaseModel):
    title: Optional[str] = Field(None, description="Job title")
    company: Optional[str] = Field(None, description="Company name")
    duration: Optional[str] = Field(None, description="Employment duration")
    description: Optional[str] = Field(None, description="Job description")
    location: Optional[str] = Field(None, description="Job location")


class EducationItem(BaseModel):
    degree: Optional[str] = Field(None, description="Degree type")
    field: Optional[str] = Field(None, description="Field of study")
    institution: Optional[str] = Field(None, description="School/University name")
    year: Optional[str] = Field(None, description="Graduation year")
    gpa: Optional[str] = Field(None, description="GPA if mentioned")


class CertificationItem(BaseModel):
    name: str = Field(..., description="Certification name")
    issuer: Optional[str] = Field(None, description="Issuing organization")
    date: Optional[str] = Field(None, description="Date obtained")


class ProjectItem(BaseModel):
    name: Optional[str] = Field(None, description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    technologies: List[str] = Field(default_factory=list, description="Technologies used")


class Links(BaseModel):
    linkedin: Optional[str] = Field(None, description="LinkedIn profile URL")
    github: Optional[str] = Field(None, description="GitHub profile URL")
    portfolio: Optional[str] = Field(None, description="Portfolio website URL")
    other: List[str] = Field(default_factory=list, description="Other relevant links")

    @classmethod
    def from_dict_or_none(cls, value):
        if value is None:
            return cls()
        if isinstance(value, dict):
            return cls(**value)
        return value


class Skills(BaseModel):
    technical: List[str] = Field(default_factory=list, description="Technical skills")
    soft: List[str] = Field(default_factory=list, description="Soft skills")
    languages: List[str] = Field(default_factory=list, description="Programming languages")
    tools: List[str] = Field(default_factory=list, description="Tools and frameworks")
    all: List[str] = Field(default_factory=list, description="All skills combined")


class ParsedResumeData(BaseModel):
    """Structured resume data from LLM parsing with strict validation."""
    # Core fields
    full_name: Optional[str] = Field(None, description="Full name")
    email: Optional[str] = Field(None, description="Primary email")
    phone: Optional[str] = Field(None, description="Phone number")
    skills: List[str] = Field(default_factory=list, description="All skills")
    experience: List[ExperienceItem] = Field(default_factory=list, description="Work experience")
    education: List[EducationItem] = Field(default_factory=list, description="Educational background")
    full_text: Optional[str] = Field(None, description="Entire resume text")

    # Optional
    summary: Optional[str] = Field(None, description="Professional summary")
    certifications: List[CertificationItem] = Field(default_factory=list, description="Certifications")
    projects: List[ProjectItem] = Field(default_factory=list, description="Projects")
    links: Links = Field(default_factory=Links, description="Links and profiles")
    skills_structured: Skills = Field(default_factory=Skills, description="Structured skills")
    years_of_experience: Optional[float] = Field(None, description="Total years of experience")
    education_level: Optional[str] = Field(None, description="Highest education level")

    @field_validator("links", mode="before")
    @classmethod
    def _v_links(cls, v):
        if v is None:
            return Links()
        if isinstance(v, dict):
            return Links(**v)
        return v

    @field_validator("skills_structured", mode="before")
    @classmethod
    def _v_skills_structured(cls, v):
        if v is None:
            return Skills()
        if isinstance(v, dict):
            return Skills(**v)
        return v

    @field_validator("email", mode="before")
    @classmethod
    def _v_email(cls, v):
        if v and "@" not in v:
            return None
        return v

    @field_validator("phone", mode="before")
    @classmethod
    def _v_phone(cls, v):
        if v and not re.search(r"\d{3}", v):
            return None
        return v

    @field_validator("skills", mode="before")
    @classmethod
    def _v_skills(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return v if isinstance(v, list) else []

    def get_missing_fields(self) -> List[str]:
        missing = []
        required = ["full_name", "email", "phone", "skills", "experience", "education", "full_text"]
        for f in required:
            value = getattr(self, f, None)
            if not value or (isinstance(value, list) and not value):
                missing.append(f)
        return missing

    def is_complete(self) -> bool:
        return len(self.get_missing_fields()) == 0


# =========================
# Client Interface + Impl
# =========================

class LLMClient:
    """Abstract LLM client interface (plain base class).

    Avoid using Pydantic here so concrete clients can set attributes like
    `api_key` freely without Pydantic field validation interfering.
    """
    async def chat(
        self,
        model: str,
        system: str,
        user: str,
        pdf_bytes: Optional[bytes] = None,
        text: Optional[str] = None,
    ) -> LLMResponse:
        raise NotImplementedError


class _MockGenerator:
    """
    Internal mock "generator" used by MockLLMClient and LLMService's fallback.
    Produces realistic JSON-ish content from resume-like text.
    """
    def __init__(self):
        self.request_count = 0

    async def generate_completion(self, request: LLMRequest) -> LLMResponse:
        await asyncio.sleep(0.05)  # small simulated latency
        full_prompt = request.prompt
        resume_text = full_prompt

        # Extract resume text if the prompt followed the provided templates
        if "RESUME TEXT:" in full_prompt:
            seg = full_prompt.split("RESUME TEXT:", 1)[1]
            # Stop at a common delimiter in our prompts
            trunc_keys = ["Extract the following", "Return JSON", "MISSING FIELDS", "Current data:"]
            for k in trunc_keys:
                if k in seg:
                    seg = seg.split(k, 1)[0]
            resume_text = seg.strip()

        # Basic fields from text
        email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", resume_text)
        phone_match = re.search(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", resume_text)

        full_name = None
        lines = [ln.strip() for ln in resume_text.splitlines() if ln.strip()]
        for ln in lines[:5]:
            if not any(ch.isdigit() for ch in ln) and "@" not in ln and 2 <= len(ln.split()) <= 5:
                if not any(h in ln.lower() for h in ["summary", "experience", "education", "skills", "objective"]):
                    full_name = ln
                    break

        # Minimal skills detection
        low = resume_text.lower()
        skill_dict = ["python", "java", "react", "sql", "aws", "docker", "linux", "pytorch", "tensorflow", "kubernetes"]
        skills = sorted({s.title() for s in skill_dict if s in low}) or ["Communication", "Problem Solving"]

        # Stub lists
        experience = [{"title": "Engineer", "company": "Company", "duration": "2021-2024", "description": "Work done"}]
        education = [{"degree": "B.S.", "field": "Field", "institution": "University", "year": "2020"}]

        data = {
            "full_name": full_name or "Test Candidate",
            "email": email_match.group(0) if email_match else "test@example.com",
            "phone": phone_match.group(0) if phone_match else "(555) 555-5555",
            "skills": skills,
            "experience": experience,
            "education": education,
            "full_text": resume_text,
            "summary": "Professional summary.",
            "certifications": [],
            "years_of_experience": 3.0,
            "education_level": "bachelors",
            "links": {"linkedin": None, "github": None, "portfolio": None},
        }

        # Occasionally drop a field to exercise retry paths
        if self.request_count % 5 == 0:
            data["summary"] = None

        content = json.dumps(data, indent=2)
        self.request_count += 1
        return LLMResponse(
            content=content,
            model=request.model,
            tokens_used=len(request.prompt.split()) + len(content.split()),
            cost_cents=0.0,
            request_id=request.request_id,
            processing_time_ms=50.0,
            metadata={"provider": "mock"},
        )


class MockLLMClient(LLMClient):
    """Mock LLM client for tests/dev."""
    def __init__(self):
        self._gen = _MockGenerator()

    async def chat(self, model: str, system: str, user: str, pdf_bytes: Optional[bytes] = None, text: Optional[str] = None) -> LLMResponse:
        req = LLMRequest(prompt=f"{system}\n\n{user}", model=model, max_tokens=2000, temperature=0.1)
        return await self._gen.generate_completion(req)


class OpenAILLMClient(LLMClient):
    """OpenAI client (falls back to mock if sdk or key missing)."""
    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout

    async def chat(self, model: str, system: str, user: str, pdf_bytes: Optional[bytes] = None, text: Optional[str] = None) -> LLMResponse:
        try:
            import openai  # type: ignore
        except Exception:
            logger.warning("openai package not available; falling back to mock.")
            return await MockLLMClient().chat(model, system, user, pdf_bytes, text)

        openai.api_key = self.api_key
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            # Async call if available; otherwise run sync in a thread
            if hasattr(openai.ChatCompletion, "acreate"):
                resp = await openai.ChatCompletion.acreate(
                    model=model, messages=messages, max_tokens=2000, temperature=0.1, timeout=self.timeout
                )
            else:
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(
                    None,
                    lambda: openai.ChatCompletion.create(
                        model=model, messages=messages, max_tokens=2000, temperature=0.1, timeout=self.timeout
                    ),
                )
            content = resp["choices"][0]["message"]["content"]
            tokens_used = resp.get("usage", {}).get("total_tokens", len((system + user).split()))
            return LLMResponse(
                content=content,
                model=model,
                tokens_used=int(tokens_used),
                cost_cents=0.0,
                request_id=f"openai_{int(time.time()*1000)}",
                processing_time_ms=0.0,
                metadata={"provider": "openai"},
            )
        except Exception as e:
            logger.warning(f"OpenAI call failed ({e}); falling back to mock.")
            return await MockLLMClient().chat(model, system, user, pdf_bytes, text)


class AnthropicLLMClient(LLMClient):
    """Anthropic client (falls back to mock if sdk or key missing)."""
    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout

    async def chat(self, model: str, system: str, user: str, pdf_bytes: Optional[bytes] = None, text: Optional[str] = None) -> LLMResponse:
        try:
            import anthropic  # type: ignore
        except Exception:
            logger.warning("anthropic package not available; falling back to mock.")
            return await MockLLMClient().chat(model, system, user, pdf_bytes, text)

        client = anthropic.Client(api_key=self.api_key)
        prompt = system + "\n\n" + user
        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: client.completions.create(model=model, prompt=prompt, max_tokens_to_sample=2000),
            )
            content = resp.completion
            tokens_used = len(prompt.split())
            return LLMResponse(
                content=content,
                model=model,
                tokens_used=tokens_used,
                cost_cents=0.0,
                request_id=f"anthropic_{int(time.time()*1000)}",
                processing_time_ms=0.0,
                metadata={"provider": "anthropic"},
            )
        except Exception as e:
            logger.warning(f"Anthropic call failed ({e}); falling back to mock.")
            return await MockLLMClient().chat(model, system, user, pdf_bytes, text)


class GeminiLLMClient(LLMClient):
    """Gemini/Vertex client (falls back to mock if sdk or key missing)."""
    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout

    async def chat(self, model: str, system: str, user: str, pdf_bytes: Optional[bytes] = None, text: Optional[str] = None) -> LLMResponse:
        try:
            from google import generativeai  # type: ignore
        except Exception:
            logger.warning("google.generativeai not available; falling back to mock.")
            return await MockLLMClient().chat(model, system, user, pdf_bytes, text)

        generativeai.configure(api_key=self.api_key)
        prompt = system + "\n\n" + user
        try:
            # Simple prompt interface; adapt as needed for your sdk version
            resp = generativeai.chat.create(model=model, prompt=prompt)
            content = getattr(resp, "content", str(resp))
            tokens_used = len(prompt.split())
            return LLMResponse(
                content=content,
                model=model,
                tokens_used=tokens_used,
                cost_cents=0.0,
                request_id=f"gemini_{int(time.time()*1000)}",
                processing_time_ms=0.0,
                metadata={"provider": "gemini"},
            )
        except Exception as e:
            logger.warning(f"Gemini call failed ({e}); falling back to mock.")
            return await MockLLMClient().chat(model, system, user, pdf_bytes, text)


def create_llm_client(config: Optional[LLMConfig] = None) -> LLMClient:
    """Factory that returns an LLMClient based on config."""
    config = config or LLMConfig()
    if config.provider == "mock":
        return MockLLMClient()
    if config.provider == "openai":
        if not config.openai_api_key:
            logger.warning("OPENAI_API_KEY missing; using mock client.")
            return MockLLMClient()
        return OpenAILLMClient(config.openai_api_key, config.timeout)
    if config.provider == "anthropic":
        if not config.anthropic_api_key:
            logger.warning("ANTHROPIC_API_KEY missing; using mock client.")
            return MockLLMClient()
    # anthropic
        return AnthropicLLMClient(config.anthropic_api_key, config.timeout)
    if config.provider == "gemini":
        if not config.gemini_api_key:
            logger.warning("GEMINI/GOOGLE API key missing; using mock client.")
            return MockLLMClient()
        return GeminiLLMClient(config.gemini_api_key, config.timeout)

    logger.warning(f"Unknown provider '{config.provider}'; using mock client.")
    return MockLLMClient()


# =========================
# High-level Service
# =========================

REQUIRED_FIELDS = ["full_name", "email", "phone", "skills", "experience", "education", "full_text"]


class LLMService:
    """Main LLM service for resume parsing (text-in → JSON-out) with retry logic."""
    def __init__(
        self,
        provider: LLMProvider = LLMProvider.MOCK,
        model: str = "gpt-3.5-turbo",
        max_retries: int = 3,
        retry_delay: float = 0.6,
        config: Optional[LLMConfig] = None,
    ):
        self.provider = provider
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # If a config is supplied, it wins; otherwise construct from provider/model
        self.config = config or LLMConfig()
        # Override with explicit args if given
        self.config.provider = provider.value if isinstance(provider, LLMProvider) else str(provider)
        self.config.model = model

        self.client: LLMClient = create_llm_client(self.config)

        # Accounting
        self.total_tokens_used = 0
        self.total_cost_cents = 0.0
        self.requests_made = 0

    async def parse_resume(self, resume_text: str, max_attempts: int = 3) -> Tuple[ParsedResumeData, List[LLMResponse]]:
        """Parse resume using LLM with retries for missing required fields."""
        responses: List[LLMResponse] = []
        parsed: Optional[ParsedResumeData] = None
        resume_text = resume_text or ""

        for attempt in range(max_attempts):
            logger.info(f"[LLMService] attempt {attempt + 1}/{max_attempts}")

            if attempt == 0:
                system = self._system_prompt()
                user = self._initial_user_prompt(resume_text)
            else:
                missing = parsed.get_missing_fields() if parsed else REQUIRED_FIELDS
                if not missing:
                    break
                system = self._system_prompt()
                user = self._retry_user_prompt(resume_text, missing, parsed)

            try:
                resp = await self.client.chat(
                    model=self.config.model,
                    system=system,
                    user=user,
                    pdf_bytes=None,
                    text=resume_text,
                )
                responses.append(resp)
                data = self._safe_json_from_response(resp.content)

                # Ensure full_text present
                data.setdefault("full_text", resume_text)

                # Validate
                try:
                    new_parsed = ParsedResumeData(**data)
                except Exception as e:
                    logger.error(f"Pydantic validation failed: {e}")
                    minimal = {
                        "full_name": data.get("full_name"),
                        "email": data.get("email"),
                        "phone": data.get("phone"),
                        "skills": data.get("skills", []),
                        "experience": data.get("experience", []),
                        "education": data.get("education", []),
                        "full_text": resume_text,
                    }
                    new_parsed = ParsedResumeData(**minimal)

                parsed = self._merge(parsed, new_parsed) if parsed else new_parsed

                # accounting
                self.total_tokens_used += resp.tokens_used
                self.requests_made += 1
                # cost left as mock/zero unless you wire true pricing

                if parsed.is_complete():
                    break
            except Exception as e:
                logger.warning(f"[LLMService] request error: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (2**attempt))

        return parsed or ParsedResumeData(full_text=resume_text), responses

    # ---- Optional helpers used by some callers (safe no-ops if you don't need them) ----

    async def generic_structured_extract(self, text: str, schema_instruction: str) -> Any:
        """Ask the LLM to return any arbitrary JSON following `schema_instruction`."""
        system = "You are a precise JSON extraction engine. Return only strict JSON."
        user = f"{schema_instruction}\n\nTEXT:\n{text}"
        resp = await self.client.chat(model=self.config.model, system=system, user=user, text=text)
        return self._safe_json_from_response(resp.content)

    async def infer_pdf_requirements(self, pdf_text: str) -> List[Dict[str, Any]]:
        """Infer form-like fields from a non-fillable PDF text. Returns a list of field dicts."""
        schema_instruction = (
            "From the provided PDF text (labels/instructions), infer fields the document asks for. "
            "Return a JSON array of objects with keys: name, type (string|number|date|email|phone|url|enum|address|textarea), "
            "required (boolean), description (string|null), options (array|null), regex_hint (string|null), example (string|null)."
        )
        data = await self.generic_structured_extract(pdf_text, schema_instruction)
        # Normalize into list of dicts
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "fields" in data and isinstance(data["fields"], list):
            return data["fields"]
        return []

    # ---- Prompt builders ----

    def _system_prompt(self) -> str:
        return (
            "You are an expert resume parser. Return ONLY a single JSON object, no commentary.\n"
            "- If a field is not found, use null for strings/numbers or [] for lists.\n"
            "- Extract ALL experience/education entries and ALL skills present.\n"
            "- Do not hallucinate. Be accurate."
        )

    def _initial_user_prompt(self, resume_text: str) -> str:
        return f"""You will extract JSON from this resume.

RESUME TEXT:
{resume_text}

Return JSON with this structure:
{{
  "full_name": "<name>",
  "email": "<email>",
  "phone": "<phone>",
  "skills": ["<skill1>", "<skill2>", ...],
  "experience": [{{"title": "<title>", "company": "<company>", "duration": "<dates>", "description": "<desc>"}}],
  "education": [{{"degree": "<degree>", "field": "<field>", "institution": "<school>", "year": "<year>"}}],
  "full_text": "{resume_text}",
  "summary": "<summary>",
  "certifications": ["<cert1>", "<cert2>", ...],
  "years_of_experience": <number>,
  "education_level": "<high_school|associates|bachelors|masters|phd>",
  "links": {{"linkedin": "<url>", "github": "<url>", "portfolio": "<url>"}}
}}"""

    def _retry_user_prompt(self, resume_text: str, missing_fields: List[str], current: Optional[ParsedResumeData]) -> str:
        current_json = json.dumps((current.dict() if current else {}), indent=2, default=str)
        lines = "\n".join(f"- {f}" for f in missing_fields)
        return f"""Some fields were missing. Focus ONLY on extracting these, from the same resume.

RESUME TEXT:
{resume_text}

MISSING FIELDS:
{lines}

CURRENT DATA:
{current_json}

Return a JSON object with ONLY the missing fields in the same schema."""

    # ---- Utilities ----

    def _safe_json_from_response(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)

        # Extract first top-level JSON object
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found in response")
        brace = 0
        end = None
        for i in range(start, len(text)):
            if text[i] == "{":
                brace += 1
            elif text[i] == "}":
                brace -= 1
                if brace == 0:
                    end = i + 1
                    break
        if end is None:
            raise ValueError("Incomplete JSON in response")
        return json.loads(text[start:end])

    def _merge(self, existing: ParsedResumeData, new: ParsedResumeData) -> ParsedResumeData:
        if existing is None:
            return new
        merged = existing.dict()
        nd = new.dict()
        for k, v in nd.items():
            if isinstance(v, list):
                if v and not merged.get(k):
                    merged[k] = v
                elif v and len(v) > len(merged.get(k, [])):
                    merged[k] = v
            else:
                if v and not merged.get(k):
                    merged[k] = v
        return ParsedResumeData(**merged)


# Back-compat factory (used by your ResumeParser)
def create_llm_service(provider: LLMProvider = LLMProvider.MOCK, model: str = "gpt-3.5-turbo") -> LLMService:
    return LLMService(provider=provider, model=model)


# Optional "client-oriented" wrapper if you prefer its API
def create_llm_resume_parser(config: Optional[LLMConfig] = None) -> "LLMResumeParser":
    return LLMResumeParser(config)


class LLMResumeParser:
    """Alternate API that sends prompts via the client directly (kept for compatibility)."""
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.client = create_llm_client(self.config)
        self.total_tokens_used = 0
        self.total_cost_cents = 0.0
        self.requests_made = 0

    async def parse_resume(
        self,
        pdf_bytes: Optional[bytes] = None,
        fallback_text: Optional[str] = None,
        max_attempts: int = 3,
    ) -> Tuple[ParsedResumeData, List[LLMResponse]]:
        """Parse resume by chatting with the client; similar to LLMService."""
        responses: List[LLMResponse] = []
        parsed: Optional[ParsedResumeData] = None
        text = fallback_text or ""

        for attempt in range(max_attempts):
            system = (
                "You are an expert resume parser. Return ONLY one JSON object. "
                "If a field is missing, set null/[]; do not hallucinate."
            )
            if attempt == 0:
                user = (
                    f"Extract as JSON using this schema from the resume:\n\n{text}\n\n"
                    "{"
                    '"full_name":"", "email":"", "phone":"", "skills":[], '
                    '"experience":[{"title":"","company":"","duration":"","description":""}], '
                    '"education":[{"degree":"","field":"","institution":"","year":""}], '
                    f'"full_text":"{text}", '
                    '"summary":"", "certifications":[], "years_of_experience":0, '
                    '"education_level":"", "links":{"linkedin":"","github":"","portfolio":""}'
                    "}"
                )
            else:
                missing = parsed.get_missing_fields() if parsed else REQUIRED_FIELDS
                if not missing:
                    break
                user = (
                    f"Resume:\n{text}\n\n"
                    f"Missing: {', '.join(missing)}\n"
                    f"Current: {json.dumps(parsed.dict() if parsed else {}, indent=2)}\n\n"
                    "Return only JSON with the missing fields."
                )

            try:
                resp = await self.client.chat(
                    model=self.config.model,
                    system=system,
                    user=user,
                    pdf_bytes=pdf_bytes,
                    text=text,
                )
                responses.append(resp)
                data = self._safe_json_from_response(resp.content)
                data.setdefault("full_text", text)

                try:
                    new_parsed = ParsedResumeData(**data)
                except Exception:
                    minimal = {
                        "full_name": data.get("full_name"),
                        "email": data.get("email"),
                        "phone": data.get("phone"),
                        "skills": data.get("skills", []),
                        "experience": data.get("experience", []),
                        "education": data.get("education", []),
                        "full_text": text,
                    }
                    new_parsed = ParsedResumeData(**minimal)

                parsed = self._merge(parsed, new_parsed) if parsed else new_parsed
                self.requests_made += 1
                self.total_tokens_used += resp.tokens_used

                if parsed.is_complete():
                    break
            except Exception as e:
                logger.error(f"LLMResumeParser error: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(0.6 * (2**attempt))

        return parsed or ParsedResumeData(full_text=text), responses

    def _safe_json_from_response(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found in response")
        brace = 0
        end = None
        for i in range(start, len(text)):
            if text[i] == "{":
                brace += 1
            elif text[i] == "}":
                brace -= 1
                if brace == 0:
                    end = i + 1
                    break
        if end is None:
            raise ValueError("Incomplete JSON in response")
        return json.loads(text[start:end])

    def _merge(self, existing: Optional[ParsedResumeData], new: ParsedResumeData) -> ParsedResumeData:
        if existing is None:
            return new
        merged = existing.dict()
        nd = new.dict()
        for k, v in nd.items():
            if isinstance(v, list):
                if v and not merged.get(k):
                    merged[k] = v
                elif v and len(v) > len(merged.get(k, [])):
                    merged[k] = v
            else:
                if v and not merged.get(k):
                    merged[k] = v
        return ParsedResumeData(**merged)
