"""
Enhanced LLM service with real OpenAI and Gemini support for comprehensive resume parsing.
Removes all mock implementations and provides production-ready LLM integration.
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator

# Load environment variables
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

class LLMConfig:
    """LLM configuration from environment variables."""
    def __init__(self):
        # Provider & model
        self.provider = os.getenv("LJS_LLM_PROVIDER", "openai").lower()
        
        # Model selection based on provider
        if self.provider == "openai":
            self.model = os.getenv("LJS_LLM_MODEL", "gpt-4-turbo-preview")
        elif self.provider == "gemini":
            self.model = os.getenv("LJS_LLM_MODEL", "gemini-1.5-pro")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        
        # Timeouts/limits
        self.timeout = int(os.getenv("LJS_LLM_TIMEOUT", "60"))
        self.max_tokens = int(os.getenv("LJS_LLM_MAX_TOKENS", "4000"))
        
        # API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        # Validate API keys
        if self.provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider")
        if self.provider == "gemini" and not self.gemini_api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable is required for Gemini provider")

@dataclass
class LLMRequest:
    """Request for LLM processing."""
    prompt: str
    model: str
    max_tokens: int = 4000
    temperature: float = 0.1
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.request_id is None:
            import hashlib
            content_hash = hashlib.sha256(f"{self.prompt}:{self.model}".encode("utf-8")).hexdigest()
            self.request_id = f"llm_{content_hash[:16]}"

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
# Enhanced Resume Data Model
# =========================

class ExperienceItem(BaseModel):
    title: Optional[str] = Field(None, description="Job title")
    company: Optional[str] = Field(None, description="Company name")
    duration: Optional[str] = Field(None, description="Employment duration")
    description: Optional[str] = Field(None, description="Job description")
    location: Optional[str] = Field(None, description="Job location")
    bullets: List[str] = Field(default_factory=list, description="Achievement bullets")
    technologies: List[str] = Field(default_factory=list, description="Technologies used")

class EducationItem(BaseModel):
    degree: Optional[str] = Field(None, description="Degree type")
    field: Optional[str] = Field(None, description="Field of study")
    institution: Optional[str] = Field(None, description="School/University name")
    year: Optional[str] = Field(None, description="Graduation year or date range")
    gpa: Optional[str] = Field(None, description="GPA if mentioned")
    specialization: Optional[str] = Field(None, description="Specialization or concentration")

class CertificationItem(BaseModel):
    name: str = Field(..., description="Certification name")
    issuer: Optional[str] = Field(None, description="Issuing organization")
    date: Optional[str] = Field(None, description="Date obtained")

class ProjectItem(BaseModel):
    name: Optional[str] = Field(None, description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    technologies: List[str] = Field(default_factory=list, description="Technologies used")
    link: Optional[str] = Field(None, description="Project link")

class Links(BaseModel):
    linkedin: Optional[str] = Field(None, description="LinkedIn profile URL")
    github: Optional[str] = Field(None, description="GitHub profile URL")
    portfolio: Optional[str] = Field(None, description="Portfolio website URL")
    other: List[str] = Field(default_factory=list, description="Other relevant links")

class Skills(BaseModel):
    programming: List[str] = Field(default_factory=list, description="Programming languages")
    frameworks: List[str] = Field(default_factory=list, description="Frameworks and libraries")
    tools: List[str] = Field(default_factory=list, description="Tools and platforms")
    databases: List[str] = Field(default_factory=list, description="Databases")
    cloud: List[str] = Field(default_factory=list, description="Cloud platforms")
    ml_ai: List[str] = Field(default_factory=list, description="ML/AI technologies")
    other: List[str] = Field(default_factory=list, description="Other technical skills")
    all: List[str] = Field(default_factory=list, description="All skills combined")

class ParsedResumeData(BaseModel):
    """Enhanced structured resume data with comprehensive fields."""
    # Core fields
    full_name: Optional[str] = Field(None, description="Full name")
    email: Optional[str] = Field(None, description="Primary email")
    phone: Optional[str] = Field(None, description="Phone number")
    location: Optional[str] = Field(None, description="Location/City")
    
    # Professional summary
    summary: Optional[str] = Field(None, description="Professional summary or objective")
    titles: List[str] = Field(default_factory=list, description="Professional titles mentioned")
    
    # Skills - both simple list and structured
    skills: List[str] = Field(default_factory=list, description="All skills as simple list")
    skills_structured: Skills = Field(default_factory=Skills, description="Categorized skills")
    
    # Experience
    experience: List[ExperienceItem] = Field(default_factory=list, description="Work experience")
    
    # Education
    education: List[EducationItem] = Field(default_factory=list, description="Educational background")
    
    # Additional sections
    certifications: List[CertificationItem] = Field(default_factory=list, description="Certifications")
    projects: List[ProjectItem] = Field(default_factory=list, description="Projects")
    awards: List[str] = Field(default_factory=list, description="Awards and achievements")
    publications: List[str] = Field(default_factory=list, description="Publications")
    
    # Links and profiles
    links: Links = Field(default_factory=Links, description="Links and profiles")
    
    # Computed/derived fields
    years_of_experience: Optional[float] = Field(None, description="Total years of experience")
    education_level: Optional[str] = Field(None, description="Highest education level")
    clearance_level: Optional[str] = Field(None, description="Security clearance if mentioned")
    
    # Full text for reference
    full_text: Optional[str] = Field(None, description="Complete resume text")
    
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

    def get_missing_fields(self) -> List[str]:
        """Check for critical missing fields."""
        missing = []
        required = ["full_name", "email", "skills", "experience", "education"]
        for f in required:
            value = getattr(self, f, None)
            if not value or (isinstance(value, list) and not value):
                missing.append(f)
        return missing

    def is_complete(self) -> bool:
        return len(self.get_missing_fields()) == 0

# =========================
# LLM Client Implementations
# =========================

class LLMClient:
    """Abstract LLM client interface."""
    async def chat(self, model: str, system: str, user: str) -> LLMResponse:
        raise NotImplementedError

class OpenAILLMClient(LLMClient):
    """OpenAI client implementation."""
    def __init__(self, api_key: str, timeout: int = 60):
        self.api_key = api_key
        self.timeout = timeout
        
    async def chat(self, model: str, system: str, user: str) -> LLMResponse:
        try:
            import openai
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")
        
        start_time = time.time()
        client = AsyncOpenAI(api_key=self.api_key, timeout=self.timeout)
        
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                max_tokens=4000,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else len((system + user).split())
            
            # Approximate cost calculation for GPT-4
            if "gpt-4" in model:
                cost_cents = (tokens_used / 1000) * 0.03  # ~$0.03 per 1k tokens
            else:
                cost_cents = (tokens_used / 1000) * 0.002  # ~$0.002 per 1k tokens for GPT-3.5
            
            processing_time = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=content,
                model=model,
                tokens_used=tokens_used,
                cost_cents=cost_cents,
                request_id=f"openai_{int(time.time()*1000)}",
                processing_time_ms=processing_time,
                metadata={"provider": "openai"}
            )
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

class GeminiLLMClient(LLMClient):
    """Google Gemini client implementation."""
    def __init__(self, api_key: str, timeout: int = 60):
        self.api_key = api_key
        self.timeout = timeout
        
    async def chat(self, model: str, system: str, user: str) -> LLMResponse:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai package is required. Install with: pip install google-generativeai")
        
        start_time = time.time()
        genai.configure(api_key=self.api_key)
        
        try:
            # Create the model
            model_obj = genai.GenerativeModel(model)
            
            # Combine system and user prompts for Gemini
            full_prompt = f"{system}\n\n{user}"
            
            # Generate response
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                model_obj.generate_content, 
                full_prompt
            )
            
            content = response.text
            # Estimate tokens (Gemini doesn't provide exact count in basic API)
            tokens_used = len(full_prompt.split()) + len(content.split())
            
            # Gemini pricing (approximate)
            cost_cents = (tokens_used / 1000) * 0.001  # ~$0.001 per 1k tokens
            
            processing_time = (time.time() - start_time) * 1000
            
            return LLMResponse(
                content=content,
                model=model,
                tokens_used=tokens_used,
                cost_cents=cost_cents,
                request_id=f"gemini_{int(time.time()*1000)}",
                processing_time_ms=processing_time,
                metadata={"provider": "gemini"}
            )
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise

def create_llm_client(config: Optional[LLMConfig] = None) -> LLMClient:
    """Factory to create appropriate LLM client."""
    config = config or LLMConfig()
    
    if config.provider == "openai":
        return OpenAILLMClient(config.openai_api_key, config.timeout)
    elif config.provider == "gemini":
        return GeminiLLMClient(config.gemini_api_key, config.timeout)
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")

# =========================
# Enhanced LLM Service
# =========================

class LLMService:
    """Enhanced LLM service for comprehensive resume parsing."""
    
    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
        config: Optional[LLMConfig] = None
    ):
        self.config = config or LLMConfig()
        if provider:
            self.config.provider = provider.value if isinstance(provider, LLMProvider) else str(provider)
        if model:
            self.config.model = model
            
        self.client = create_llm_client(self.config)
        self.max_retries = max_retries
        
        # Accounting
        self.total_tokens_used = 0
        self.total_cost_cents = 0.0
        self.requests_made = 0

    async def parse_resume(self, resume_text: str, max_attempts: int = 3) -> Tuple[ParsedResumeData, List[LLMResponse]]:
        """Parse resume with comprehensive extraction."""
        responses: List[LLMResponse] = []
        parsed: Optional[ParsedResumeData] = None
        
        for attempt in range(max_attempts):
            logger.info(f"[LLMService] Resume parsing attempt {attempt + 1}/{max_attempts}")
            
            if attempt == 0:
                system = self._comprehensive_system_prompt()
                user = self._comprehensive_user_prompt(resume_text)
            else:
                missing = parsed.get_missing_fields() if parsed else []
                if not missing:
                    break
                system = self._comprehensive_system_prompt()
                user = self._retry_user_prompt(resume_text, missing, parsed)
            
            try:
                resp = await self.client.chat(
                    model=self.config.model,
                    system=system,
                    user=user
                )
                responses.append(resp)
                
                # Parse JSON response
                data = self._safe_json_from_response(resp.content)
                data.setdefault("full_text", resume_text)
                
                # Create ParsedResumeData
                try:
                    new_parsed = ParsedResumeData(**data)
                except Exception as e:
                    logger.error(f"Validation error: {e}")
                    # Create with available data
                    new_parsed = ParsedResumeData(
                        full_text=resume_text,
                        **{k: v for k, v in data.items() if k in ParsedResumeData.__fields__}
                    )
                
                parsed = self._merge(parsed, new_parsed) if parsed else new_parsed
                
                # Update accounting
                self.total_tokens_used += resp.tokens_used
                self.total_cost_cents += resp.cost_cents
                self.requests_made += 1
                
                if parsed.is_complete():
                    logger.info(f"Resume parsing complete with {len(parsed.skills)} skills extracted")
                    break
                    
            except Exception as e:
                logger.error(f"Parse attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(1.0 * (2 ** attempt))
        
        return parsed or ParsedResumeData(full_text=resume_text), responses

    def _comprehensive_system_prompt(self) -> str:
        return """You are an expert resume parser specializing in technical resumes. 
Extract ALL information with extreme precision. Pay special attention to:
1. ALL technical skills, tools, frameworks, languages, and technologies mentioned anywhere
2. Complete work experience with all bullet points and achievements
3. All educational degrees and certifications
4. Any security clearances mentioned
5. Awards, publications, and projects

Return ONLY valid JSON. Do not skip or summarize any information.
For skills, extract every single technology, tool, framework, library, and platform mentioned.
Include parenthetical lists (e.g., "Python (pandas, numpy, scikit-learn)" should extract Python, pandas, numpy, scikit-learn as separate skills)."""

    def _comprehensive_user_prompt(self, resume_text: str) -> str:
        return f"""Extract ALL information from this resume into the exact JSON structure below.
        
CRITICAL: Extract EVERY skill, technology, tool, framework, and platform mentioned.
If skills are listed with examples in parentheses, extract both the main skill and all examples as separate skills.

RESUME TEXT:
{resume_text}

Required JSON structure:
{{
  "full_name": "string",
  "email": "string",
  "phone": "string",
  "location": "string or null",
  "summary": "string or null",
  "titles": ["list of professional titles mentioned"],
  "skills": ["COMPLETE list of ALL skills, technologies, tools mentioned"],
  "skills_structured": {{
    "programming": ["programming languages"],
    "frameworks": ["frameworks and libraries"],
    "tools": ["development tools and platforms"],
    "databases": ["database technologies"],
    "cloud": ["cloud platforms"],
    "ml_ai": ["ML/AI related technologies"],
    "other": ["other technical skills"],
    "all": ["complete combined list"]
  }},
  "experience": [
    {{
      "title": "job title",
      "company": "company name",
      "duration": "date range",
      "location": "location",
      "description": "role description",
      "bullets": ["list", "of", "all", "achievement", "bullets"],
      "technologies": ["technologies used in this role"]
    }}
  ],
  "education": [
    {{
      "degree": "degree type",
      "field": "field of study",
      "institution": "school name",
      "year": "graduation year or range",
      "gpa": "GPA if mentioned",
      "specialization": "any specialization mentioned"
    }}
  ],
  "certifications": [
    {{"name": "cert name", "issuer": "issuing org", "date": "date"}}
  ],
  "projects": [
    {{"name": "project", "description": "description", "technologies": ["tech used"], "link": "url or null"}}
  ],
  "awards": ["list of awards and honors"],
  "publications": ["list of publications"],
  "links": {{
    "linkedin": "linkedin url or null",
    "github": "github url or null",
    "portfolio": "portfolio url or null",
    "other": ["other urls"]
  }},
  "years_of_experience": null or number,
  "education_level": "highest degree",
  "clearance_level": "security clearance if mentioned or null"
}}"""

    def _retry_user_prompt(self, resume_text: str, missing_fields: List[str], current: Optional[ParsedResumeData]) -> str:
        current_json = json.dumps(current.dict() if current else {}, indent=2, default=str)
        return f"""Some critical fields are missing. Extract them from the resume.

RESUME TEXT:
{resume_text}

MISSING FIELDS: {', '.join(missing_fields)}

CURRENT EXTRACTED DATA:
{current_json}

Return JSON with the missing fields filled in. Be comprehensive - don't skip any information."""

    def _safe_json_from_response(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response."""
        text = text.strip()
        
        # Try direct JSON parse
        try:
            return json.loads(text)
        except:
            pass
        
        # Extract JSON block from markdown
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                try:
                    return json.loads(text[start:end].strip())
                except:
                    pass
        
        # Find first JSON object
        start = text.find("{")
        if start >= 0:
            brace_count = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            return json.loads(text[start:i+1])
                        except:
                            pass
        
        raise ValueError("No valid JSON found in response")

    def _merge(self, existing: ParsedResumeData, new: ParsedResumeData) -> ParsedResumeData:
        """Merge two parsed resume objects."""
        if not existing:
            return new
            
        merged = existing.dict()
        new_dict = new.dict()
        
        for key, value in new_dict.items():
            if isinstance(value, list):
                # Merge lists, avoiding duplicates
                existing_list = merged.get(key, [])
                if value and len(value) > len(existing_list):
                    merged[key] = value
            elif value and not merged.get(key):
                merged[key] = value
                
        return ParsedResumeData(**merged)

# Factory function for backward compatibility
def create_llm_service(provider: Optional[LLMProvider] = None, model: Optional[str] = None) -> LLMService:
    return LLMService(provider=provider, model=model)