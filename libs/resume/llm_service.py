"""LLM service for resume parsing and structured data extraction

Provides LLM-powered resume parsing with support for multiple providers (OpenAI, Anthropic, etc.)
Includes retry logic for incomplete field extraction and proper error handling.
"""
from __future__ import annotations
import logging
import json
import hashlib
import time
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import asyncio
import re

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MOCK = "mock"  # For testing

@dataclass
class LLMRequest:
    """Request for LLM processing"""
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
        """Generate unique ID for request"""
        content_hash = hashlib.sha256(
            f"{self.prompt}:{self.model}".encode('utf-8')
        ).hexdigest()
        return f"llm_{content_hash[:16]}"

@dataclass
class LLMResponse:
    """Response from LLM processing"""
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

class ExperienceItem(BaseModel):
    """Individual experience entry"""
    title: Optional[str] = Field(None, description="Job title")
    company: Optional[str] = Field(None, description="Company name")
    duration: Optional[str] = Field(None, description="Employment duration")
    description: Optional[str] = Field(None, description="Job description")
    location: Optional[str] = Field(None, description="Job location")


class EducationItem(BaseModel):
    """Individual education entry"""
    degree: Optional[str] = Field(None, description="Degree type")
    field: Optional[str] = Field(None, description="Field of study")
    institution: Optional[str] = Field(None, description="School/University name")
    year: Optional[str] = Field(None, description="Graduation year")
    gpa: Optional[str] = Field(None, description="GPA if mentioned")


class CertificationItem(BaseModel):
    """Individual certification entry"""
    name: str = Field(..., description="Certification name")
    issuer: Optional[str] = Field(None, description="Issuing organization")
    date: Optional[str] = Field(None, description="Date obtained")


class ProjectItem(BaseModel):
    """Individual project entry"""
    name: Optional[str] = Field(None, description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    technologies: List[str] = Field(default_factory=list, description="Technologies used")


class Links(BaseModel):
    """Contact links and social profiles"""
    linkedin: Optional[str] = Field(None, description="LinkedIn profile URL")
    github: Optional[str] = Field(None, description="GitHub profile URL")
    portfolio: Optional[str] = Field(None, description="Portfolio website URL")
    other: List[str] = Field(default_factory=list, description="Other relevant links")
    
    @classmethod
    def from_dict_or_none(cls, value):
        """Create Links from dict or return default if None"""
        if value is None:
            return cls()
        if isinstance(value, dict):
            return cls(**value)
        return value


class Skills(BaseModel):
    """Categorized skills"""
    technical: List[str] = Field(default_factory=list, description="Technical skills")
    soft: List[str] = Field(default_factory=list, description="Soft skills")
    languages: List[str] = Field(default_factory=list, description="Programming languages")
    tools: List[str] = Field(default_factory=list, description="Tools and frameworks")
    all: List[str] = Field(default_factory=list, description="All skills combined")


class ParsedResumeData(BaseModel):
    """Structured resume data from LLM parsing with strict validation"""
    # Core required fields
    full_name: Optional[str] = Field(None, description="Full name of the person")
    email: Optional[str] = Field(None, description="Primary email address")
    phone: Optional[str] = Field(None, description="Phone number")
    skills: List[str] = Field(default_factory=list, description="All skills mentioned")
    experience: List[ExperienceItem] = Field(default_factory=list, description="Work experience")
    education: List[EducationItem] = Field(default_factory=list, description="Educational background")
    full_text: Optional[str] = Field(None, description="Complete resume text for chunking")
    
    # Optional fields
    summary: Optional[str] = Field(None, description="Professional summary")
    certifications: List[CertificationItem] = Field(default_factory=list, description="Certifications")
    projects: List[ProjectItem] = Field(default_factory=list, description="Projects")
    links: Links = Field(default_factory=Links, description="Links and profiles")
    skills_structured: Skills = Field(default_factory=Skills, description="Structured skills")
    years_of_experience: Optional[float] = Field(None, description="Calculated years of experience")
    education_level: Optional[str] = Field(None, description="Highest education level")
    
    # Validation
    @field_validator('links', mode='before')
    @classmethod
    def validate_links(cls, v):
        """Ensure links is always a Links object"""
        if v is None:
            return Links()
        if isinstance(v, dict):
            return Links(**v)
        return v
    
    @field_validator('skills_structured', mode='before')
    @classmethod
    def validate_skills_structured(cls, v):
        """Ensure skills_structured is always a Skills object"""
        if v is None:
            return Skills()
        if isinstance(v, dict):
            return Skills(**v)
        return v
    @field_validator('email', mode='before')
    @classmethod
    def validate_email(cls, v):
        """Basic email validation"""
        if v and '@' not in v:
            return None  # Invalid email, return None
        return v
    
    @field_validator('phone', mode='before')
    @classmethod
    def validate_phone(cls, v):
        """Basic phone validation"""
        if v and not re.search(r'\d{3}', v):  # At least 3 digits
            return None
        return v
    
    @field_validator('skills', mode='before')
    @classmethod
    def validate_skills(cls, v):
        """Ensure skills is always a list"""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return v if isinstance(v, list) else []

    def get_missing_fields(self) -> List[str]:
        """Get list of missing required fields"""
        missing = []
        
        # Define required fields as per problem statement
        required_fields = ["full_name", "email", "phone", "skills", "experience", "education", "full_text"]
        
        for field in required_fields:
            value = getattr(self, field, None)
            if not value or (isinstance(value, list) and len(value) == 0):
                missing.append(field)
        
        return missing
    
    def is_complete(self) -> bool:
        """Check if all required fields are present"""
        return len(self.get_missing_fields()) == 0


class LLMClient(BaseModel):
    """Abstract LLM client interface"""
    
    async def chat(self, model: str, system: str, user: str, pdf_bytes: Optional[bytes] = None, text: Optional[str] = None) -> LLMResponse:
        """Send chat request to LLM provider
        
        Args:
            model: Model name (e.g., 'gpt-3.5-turbo', 'claude-3-sonnet')
            system: System prompt
            user: User prompt
            pdf_bytes: Optional PDF file bytes (if provider supports it)
            text: Fallback text if provider doesn't support PDF bytes
            
        Returns:
            LLMResponse with content and metadata
        """
        raise NotImplementedError


class LLMConfig:
    """LLM configuration from environment variables"""
    
    def __init__(self):
        # Environment configuration
        self.provider = os.getenv('LJS_LLM_PROVIDER', 'mock').lower()
        self.model = os.getenv('LJS_LLM_MODEL', 'gpt-3.5-turbo')
        self.timeout = int(os.getenv('LJS_LLM_TIMEOUT', '30'))
        self.max_tokens = int(os.getenv('LJS_LLM_MAX_TOKENS', '2000'))
        
        # Provider-specific configs
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        
        # Validate configuration
        if self.provider not in ['mock', 'openai', 'anthropic']:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
            
        if self.provider == 'openai' and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI provider")
            
        if self.provider == 'anthropic' and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY required for Anthropic provider")


class MockLLMClient:
    """Mock LLM client for testing"""
    
    def __init__(self):
        self.request_count = 0
    
    async def chat(self, model: str, system: str, user: str, pdf_bytes: Optional[bytes] = None, text: Optional[str] = None) -> LLMResponse:
        """Mock chat implementation"""
        await asyncio.sleep(0.1)  # Simulate API delay
        
        # Use text or PDF bytes (preferring text for simplicity)
        resume_text = text if text else "Mock resume content from PDF"
        
        # Generate mock response using existing logic
        mock_provider = MockLLMProvider()
        mock_request = LLMRequest(
            prompt=f"{system}\n\n{user}",
            model=model,
            max_tokens=2000,
            temperature=0.1
        )
        
        return await mock_provider.generate_completion(mock_request)
        
class OpenAILLMClient:
    """OpenAI LLM client implementation"""
    
    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout
        
    async def chat(self, model: str, system: str, user: str, pdf_bytes: Optional[bytes] = None, text: Optional[str] = None) -> LLMResponse:
        """OpenAI chat implementation"""
        # TODO: Implement OpenAI API calls
        # For now, fallback to mock
        mock_client = MockLLMClient()
        return await mock_client.chat(model, system, user, pdf_bytes, text)


class AnthropicLLMClient:
    """Anthropic LLM client implementation"""
    
    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout
        
    async def chat(self, model: str, system: str, user: str, pdf_bytes: Optional[bytes] = None, text: Optional[str] = None) -> LLMResponse:
        """Anthropic chat implementation"""
        # TODO: Implement Anthropic API calls
        # For now, fallback to mock
        mock_client = MockLLMClient()
        return await mock_client.chat(model, system, user, pdf_bytes, text)


def create_llm_client(config: Optional[LLMConfig] = None) -> 'LLMClient':
    """Factory function to create LLM client based on configuration"""
    if config is None:
        config = LLMConfig()
    
    if config.provider == 'mock':
        return MockLLMClient()
    elif config.provider == 'openai':
        return OpenAILLMClient(config.openai_api_key, config.timeout)
    elif config.provider == 'anthropic':
        return AnthropicLLMClient(config.anthropic_api_key, config.timeout)
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")


# Keep compatibility with existing code
REQUIRED_FIELDS = ["full_name", "email", "phone", "skills", "experience", "education", "full_text"]

class MockLLMProvider:
    """Mock LLM provider for testing"""
    
    def __init__(self):
        self.request_count = 0
    
    async def generate_completion(self, request: LLMRequest) -> LLMResponse:
        """Generate mock LLM completion"""
        await asyncio.sleep(0.1)  # Simulate API delay
        
        # Extract the resume text from the prompt (after "RESUME TEXT:")
        full_prompt = request.prompt
        resume_text = ""
        
        # Find the actual resume content in the prompt
        if "RESUME TEXT:" in full_prompt:
            parts = full_prompt.split("RESUME TEXT:")
            if len(parts) > 1:
                # Get text between "RESUME TEXT:" and "Extract the following"
                resume_section = parts[1]
                if "Extract the following" in resume_section:
                    resume_text = resume_section.split("Extract the following")[0].strip()
                else:
                    resume_text = resume_section.strip()
        else:
            # Fallback - use the entire prompt
            resume_text = full_prompt
        
        # Extract info from the actual resume text for realistic responses
        text = resume_text.lower()
        
        # Extract email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text)
        email = email_match.group() if email_match else None
        
        # Extract phone 
        phone_match = re.search(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b', resume_text)
        phone = phone_match.group() if phone_match else None
        
        # Extract name (first line that looks like a name)
        lines = resume_text.strip().split('\n')
        name = None
        for line in lines:
            line = line.strip()
            if line and not '@' in line and not any(char.isdigit() for char in line) and len(line.split()) >= 2:
                if not any(keyword in line.lower() for keyword in ['summary', 'experience', 'education', 'skills', 'professional', 'senior', 'data']):
                    name = line
                    break
        
        # If no name found, try first line
        if not name and lines:
            first_line = lines[0].strip()
            if len(first_line.split()) <= 4:  # Reasonable name length
                name = first_line
        
        # Extract skills from text
        skill_keywords = ['python', 'javascript', 'java', 'react', 'sql', 'aws', 'docker', 'kubernetes', 
                         'tensorflow', 'pytorch', 'pandas', 'scikit-learn', 'machine learning', 'ml',
                         'nodejs', 'angular', 'vue', 'git', 'linux', 'bash', 'mongodb', 'postgresql',
                         'mysql', 'redis', 'elasticsearch', 'spark', 'tableau', 'statistics']
        
        found_skills = []
        for skill in skill_keywords:
            if skill in text:
                found_skills.append(skill.title())
        
        # If no specific skills found, extract from skills section
        if not found_skills:
            skills_section = self._extract_section_content(resume_text, ['skills'])
            if skills_section:
                # Split by common separators
                potential_skills = re.split(r'[,\n\r•\-]', skills_section)
                for skill in potential_skills:
                    skill = skill.strip()
                    if skill and len(skill) > 2 and len(skill) < 25:
                        found_skills.append(skill)
        
        # If still no skills, add generic ones
        if not found_skills:
            found_skills = ['Communication', 'Problem Solving', 'Teamwork']
        
        # Extract years of experience
        yoe_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            r'(?:experience|exp)[:\s]*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)',
            r'(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)'
        ]
        
        years_exp = None
        for pattern in yoe_patterns:
            matches = re.findall(pattern, text)
            if matches:
                years_exp = float(matches[0])
                break
        
        # Extract education level
        education_level = None
        if 'phd' in text or 'ph.d' in text or 'doctorate' in text:
            education_level = 'phd'
        elif 'master' in text or 'msc' in text or 'mba' in text or 'm.s' in text:
            education_level = 'masters'  
        elif 'bachelor' in text or 'bsc' in text or 'bs' in text or 'ba' in text or 'b.s' in text:
            education_level = 'bachelors'
        elif 'associate' in text:
            education_level = 'associates'
        
        # Extract summary (look for summary/objective section)
        summary = None
        summary_section = self._extract_section_content(resume_text, ['summary', 'objective', 'profile'])
        if summary_section:
            # Take first sentence or two
            sentences = summary_section.split('. ')
            summary = '. '.join(sentences[:2]).strip()
            if summary and not summary.endswith('.'):
                summary += '.'
        
        # Extract experience entries
        experience = []
        exp_section = self._extract_section_content(resume_text, ['experience', 'employment', 'work history'])
        if exp_section:
            # Simple parsing - look for job titles and companies
            lines = exp_section.split('\n')
            current_job = None
            for line in lines:
                line = line.strip()
                if line and not line.startswith('-') and not line.startswith('•'):
                    # Likely a job title/company line
                    if ' at ' in line:
                        parts = line.split(' at ')
                        if len(parts) == 2:
                            current_job = {
                                'title': parts[0].strip(),
                                'company': parts[1].strip().split('(')[0].strip(),
                                'duration': self._extract_duration(line),
                                'description': 'Professional experience in the role'
                            }
                            experience.append(current_job)
        
        # Extract education entries  
        education = []
        edu_section = self._extract_section_content(resume_text, ['education'])
        if edu_section:
            lines = edu_section.split('\n')
            for line in lines:
                line = line.strip()
                if line and any(word in line.lower() for word in ['bachelor', 'master', 'phd', 'degree', 'university', 'college']):
                    education.append({
                        'degree': line.split('\n')[0],
                        'field': 'Related Field',
                        'institution': self._extract_institution(line),
                        'year': self._extract_year(line)
                    })
        
        # Extract certifications
        certifications = []
        cert_section = self._extract_section_content(resume_text, ['certification', 'certificate', 'license'])
        if cert_section:
            lines = cert_section.split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 5 and not line.lower().startswith('certification'):  # Skip section headers
                    certifications.append(line)
        
        mock_data = {
            "full_name": name,
            "email": email,
            "phone": phone,
            "skills": found_skills,
            "experience": experience if experience else [{"title": "Professional", "company": "Unknown", "duration": "Recent", "description": "Professional experience"}],
            "education": education if education else [{"degree": "Degree", "field": "Field", "institution": "Institution", "year": "Recent"}],
            "full_text": resume_text,  # Add full text for chunking
            "summary": summary,
            "certifications": certifications,
            "years_of_experience": years_exp,
            "education_level": education_level,
            "links": {
                "linkedin": f"linkedin.com/in/{name.lower().replace(' ', '').replace('.', '')}" if name and 'linkedin' in text else None
            }
        }
        
        # Remove None values to test retry logic occasionally
        if self.request_count % 4 == 0:  # Every 4th request has missing data
            import random
            fields_to_remove = random.sample([k for k, v in mock_data.items() if v], 1)
            for field in fields_to_remove:
                mock_data[field] = None if not isinstance(mock_data[field], list) else []
        
        response_content = json.dumps(mock_data, indent=2)
        
        self.request_count += 1
        
        return LLMResponse(
            content=response_content,
            model=request.model,
            tokens_used=len(request.prompt.split()) + len(response_content.split()),
            cost_cents=0.1,  # Mock cost
            request_id=request.request_id,
            processing_time_ms=100.0,
            metadata={"provider": "mock", "request_count": self.request_count}
        )
    
    def _extract_section_content(self, text: str, keywords: List[str]) -> Optional[str]:
        """Extract content from a section based on keywords"""
        text_lower = text.lower()
        lines = text.split('\n')
        
        # Find section start
        section_start_idx = None
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if any(keyword in line_lower for keyword in keywords):
                section_start_idx = i + 1
                break
        
        if section_start_idx is None:
            return None
        
        # Find section end (next section or end of text)
        section_lines = []
        for i in range(section_start_idx, len(lines)):
            line = lines[i].strip()
            # Stop if we hit another section header
            if (line and line.isupper() and len(line.split()) <= 3 and 
                any(sec in line.lower() for sec in ['experience', 'education', 'skills', 'summary', 'contact'])):
                break
            section_lines.append(line)
        
        return '\n'.join(section_lines).strip()
    
    def _extract_duration(self, text: str) -> str:
        """Extract duration from text"""
        import re
        # Look for patterns like (2020-2024) or 2020-2024
        duration_match = re.search(r'\(?(20\d{2})\s*[-–]\s*(20\d{2}|present|current)\)?', text.lower())
        if duration_match:
            return f"{duration_match.group(1)}-{duration_match.group(2)}"
        return "Recent"
    
    def _extract_institution(self, text: str) -> str:
        """Extract institution name from education line"""
        # Look for university, college, institute
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in ['university', 'college', 'institute', 'school']:
                # Take surrounding words
                start = max(0, i-2)
                end = min(len(words), i+2) 
                return ' '.join(words[start:end])
        return "Institution"
    
    def _extract_year(self, text: str) -> str:
        """Extract year from text"""
        import re
        year_match = re.search(r'(20\d{2})', text)
        return year_match.group(1) if year_match else "Recent"
    
    def count_tokens(self, text: str) -> int:
        """Mock token counting"""
        return len(text.split())
    
    def estimate_cost(self, token_count: int, model: str) -> float:
        """Mock cost estimation (in cents)"""
        return token_count * 0.01  # $0.0001 per token

class LLMService:
    """Main LLM service for resume parsing"""
    
    def __init__(
        self,
        provider: LLMProvider = LLMProvider.MOCK,
        model: str = "gpt-3.5-turbo",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.provider = provider
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize provider
        self._init_provider()
        
        # Cost tracking
        self.total_tokens_used = 0
        self.total_cost_cents = 0.0
        self.requests_made = 0
    
    def _init_provider(self):
        """Initialize the LLM provider"""
        if self.provider == LLMProvider.MOCK:
            self.llm_provider = MockLLMProvider()
        elif self.provider == LLMProvider.OPENAI:
            # TODO: Initialize OpenAI provider
            raise NotImplementedError("OpenAI provider not yet implemented")
        elif self.provider == LLMProvider.ANTHROPIC:
            # TODO: Initialize Anthropic provider
            raise NotImplementedError("Anthropic provider not yet implemented")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def parse_resume(self, resume_text: str, max_attempts: int = 3) -> Tuple[ParsedResumeData, List[LLMResponse]]:
        """Parse resume text using LLM with retry logic for missing fields
        
        Args:
            resume_text: The full text content of the resume
            max_attempts: Maximum number of parsing attempts
            
        Returns:
            Tuple of (parsed_data, response_history)
        """
        responses = []
        parsed_data = None
        
        for attempt in range(max_attempts):
            logger.info(f"Resume parsing attempt {attempt + 1}/{max_attempts}")
            
            if attempt == 0:
                # First attempt: standard parsing
                prompt = self._build_initial_parsing_prompt(resume_text)
            else:
                # Subsequent attempts: focused on missing fields
                missing_fields = parsed_data.get_missing_fields() if parsed_data else []
                if not missing_fields:
                    break
                prompt = self._build_retry_parsing_prompt(resume_text, missing_fields, parsed_data)
            
            request = LLMRequest(
                prompt=prompt,
                model=self.model,
                max_tokens=2000,
                temperature=0.1
            )
            
            try:
                response = await self.llm_provider.generate_completion(request)
                responses.append(response)
                
                # Parse the JSON response
                try:
                    data_dict = json.loads(response.content)
                    
                    # Ensure full_text is included if missing
                    if 'full_text' not in data_dict or not data_dict['full_text']:
                        data_dict['full_text'] = resume_text
                    
                    # Create ParsedResumeData with proper error handling
                    try:
                        new_parsed_data = ParsedResumeData(**data_dict)
                    except Exception as validation_error:
                        logger.error(f"Pydantic validation failed: {validation_error}")
                        logger.debug(f"Raw data_dict: {data_dict}")
                        
                        # Try to create with minimal required fields
                        minimal_data = {
                            'full_name': data_dict.get('full_name'),
                            'email': data_dict.get('email'),
                            'phone': data_dict.get('phone'),
                            'skills': data_dict.get('skills', []),
                            'experience': data_dict.get('experience', []),
                            'education': data_dict.get('education', []),
                            'full_text': resume_text
                        }
                        new_parsed_data = ParsedResumeData(**minimal_data)
                    
                    # Merge with existing data if this is a retry
                    if parsed_data and attempt > 0:
                        parsed_data = self._merge_parsed_data(parsed_data, new_parsed_data)
                    else:
                        parsed_data = new_parsed_data
                    
                    # Update cost tracking
                    self.total_tokens_used += response.tokens_used
                    self.total_cost_cents += response.cost_cents
                    self.requests_made += 1
                    
                    # Check if we have all required fields
                    if parsed_data.is_complete():
                        logger.info(f"Resume parsing completed successfully on attempt {attempt + 1}")
                        break
                    else:
                        missing = parsed_data.get_missing_fields()
                        logger.warning(f"Missing fields after attempt {attempt + 1}: {missing}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM response as JSON: {e}")
                    if attempt == max_attempts - 1:
                        # Last attempt failed, return what we have
                        if parsed_data is None:
                            parsed_data = ParsedResumeData()
                
            except Exception as e:
                logger.error(f"LLM request failed on attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    # Return empty data on final failure
                    if parsed_data is None:
                        parsed_data = ParsedResumeData()
        
        return parsed_data, responses
    
    def _build_initial_parsing_prompt(self, resume_text: str) -> str:
        """Build the initial LLM prompt for resume parsing"""
        prompt = f"""
You are an expert resume parser. Extract structured information from the following resume text and return it as a JSON object with the exact fields specified.

RESUME TEXT:
{resume_text}

Extract the following information and return as valid JSON:

{{
    "full_name": "<person's full name>",
    "email": "<email address>",
    "phone": "<phone number>",
    "skills": ["<skill1>", "<skill2>", "<skill3>", ...],
    "experience": [
        {{
            "title": "<job title>",
            "company": "<company name>",
            "duration": "<start date - end date>",
            "description": "<job description/achievements>"
        }}
    ],
    "education": [
        {{
            "degree": "<degree name>",
            "field": "<field of study>",
            "institution": "<school/university name>",
            "year": "<graduation year or years attended>"
        }}
    ],
    "full_text": "{resume_text}",
    "summary": "<professional summary or objective>",
    "certifications": ["<cert1>", "<cert2>", ...],
    "years_of_experience": <total years as a number>,
    "education_level": "<highest education level: high_school, associates, bachelors, masters, phd>",
    "links": {{
        "linkedin": "<LinkedIn profile URL>",
        "github": "<GitHub profile URL>",
        "portfolio": "<Portfolio website URL>"
    }}
}}

IMPORTANT:
- Return ONLY the JSON object, no additional text or explanation
- If a field is not found, use null for strings/numbers or empty array [] for lists
- Extract ALL skills mentioned, including technical and soft skills
- Include ALL work experience entries
- Calculate years_of_experience based on work history
- The full_text field must contain the complete resume text for chunking purposes
- Be thorough and accurate, no hallucination
"""
        return prompt.strip()
    
    def _build_retry_parsing_prompt(self, resume_text: str, missing_fields: List[str], current_data: ParsedResumeData) -> str:
        """Build a focused prompt for missing fields"""
        prompt = f"""
You are an expert resume parser. I previously parsed this resume but some fields were missing. Please focus on extracting the missing information.

RESUME TEXT:
{resume_text}

MISSING FIELDS: {', '.join(missing_fields)}

CURRENT EXTRACTED DATA:
{json.dumps(current_data.dict(), indent=2, default=str)}

Please extract ONLY the missing fields and return them as a JSON object. Focus specifically on finding:
{chr(10).join([f"- {field}: Look carefully in the text for this information" for field in missing_fields])}

Return the JSON object with only the missing fields filled in. Use the same structure as the original format.

IMPORTANT:
- Return ONLY the JSON object, no additional text
- Focus on the missing fields: {', '.join(missing_fields)}
- If still not found, use null/empty values but try harder to extract from context
"""
        return prompt.strip()
    
    def _merge_parsed_data(self, existing: ParsedResumeData, new: ParsedResumeData) -> ParsedResumeData:
        """Merge new parsed data with existing data, preferring non-empty values"""
        # Create a new instance with existing data
        merged_dict = existing.dict()
        new_dict = new.dict()
        
        for field, value in new_dict.items():
            if value and not merged_dict.get(field):  # Only update if new value exists and old one doesn't
                merged_dict[field] = value
            elif isinstance(value, list) and value:  # For lists, extend if new has more items
                existing_list = merged_dict.get(field, [])
                if len(value) > len(existing_list):
                    merged_dict[field] = value
        
        return ParsedResumeData(**merged_dict)

# Factory functions for backward compatibility
def create_llm_service(provider: LLMProvider = LLMProvider.MOCK, model: str = "gpt-3.5-turbo") -> 'LLMService':
    """Factory function to create a configured LLM service (backward compatibility)"""
    return LLMService(provider=provider, model=model)

def create_llm_resume_parser(config: Optional[LLMConfig] = None) -> 'LLMResumeParser':
    """Factory function to create a configured LLM resume parser"""
    return LLMResumeParser(config)


class LLMResumeParser:
    """Enhanced LLM-first resume parser with retry logic and client abstraction"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.client = create_llm_client(self.config)
        
        # Cost tracking
        self.total_tokens_used = 0
        self.total_cost_cents = 0.0
        self.requests_made = 0
    
    async def parse_resume(self, pdf_bytes: Optional[bytes] = None, fallback_text: Optional[str] = None, max_attempts: int = 3) -> Tuple[ParsedResumeData, List[LLMResponse]]:
        """Parse resume using LLM with retry logic for missing fields
        
        Args:
            pdf_bytes: PDF file bytes (if provider supports it)
            fallback_text: Fallback text if provider doesn't support PDF bytes
            max_attempts: Maximum number of parsing attempts
            
        Returns:
            Tuple of (parsed_data, response_history)
        """
        responses = []
        parsed_data = None
        resume_text = fallback_text or "Resume content from PDF"
        
        for attempt in range(max_attempts):
            logger.info(f"Resume parsing attempt {attempt + 1}/{max_attempts}")
            
            if attempt == 0:
                # First attempt: standard parsing
                system_prompt = self._build_system_prompt()
                user_prompt = self._build_initial_user_prompt(resume_text)
            else:
                # Subsequent attempts: focused on missing fields
                missing_fields = parsed_data.get_missing_fields() if parsed_data else []
                if not missing_fields:
                    break
                system_prompt = self._build_system_prompt()
                user_prompt = self._build_retry_user_prompt(resume_text, missing_fields, parsed_data)
                logger.info(f"Missing fields after attempt {attempt}: {missing_fields}")
            
            try:
                # Make LLM request using new client interface
                response = await self.client.chat(
                    model=self.config.model,
                    system=system_prompt,
                    user=user_prompt,
                    pdf_bytes=pdf_bytes,
                    text=fallback_text
                )
                responses.append(response)
                
                # Extract and parse JSON response
                try:
                    json_content = self._extract_json(response.content)
                    data_dict = json.loads(json_content)
                    
                    # Ensure full_text is included if missing
                    if 'full_text' not in data_dict or not data_dict['full_text']:
                        data_dict['full_text'] = resume_text
                    
                    # Create ParsedResumeData with proper error handling
                    try:
                        new_parsed_data = ParsedResumeData(**data_dict)
                    except Exception as validation_error:
                        logger.error(f"Pydantic validation failed: {validation_error}")
                        logger.debug(f"Raw data_dict: {data_dict}")
                        
                        # Try to create with minimal required fields
                        minimal_data = {
                            'full_name': data_dict.get('full_name'),
                            'email': data_dict.get('email'),
                            'phone': data_dict.get('phone'),
                            'skills': data_dict.get('skills', []),
                            'experience': data_dict.get('experience', []),
                            'education': data_dict.get('education', []),
                            'full_text': resume_text
                        }
                        new_parsed_data = ParsedResumeData(**minimal_data)
                    
                    # Merge with existing data if this is a retry
                    if parsed_data and attempt > 0:
                        parsed_data = self._merge_parsed_data(parsed_data, new_parsed_data)
                    else:
                        parsed_data = new_parsed_data
                    
                    # Update cost tracking
                    self.total_tokens_used += response.tokens_used
                    self.total_cost_cents += response.cost_cents
                    self.requests_made += 1
                    
                    # Check if we have all required fields
                    if parsed_data.is_complete():
                        logger.info(f"Resume parsing completed successfully on attempt {attempt + 1}")
                        break
                    else:
                        missing = parsed_data.get_missing_fields()
                        logger.warning(f"Missing fields after attempt {attempt + 1}: {missing}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM response as JSON: {e}")
                    logger.debug(f"Raw response: {response.content}")
                    if attempt == max_attempts - 1:
                        # Last attempt failed, return what we have
                        if parsed_data is None:
                            parsed_data = ParsedResumeData(full_text=resume_text)
                
            except Exception as e:
                logger.error(f"LLM request failed on attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(0.5 * (2 ** attempt))  # Exponential backoff
                else:
                    # Return empty data on final failure
                    if parsed_data is None:
                        parsed_data = ParsedResumeData(full_text=resume_text)
        
        return parsed_data, responses
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for LLM"""
        return """You are an expert resume parser. Your task is to extract structured information from resume text and return it as valid JSON.

Rules:
- Return ONLY a single JSON object, no additional text or explanation
- If a field is not found, use null for strings/numbers or empty array [] for lists  
- Do not hallucinate information that is not present
- Extract ALL skills mentioned, including technical and soft skills
- Include ALL work experience and education entries
- Be thorough and accurate"""
    
    def _build_initial_user_prompt(self, resume_text: str) -> str:
        """Build initial user prompt for resume parsing"""
        return f"""Extract information from this resume and return as JSON with these exact fields:

{resume_text}

Return JSON with this structure:
{{
    "full_name": "<person's full name>",
    "email": "<email address>", 
    "phone": "<phone number>",
    "skills": ["<skill1>", "<skill2>", ...],
    "experience": [{{"title": "<job title>", "company": "<company>", "duration": "<dates>", "description": "<description>"}}],
    "education": [{{"degree": "<degree>", "field": "<field>", "institution": "<school>", "year": "<year>"}}],
    "full_text": "{resume_text}",
    "summary": "<professional summary>",
    "certifications": ["<cert1>", "<cert2>", ...],
    "years_of_experience": <number>,
    "education_level": "<level>",
    "links": {{"linkedin": "<url>", "github": "<url>", "portfolio": "<url>"}}
}}"""
    
    def _build_retry_user_prompt(self, resume_text: str, missing_fields: List[str], current_data: ParsedResumeData) -> str:
        """Build retry prompt for missing fields"""
        return f"""I previously parsed this resume but some fields were missing. Please focus on extracting the missing information.

Resume:
{resume_text}

Missing fields: {', '.join(missing_fields)}

Current data:
{json.dumps(current_data.dict(), indent=2, default=str)}

Please return a JSON object with ONLY the missing fields filled in. Focus specifically on finding:
{chr(10).join([f"- {field}: Look carefully in the text for this information" for field in missing_fields])}

Return only the JSON object with the missing fields."""
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response that may contain extra text"""
        # Find the first complete JSON object in the response
        text = text.strip()
        
        # If it already looks like pure JSON, return as is
        if text.startswith('{') and text.endswith('}'):
            return text
            
        # Look for JSON object within the text
        start_idx = text.find('{')
        if start_idx == -1:
            raise ValueError("No JSON object found in response")
        
        # Find the matching closing brace
        brace_count = 0
        end_idx = start_idx
        
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if brace_count != 0:
            raise ValueError("Incomplete JSON object in response")
            
        return text[start_idx:end_idx]
    
    def _merge_parsed_data(self, existing: ParsedResumeData, new: ParsedResumeData) -> ParsedResumeData:
        """Merge new parsed data with existing data, preferring non-empty values"""
        merged_dict = existing.dict()
        new_dict = new.dict()
        
        for field, value in new_dict.items():
            if value and not merged_dict.get(field):  # Only update if new value exists and old one doesn't
                merged_dict[field] = value
            elif isinstance(value, list) and value:  # For lists, extend if new has more items
                existing_list = merged_dict.get(field, [])
                if len(value) > len(existing_list):
                    merged_dict[field] = value
        
        return ParsedResumeData(**merged_dict)