"""LLM service for resume parsing and structured data extraction

Provides LLM-powered resume parsing with support for multiple providers (OpenAI, Anthropic, etc.)
Includes retry logic for incomplete field extraction and proper error handling.
"""
from __future__ import annotations
import logging
import json
import hashlib
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import asyncio

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

@dataclass
class ParsedResumeData:
    """Structured resume data from LLM parsing"""
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    summary: Optional[str] = None
    skills: List[str] = None
    experience: List[Dict[str, Any]] = None
    education: List[Dict[str, Any]] = None
    certifications: List[str] = None
    years_of_experience: Optional[float] = None
    education_level: Optional[str] = None
    
    def __post_init__(self):
        if self.skills is None:
            self.skills = []
        if self.experience is None:
            self.experience = []
        if self.education is None:
            self.education = []
        if self.certifications is None:
            self.certifications = []
    
    def get_missing_fields(self) -> List[str]:
        """Get list of missing required fields"""
        missing = []
        
        if not self.full_name:
            missing.append("full_name")
        if not self.email:
            missing.append("email")
        if not self.skills:
            missing.append("skills")
        if not self.experience:
            missing.append("experience")
        if not self.education:
            missing.append("education")
        
        return missing
    
    def is_complete(self) -> bool:
        """Check if all required fields are present"""
        return len(self.get_missing_fields()) == 0

class MockLLMProvider:
    """Mock LLM provider for testing"""
    
    def __init__(self):
        self.request_count = 0
    
    async def generate_completion(self, request: LLMRequest) -> LLMResponse:
        """Generate mock LLM completion"""
        await asyncio.sleep(0.1)  # Simulate API delay
        
        # Extract basic info from the text for more realistic mock responses
        text = request.prompt.lower()
        
        # Mock structured response based on typical resume content
        mock_data = {
            "full_name": "John Doe",
            "email": "john.doe@email.com" if "@" in request.prompt else None,
            "phone": "555-123-4567" if any(c.isdigit() for c in request.prompt) else None,
            "linkedin": "linkedin.com/in/johndoe" if "linkedin" in text else None,
            "summary": "Experienced professional with strong technical skills",
            "skills": ["Python", "JavaScript", "React", "SQL"] if "python" in text or "javascript" in text else ["Communication", "Problem Solving"],
            "experience": [
                {
                    "title": "Software Engineer",
                    "company": "Tech Corp",
                    "duration": "2020-2024",
                    "description": "Developed software solutions"
                }
            ],
            "education": [
                {
                    "degree": "Bachelor of Science",
                    "field": "Computer Science",
                    "institution": "University",
                    "year": "2020"
                }
            ],
            "certifications": [],
            "years_of_experience": 4.0,
            "education_level": "bachelors"
        }
        
        # Randomly remove some fields to test retry logic
        import random
        if self.request_count % 3 == 0:  # Every 3rd request has missing data
            fields_to_remove = random.sample(["email", "phone", "skills"], 1)
            for field in fields_to_remove:
                mock_data[field] = None if field in ["email", "phone"] else []
        
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
                    new_parsed_data = ParsedResumeData(**data_dict)
                    
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
    "linkedin": "<LinkedIn profile URL or username>",
    "summary": "<professional summary or objective>",
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
    "certifications": ["<cert1>", "<cert2>", ...],
    "years_of_experience": <total years as a number>,
    "education_level": "<highest education level: high_school, associates, bachelors, masters, phd>"
}}

IMPORTANT:
- Return ONLY the JSON object, no additional text
- If a field is not found, use null for strings/numbers or empty array [] for lists
- Extract ALL skills mentioned, including technical and soft skills
- Include ALL work experience entries
- Calculate years_of_experience based on work history
- Be thorough and accurate
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
{json.dumps(asdict(current_data), indent=2, default=str)}

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
        merged_dict = asdict(existing)
        new_dict = asdict(new)
        
        for field, value in new_dict.items():
            if value and not merged_dict.get(field):  # Only update if new value exists and old one doesn't
                merged_dict[field] = value
            elif isinstance(value, list) and value:  # For lists, extend if new has more items
                existing_list = merged_dict.get(field, [])
                if len(value) > len(existing_list):
                    merged_dict[field] = value
        
        return ParsedResumeData(**merged_dict)

def create_llm_service(provider: LLMProvider = LLMProvider.MOCK, model: str = "gpt-3.5-turbo") -> LLMService:
    """Factory function to create a configured LLM service"""
    return LLMService(provider=provider, model=model)