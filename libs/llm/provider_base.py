"""Base LLM provider interface for job matching."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Structured response from LLM scoring."""
    score: int  # 0-100
    action: str  # "apply", "skip", "maybe"
    reasoning: str
    skill_gaps: Dict[str, Any]
    confidence: float
    model_used: str
    tokens_used: int


class LLMProvider(ABC):
    """Abstract base class for LLM providers used in job matching."""
    
    @abstractmethod
    def score_job_match(
        self,
        resume_chunks: list[str],
        job_chunks: list[str],
        job_title: str,
        job_seniority: Optional[str] = None
    ) -> LLMResponse:
        """Score how well a resume matches a job posting.
        
        Args:
            resume_chunks: Most relevant resume text chunks
            job_chunks: Most relevant job description chunks  
            job_title: Job title for context
            job_seniority: Seniority level if available
            
        Returns:
            LLMResponse with score, reasoning, and recommendations
        """
        pass
    
    @abstractmethod
    def generate_cover_letter(
        self,
        resume_text: str,
        job_description: str,
        company_name: str,
        job_title: str
    ) -> str:
        """Generate a personalized cover letter.
        
        Args:
            resume_text: Full resume text
            job_description: Full job description
            company_name: Company name
            job_title: Job title
            
        Returns:
            Generated cover letter text
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier."""
        pass
    
    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request."""
        pass