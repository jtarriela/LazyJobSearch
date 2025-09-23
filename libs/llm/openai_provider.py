"""OpenAI LLM provider for job matching and scoring."""

import json
import os
from typing import List, Optional
from openai import OpenAI

from .provider_base import LLMProvider, LLMResponse
from .scoring_prompts import JobMatchPrompt


class OpenAILLMProvider(LLMProvider):
    """OpenAI LLM provider using GPT-4 for job matching.
    
    Uses structured prompts to evaluate job-resume fit and generate
    actionable recommendations with reasoning.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """Initialize OpenAI LLM provider.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: OpenAI model to use (gpt-4, gpt-4o-mini, etc.)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter.")
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        
        # Model pricing (per 1K tokens, as of implementation)
        self._pricing = {
            "gpt-4": {"input": 0.030, "output": 0.060},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        }
    
    def score_job_match(
        self,
        resume_chunks: List[str],
        job_chunks: List[str],
        job_title: str,
        job_seniority: Optional[str] = None,
        yoe: Optional[float] = None,
        resume_skills: Optional[List[str]] = None,
        job_skills: Optional[List[str]] = None
    ) -> LLMResponse:
        """Score how well a resume matches a job posting."""
        
        # Format the prompt
        user_prompt = JobMatchPrompt.format_scoring_prompt(
            job_title=job_title,
            job_chunks=job_chunks,
            resume_chunks=resume_chunks,
            seniority=job_seniority,
            yoe=yoe,
            resume_skills=resume_skills,
            job_skills=job_skills
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JobMatchPrompt.SCORING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent scoring
                max_tokens=1000,
                response_format={"type": "json_object"}  # Ensure JSON response
            )
            
            # Parse the JSON response
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Extract and validate fields
            score = max(0, min(100, int(result.get("score", 0))))
            action = result.get("action", JobMatchPrompt.get_action_from_score(score))
            reasoning = result.get("reasoning", "No reasoning provided")
            skill_gaps = result.get("skill_gaps", {})
            confidence = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
            
            return LLMResponse(
                score=score,
                action=action,
                reasoning=reasoning,
                skill_gaps=skill_gaps,
                confidence=confidence,
                model_used=self.model,
                tokens_used=response.usage.total_tokens
            )
            
        except json.JSONDecodeError as e:
            # Fallback if JSON parsing fails
            return LLMResponse(
                score=0,
                action="skip",
                reasoning=f"Error parsing LLM response: {str(e)}",
                skill_gaps={},
                confidence=0.0,
                model_used=self.model,
                tokens_used=0
            )
        except Exception as e:
            # General error handling
            return LLMResponse(
                score=0,
                action="skip", 
                reasoning=f"Error calling LLM: {str(e)}",
                skill_gaps={},
                confidence=0.0,
                model_used=self.model,
                tokens_used=0
            )
    
    def generate_cover_letter(
        self,
        resume_text: str,
        job_description: str,
        company_name: str,
        job_title: str
    ) -> str:
        """Generate a personalized cover letter."""
        
        user_prompt = JobMatchPrompt.format_cover_letter_prompt(
            company_name=company_name,
            job_title=job_title,
            job_description=job_description,
            resume_text=resume_text
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JobMatchPrompt.COVER_LETTER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Slightly higher temperature for creativity
                max_tokens=800
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating cover letter: {str(e)}"
    
    def get_model_name(self) -> str:
        """Return the model identifier."""
        return self.model
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request."""
        pricing = self._pricing.get(self.model, self._pricing["gpt-4o-mini"])
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    def count_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters for English)."""
        return max(1, len(text) // 4)