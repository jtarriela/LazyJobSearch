"""Basic matching pipeline for LazyJobSearch

This implements the core matching logic identified as missing in the problem statement.
Provides skill-based and experience-based matching as a foundation for the full
FTS + Vector + LLM pipeline described in ADR 0003.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum

from libs.db.models import Job, Resume, Match
from libs.observability import get_logger, timer, counter

logger = get_logger(__name__)

class MatchingStrategy(Enum):
    """Different matching strategies available"""
    SKILL_BASED = "skill_based"
    EXPERIENCE_BASED = "experience" 
    COMPOSITE = "composite"
    # Future: FTS = "fts", VECTOR = "vector", LLM = "llm"

@dataclass
class MatchResult:
    """Result of matching a resume against a job"""
    job_id: str
    resume_id: str
    skill_score: float
    experience_score: float
    composite_score: float
    reasoning: str
    matched_skills: List[str]
    
@dataclass
class MatchingConfig:
    """Configuration for the matching pipeline"""
    strategy: MatchingStrategy = MatchingStrategy.COMPOSITE
    skill_weight: float = 0.7
    experience_weight: float = 0.3
    min_score_threshold: float = 0.4
    max_results: int = 50

class BasicMatchingPipeline:
    """Basic implementation of the matching pipeline
    
    This addresses the critical gap where no matching pipeline existed.
    Implements simple skill and experience matching as foundation for
    the full FTS + Vector + LLM pipeline described in ADR 0003.
    """
    
    def __init__(self, db_session, config: Optional[MatchingConfig] = None):
        self.db_session = db_session
        self.config = config or MatchingConfig()
        
    def match_resume_to_jobs(self, resume_id: str, limit: Optional[int] = None) -> List[MatchResult]:
        """Match a resume against all available jobs
        
        Args:
            resume_id: ID of resume to match
            limit: Maximum number of results to return
            
        Returns:
            List of match results sorted by score (highest first)
        """
        with timer("matching.resume_to_jobs"):
            try:
                # Get resume from database
                resume = self.db_session.query(Resume).filter(Resume.id == resume_id).first()
                if not resume:
                    raise ValueError(f"Resume not found: {resume_id}")
                    
                # Get all jobs from database
                jobs = self.db_session.query(Job).all()
                
                if not jobs:
                    logger.warning("No jobs found in database for matching")
                    return []
                
                logger.info("Starting matching process", extra={
                    "resume_id": resume_id,
                    "num_jobs": len(jobs),
                    "strategy": self.config.strategy.value
                })
                
                # Extract resume skills and experience
                resume_skills = self._extract_skills_from_resume(resume)
                resume_experience = resume.yoe_raw or 0
                
                matches = []
                
                # Match against each job
                for job in jobs:
                    match_result = self._match_resume_to_job(
                        resume_id=resume_id,
                        resume_skills=resume_skills,
                        resume_experience=resume_experience,
                        job=job
                    )
                    
                    # Only include matches above threshold
                    if match_result.composite_score >= self.config.min_score_threshold:
                        matches.append(match_result)
                
                # Sort by score (highest first)
                matches.sort(key=lambda m: m.composite_score, reverse=True)
                
                # Apply limit
                result_limit = limit or self.config.max_results
                matches = matches[:result_limit]
                
                counter("matching.matches_found", value=len(matches))
                logger.info("Matching completed", extra={
                    "matches_found": len(matches),
                    "top_score": matches[0].composite_score if matches else 0
                })
                
                return matches
                
            except Exception as e:
                counter("matching.error")
                logger.error("Matching pipeline failed", extra={"error": str(e)})
                raise
    
    def _extract_skills_from_resume(self, resume: Resume) -> List[str]:
        """Extract skills list from resume"""
        if not resume.skills_csv:
            return []
        
        # Parse CSV skills and normalize
        skills = [skill.strip().lower() for skill in resume.skills_csv.split(',') if skill.strip()]
        return skills
    
    def _match_resume_to_job(
        self, 
        resume_id: str,
        resume_skills: List[str], 
        resume_experience: float,
        job: Job
    ) -> MatchResult:
        """Match a single resume against a single job"""
        
        # Extract job skills
        job_skills = self._extract_skills_from_job(job)
        
        # Calculate skill match score
        skill_score = self._calculate_skill_score(resume_skills, job_skills)
        
        # Calculate experience match score  
        experience_score = self._calculate_experience_score(resume_experience, job)
        
        # Calculate composite score
        composite_score = (
            skill_score * self.config.skill_weight +
            experience_score * self.config.experience_weight
        )
        
        # Find matched skills for reasoning
        matched_skills = list(set(resume_skills) & set(job_skills))
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            matched_skills=matched_skills,
            skill_score=skill_score,
            experience_score=experience_score,
            job=job
        )
        
        return MatchResult(
            job_id=job.id,
            resume_id=resume_id,
            skill_score=skill_score,
            experience_score=experience_score,
            composite_score=composite_score,
            reasoning=reasoning,
            matched_skills=matched_skills
        )
    
    def _extract_skills_from_job(self, job: Job) -> List[str]:
        """Extract skills list from job description"""
        if not job.jd_skills_csv:
            return []
            
        # Parse CSV skills and normalize
        skills = [skill.strip().lower() for skill in job.jd_skills_csv.split(',') if skill.strip()]
        return skills
    
    def _calculate_skill_score(self, resume_skills: List[str], job_skills: List[str]) -> float:
        """Calculate skill match score based on overlap"""
        if not job_skills:
            return 0.0
            
        # Calculate overlap ratio
        skill_overlap = set(resume_skills) & set(job_skills)
        score = len(skill_overlap) / len(job_skills)
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _calculate_experience_score(self, resume_experience: float, job: Job) -> float:
        """Calculate experience level match score"""
        
        # Simple experience level matching based on seniority
        if not job.seniority:
            return 0.5  # Neutral score if no seniority specified
            
        seniority_lower = job.seniority.lower()
        
        # Define experience thresholds
        if "senior" in seniority_lower or "lead" in seniority_lower:
            required_experience = 5.0
        elif "mid" in seniority_lower or "intermediate" in seniority_lower:
            required_experience = 3.0
        elif "junior" in seniority_lower or "entry" in seniority_lower:
            required_experience = 1.0
        else:
            return 0.5  # Neutral score for unknown seniority
        
        # Score based on experience match
        if resume_experience >= required_experience:
            # Bonus for having more experience than required (but cap the bonus)
            bonus = min((resume_experience - required_experience) / required_experience * 0.2, 0.3)
            return min(1.0 + bonus, 1.0)
        else:
            # Penalty for having less experience than required
            penalty_ratio = resume_experience / required_experience
            return max(penalty_ratio * 0.7, 0.1)  # Minimum score of 0.1
    
    def _generate_reasoning(
        self, 
        matched_skills: List[str],
        skill_score: float,
        experience_score: float,
        job: Job
    ) -> str:
        """Generate human-readable reasoning for the match"""
        
        reasoning_parts = []
        
        # Skill reasoning
        if matched_skills:
            reasoning_parts.append(f"Skills match: {', '.join(matched_skills)} ({skill_score:.2f})")
        else:
            reasoning_parts.append(f"Limited skill overlap ({skill_score:.2f})")
        
        # Experience reasoning
        exp_desc = "good" if experience_score > 0.8 else "adequate" if experience_score > 0.5 else "limited"
        reasoning_parts.append(f"Experience level: {exp_desc} match ({experience_score:.2f})")
        
        # Job context
        if job.location:
            reasoning_parts.append(f"Location: {job.location}")
            
        return "; ".join(reasoning_parts)

def create_matching_pipeline(db_session, config: Optional[MatchingConfig] = None) -> BasicMatchingPipeline:
    """Factory function to create matching pipeline"""
    return BasicMatchingPipeline(db_session, config)