"""Prompt templates for job matching and scoring."""

from typing import List, Optional


class JobMatchPrompt:
    """Prompt templates for LLM-based job matching."""
    
    SCORING_SYSTEM_PROMPT = """You are an expert technical recruiter and career advisor. Your job is to evaluate how well a candidate's resume matches a specific job posting.

You should:
1. Analyze technical skills alignment
2. Assess experience level fit
3. Identify skill gaps and strengths
4. Consider transferable skills
5. Provide actionable feedback

Always respond with a JSON object containing:
- "score": integer from 0-100 (overall match quality)
- "action": one of "apply", "skip", "maybe"
- "reasoning": detailed explanation of your assessment
- "skill_gaps": object with missing or weak areas
- "strengths": object with matching qualifications
- "confidence": float from 0.0-1.0 indicating your confidence in this assessment

Be honest about gaps but also recognize transferable skills and growth potential."""

    SCORING_USER_PROMPT = """Please evaluate this candidate-job match:

**Job Title:** {job_title}
**Seniority Level:** {seniority}

**Job Requirements (most relevant sections):**
{job_chunks}

**Candidate Background (most relevant sections):**  
{resume_chunks}

**Additional Context:**
- Candidate has {yoe} years of total experience
- Key skills from resume: {resume_skills}
- Key requirements from job: {job_skills}

Provide your assessment as a JSON object."""

    COVER_LETTER_SYSTEM_PROMPT = """You are a professional career writer specializing in personalized cover letters. 

Create cover letters that:
1. Are concise and compelling (2-3 paragraphs)
2. Highlight specific relevant experience
3. Show genuine interest in the company/role
4. Use a professional but personable tone
5. Include specific examples and achievements
6. Avoid generic language and clichés

The cover letter should complement the resume, not repeat it."""

    COVER_LETTER_USER_PROMPT = """Write a cover letter for this application:

**Company:** {company_name}
**Position:** {job_title}

**Job Description:**
{job_description}

**Candidate Resume:**
{resume_text}

Create a personalized cover letter that highlights the most relevant qualifications and shows enthusiasm for this specific role."""

    @classmethod
    def format_scoring_prompt(
        cls,
        job_title: str,
        job_chunks: List[str],
        resume_chunks: List[str],
        seniority: Optional[str] = None,
        yoe: Optional[float] = None,
        resume_skills: Optional[List[str]] = None,
        job_skills: Optional[List[str]] = None
    ) -> str:
        """Format the job scoring prompt with specific data."""
        
        # Join chunks with clear separators
        job_text = "\n\n---\n\n".join(job_chunks)
        resume_text = "\n\n---\n\n".join(resume_chunks)
        
        # Format optional fields
        seniority_str = seniority or "Not specified"
        yoe_str = f"{yoe:.1f}" if yoe else "Not specified"
        resume_skills_str = ", ".join(resume_skills[:10]) if resume_skills else "Not specified"
        job_skills_str = ", ".join(job_skills[:10]) if job_skills else "Not specified"
        
        return cls.SCORING_USER_PROMPT.format(
            job_title=job_title,
            seniority=seniority_str,
            job_chunks=job_text,
            resume_chunks=resume_text,
            yoe=yoe_str,
            resume_skills=resume_skills_str,
            job_skills=job_skills_str
        )
    
    @classmethod
    def format_cover_letter_prompt(
        cls,
        company_name: str,
        job_title: str,
        job_description: str,
        resume_text: str
    ) -> str:
        """Format the cover letter generation prompt."""
        
        return cls.COVER_LETTER_USER_PROMPT.format(
            company_name=company_name,
            job_title=job_title,
            job_description=job_description[:2000],  # Truncate for token limits
            resume_text=resume_text[:3000]  # Truncate for token limits
        )
    
    @classmethod
    def get_action_from_score(cls, score: int) -> str:
        """Convert numeric score to action recommendation."""
        if score >= 75:
            return "apply"
        elif score >= 50:
            return "maybe"
        else:
            return "skip"