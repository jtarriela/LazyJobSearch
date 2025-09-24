"""Resume review and iteration loop

Implements the LLM-powered resume critique and iterative improvement system.
Handles version lineage, diff generation, and acceptance workflow.
"""
from __future__ import annotations
import logging
import json
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class ReviewStatus(Enum):
    """Status of a resume review"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"

# Valid state transitions for review workflow
VALID_TRANSITIONS = {
    ReviewStatus.PENDING: [ReviewStatus.IN_PROGRESS, ReviewStatus.REJECTED],
    ReviewStatus.IN_PROGRESS: [ReviewStatus.COMPLETED, ReviewStatus.ACCEPTED, ReviewStatus.REJECTED],
    ReviewStatus.COMPLETED: [ReviewStatus.ACCEPTED, ReviewStatus.REJECTED],
    ReviewStatus.ACCEPTED: [],  # Terminal state
    ReviewStatus.REJECTED: []   # Terminal state
}

def validate_status_transition(current_status: str, new_status: str) -> bool:
    """Validate if a status transition is allowed.
    
    Args:
        current_status: Current status string
        new_status: Desired new status string
        
    Returns:
        True if transition is valid, False otherwise
    """
    try:
        current = ReviewStatus(current_status)
        new = ReviewStatus(new_status)
        return new in VALID_TRANSITIONS.get(current, [])
    except ValueError:
        # Invalid status values
        return False

def get_valid_next_states(current_status: str) -> List[str]:
    """Get list of valid next states for the current status.
    
    Args:
        current_status: Current status string
        
    Returns:
        List of valid next status strings
    """
    try:
        current = ReviewStatus(current_status)
        return [status.value for status in VALID_TRANSITIONS.get(current, [])]
    except ValueError:
        return []

@dataclass
class ReviewCritique:
    """LLM critique of a resume"""
    overall_score: int  # 0-100
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]
    specific_feedback: Dict[str, str]  # section -> feedback
    reasoning: str
    estimated_effort: str  # "low", "medium", "high"

@dataclass
class ResumeVersion:
    """A version of a resume with lineage tracking"""
    version_id: str
    resume_id: str
    content: str
    version_number: int
    parent_version_id: Optional[str] = None
    created_at: Optional[datetime] = None
    is_active: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class ReviewRequest:
    """Request for resume review against a job"""
    request_id: str
    resume_id: str
    job_id: str
    current_version_id: str
    status: ReviewStatus = ReviewStatus.PENDING
    created_at: Optional[datetime] = None
    max_iterations: int = 3
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass 
class ResumeRewrite:
    """A rewritten version of a resume based on critique"""
    rewrite_id: str
    original_version_id: str
    new_content: str
    changes_summary: str
    sections_changed: List[str]
    estimated_improvement: Dict[str, int]  # metric -> improvement score
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ResumeCritic:
    """LLM-powered resume critic"""
    
    def __init__(self):
        self.total_critiques = 0
        self.total_cost_cents = 0.0
    
    async def critique_resume(
        self,
        resume_content: str,
        job_description: str,
        job_title: str,
        company_name: str
    ) -> Tuple[ReviewCritique, float]:
        """Generate critique of resume against job requirements
        
        Args:
            resume_content: Full resume text
            job_description: Job description text
            job_title: Job title
            company_name: Company name
            
        Returns:
            (ReviewCritique, cost_in_cents)
        """
        try:
            prompt = self._build_critique_prompt(
                resume_content, job_description, job_title, company_name
            )
            
            critique_data, cost = await self._call_llm_for_critique(prompt)
            
            critique = ReviewCritique(
                overall_score=critique_data.get('overall_score', 0),
                strengths=critique_data.get('strengths', []),
                weaknesses=critique_data.get('weaknesses', []),
                improvement_suggestions=critique_data.get('improvement_suggestions', []),
                specific_feedback=critique_data.get('specific_feedback', {}),
                reasoning=critique_data.get('reasoning', ''),
                estimated_effort=critique_data.get('estimated_effort', 'medium')
            )
            
            self.total_critiques += 1
            self.total_cost_cents += cost
            
            return critique, cost
            
        except Exception as e:
            logger.error(f"Failed to critique resume: {e}")
            # Return a minimal critique
            return ReviewCritique(
                overall_score=50,
                strengths=[],
                weaknesses=["Unable to complete automatic review"],
                improvement_suggestions=["Manual review recommended"],
                specific_feedback={},
                reasoning="Automatic review failed",
                estimated_effort="unknown"
            ), 0.0
    
    def _build_critique_prompt(
        self,
        resume_content: str,
        job_description: str,
        job_title: str,
        company_name: str
    ) -> str:
        """Build LLM prompt for resume critique"""
        
        prompt = f"""
You are an expert resume reviewer and career coach. Analyze this resume against the specific job requirements and provide detailed, actionable feedback.

JOB DETAILS:
Title: {job_title}
Company: {company_name}
Description: {job_description[:1500]}...

RESUME TO REVIEW:
{resume_content[:2000]}...

ANALYSIS REQUIREMENTS:
1. Score the overall fit (0-100) considering:
   - Skills alignment with job requirements
   - Experience level match
   - Industry/domain relevance
   - Resume quality and presentation

2. Identify key strengths that align with this role
3. Identify critical weaknesses or gaps
4. Provide specific, actionable improvement suggestions
5. Give section-specific feedback (Summary, Experience, Skills, etc.)
6. Estimate effort needed for improvements (low/medium/high)

Respond with JSON in this exact format:
{{
    "overall_score": <integer 0-100>,
    "strengths": ["<strength1>", "<strength2>", "<strength3>"],
    "weaknesses": ["<weakness1>", "<weakness2>", "<weakness3>"],
    "improvement_suggestions": [
        "<actionable suggestion 1>",
        "<actionable suggestion 2>",
        "<actionable suggestion 3>"
    ],
    "specific_feedback": {{
        "summary": "<feedback on summary/objective section>",
        "experience": "<feedback on experience section>",
        "skills": "<feedback on skills section>",
        "education": "<feedback on education section>",
        "overall_presentation": "<feedback on formatting/structure>"
    }},
    "reasoning": "<2-3 sentence explanation of the score and key factors>",
    "estimated_effort": "<low|medium|high>"
}}
"""
        
        return prompt
    
    async def _call_llm_for_critique(self, prompt: str) -> Tuple[Dict[str, Any], float]:
        """Call LLM service for critique (mock implementation)"""
        import asyncio
        
        # Simulate LLM processing time
        await asyncio.sleep(0.1)
        
        # Mock response based on prompt content
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        score = int(prompt_hash[:2], 16) % 40 + 60  # Score 60-99
        
        mock_response = {
            "overall_score": score,
            "strengths": [
                "Relevant technical skills mentioned",
                "Clear work experience progression",
                "Good educational background"
            ],
            "weaknesses": [
                "Missing specific achievements with metrics",
                "Could better highlight leadership experience", 
                "Skills section could be more targeted"
            ],
            "improvement_suggestions": [
                "Add quantified achievements (e.g., 'Improved system performance by 30%')",
                "Tailor skills section to match job requirements more closely",
                "Include relevant certifications or training"
            ],
            "specific_feedback": {
                "summary": "Consider making the summary more specific to this role",
                "experience": "Add more metrics and impact statements to experience",
                "skills": "Reorganize skills to highlight most relevant ones first",
                "education": "Education section is adequate",
                "overall_presentation": "Good structure, consider minor formatting improvements"
            },
            "reasoning": f"Score of {score} reflects good alignment with some room for targeted improvements.",
            "estimated_effort": "medium"
        }
        
        cost_cents = 2.5  # Mock cost ~$0.025 per critique
        
        return mock_response, cost_cents

class ResumeRewriter:
    """LLM-powered resume rewriter"""
    
    def __init__(self):
        self.total_rewrites = 0
        self.total_cost_cents = 0.0
    
    async def rewrite_resume(
        self,
        original_content: str,
        critique: ReviewCritique,
        job_description: str,
        focus_areas: Optional[List[str]] = None
    ) -> Tuple[ResumeRewrite, float]:
        """Rewrite resume based on critique
        
        Args:
            original_content: Original resume text
            critique: Critique to address
            job_description: Target job description
            focus_areas: Specific areas to focus improvements on
            
        Returns:
            (ResumeRewrite, cost_in_cents)
        """
        try:
            prompt = self._build_rewrite_prompt(
                original_content, critique, job_description, focus_areas
            )
            
            rewrite_data, cost = await self._call_llm_for_rewrite(prompt)
            
            rewrite = ResumeRewrite(
                rewrite_id=self._generate_rewrite_id(),
                original_version_id="",  # Will be set by caller
                new_content=rewrite_data.get('new_content', original_content),
                changes_summary=rewrite_data.get('changes_summary', ''),
                sections_changed=rewrite_data.get('sections_changed', []),
                estimated_improvement=rewrite_data.get('estimated_improvement', {}),
                metadata={
                    'critique_score': critique.overall_score,
                    'focus_areas': focus_areas or [],
                    'rewrite_timestamp': datetime.now().isoformat()
                }
            )
            
            self.total_rewrites += 1
            self.total_cost_cents += cost
            
            return rewrite, cost
            
        except Exception as e:
            logger.error(f"Failed to rewrite resume: {e}")
            raise
    
    def _build_rewrite_prompt(
        self,
        original_content: str,
        critique: ReviewCritique,
        job_description: str,
        focus_areas: Optional[List[str]]
    ) -> str:
        """Build LLM prompt for resume rewriting"""
        
        focus_text = ""
        if focus_areas:
            focus_text = f"\nFOCUS AREAS: Pay special attention to: {', '.join(focus_areas)}"
        
        prompt = f"""
You are an expert resume writer. Rewrite this resume to address the identified weaknesses and better align with the job requirements.

JOB REQUIREMENTS:
{job_description[:1000]}...

ORIGINAL RESUME:
{original_content}

CRITIQUE TO ADDRESS:
Overall Score: {critique.overall_score}/100
Weaknesses: {', '.join(critique.weaknesses)}
Improvement Suggestions: {', '.join(critique.improvement_suggestions)}
{focus_text}

REWRITING INSTRUCTIONS:
1. Keep all factual information accurate - do not invent experience or skills
2. Improve presentation, formatting, and word choice
3. Add specific achievements and metrics where possible (use placeholders like [X%] if exact numbers aren't known)
4. Reorganize sections for better impact
5. Tailor language to match job requirements
6. Maintain professional tone throughout

Respond with JSON in this exact format:
{{
    "new_content": "<complete rewritten resume text>",
    "changes_summary": "<brief description of major changes made>",
    "sections_changed": ["<section1>", "<section2>"],
    "estimated_improvement": {{
        "skills_alignment": <0-100 improvement estimate>,
        "presentation": <0-100 improvement estimate>,
        "impact_statements": <0-100 improvement estimate>
    }}
}}
"""
        
        return prompt
    
    async def _call_llm_for_rewrite(self, prompt: str) -> Tuple[Dict[str, Any], float]:
        """Call LLM service for rewrite (mock implementation)"""
        import asyncio
        
        # Simulate LLM processing time
        await asyncio.sleep(0.2)
        
        # Extract original content from prompt for mock rewrite
        lines = prompt.split('\n')
        original_start = -1
        for i, line in enumerate(lines):
            if "ORIGINAL RESUME:" in line:
                original_start = i + 1
                break
        
        original_content = "Enhanced resume content based on critique feedback..."
        if original_start > 0:
            original_lines = []
            for i in range(original_start, len(lines)):
                if lines[i].startswith("CRITIQUE TO ADDRESS:"):
                    break
                original_lines.append(lines[i])
            if original_lines:
                original_content = '\n'.join(original_lines).strip()
        
        # Mock rewrite - in reality, this would be a sophisticated LLM rewrite
        mock_response = {
            "new_content": f"[IMPROVED VERSION]\n{original_content}\n\n[Additional improvements applied based on critique]",
            "changes_summary": "Enhanced skills presentation, added quantified achievements, improved summary section",
            "sections_changed": ["summary", "experience", "skills"],
            "estimated_improvement": {
                "skills_alignment": 25,
                "presentation": 30,
                "impact_statements": 40
            }
        }
        
        cost_cents = 5.0  # Mock cost ~$0.05 per rewrite
        
        return mock_response, cost_cents
    
    def _generate_rewrite_id(self) -> str:
        """Generate unique rewrite ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"rewrite_{timestamp}_{self.total_rewrites:04d}"

class ReviewIterationManager:
    """Manages the complete review and iteration process"""
    
    def __init__(self, session):
        self.session = session
        self.critic = ResumeCritic()
        self.rewriter = ResumeRewriter()
    
    async def start_review_process(
        self,
        resume_id: str,
        job_id: str,
        max_iterations: int = 3
    ) -> ReviewRequest:
        """Start a new review process for a resume-job pair
        
        Args:
            resume_id: Resume to review
            job_id: Job to review against
            max_iterations: Maximum number of iteration cycles
            
        Returns:
            ReviewRequest object
        """
        # Get current resume version
        current_version = self._get_current_resume_version(resume_id)
        if not current_version:
            raise ValueError(f"Resume {resume_id} not found")
        
        # Get job details
        job_details = self._get_job_details(job_id)
        if not job_details:
            raise ValueError(f"Job {job_id} not found")
        
        # Create review request
        request = ReviewRequest(
            request_id=self._generate_request_id(),
            resume_id=resume_id,
            job_id=job_id,
            current_version_id=current_version.version_id,
            max_iterations=max_iterations,
            metadata={
                'job_title': job_details.get('title', ''),
                'company_name': job_details.get('company', ''),
                'started_at': datetime.now().isoformat()
            }
        )
        
        # Persist request
        self._save_review_request(request)
        
        return request
    
    async def perform_review_iteration(
        self,
        request: ReviewRequest,
        accept_threshold: int = 80
    ) -> Tuple[ReviewCritique, Optional[ResumeRewrite], bool]:
        """Perform one iteration of the review process
        
        Args:
            request: Review request to process
            accept_threshold: Score threshold for auto-acceptance
            
        Returns:
            (critique, rewrite_if_needed, should_continue)
        """
        # Get current resume version
        current_version = self._get_resume_version(request.current_version_id)
        job_details = self._get_job_details(request.job_id)
        
        # Generate critique
        critique, critique_cost = await self.critic.critique_resume(
            current_version.content,
            job_details.get('description', ''),
            job_details.get('title', ''),
            job_details.get('company', '')
        )
        
        # Check if score meets acceptance threshold
        if critique.overall_score >= accept_threshold:
            request.status = ReviewStatus.ACCEPTED
            self._update_review_request(request)
            return critique, None, False
        
        # Check iteration limit
        current_iteration = request.metadata.get('current_iteration', 0)
        if current_iteration >= request.max_iterations:
            request.status = ReviewStatus.COMPLETED
            self._update_review_request(request)
            return critique, None, False
        
        # Generate rewrite
        rewrite, rewrite_cost = await self.rewriter.rewrite_resume(
            current_version.content,
            critique,
            job_details.get('description', '')
        )
        
        # Create new resume version
        new_version = ResumeVersion(
            version_id=self._generate_version_id(),
            resume_id=request.resume_id,
            content=rewrite.new_content,
            version_number=current_version.version_number + 1,
            parent_version_id=current_version.version_id,
            metadata={
                'review_request_id': request.request_id,
                'critique_score': critique.overall_score,
                'changes_summary': rewrite.changes_summary,
                'cost_cents': critique_cost + rewrite_cost
            }
        )
        
        # Save new version and update request
        self._save_resume_version(new_version)
        request.current_version_id = new_version.version_id
        request.metadata['current_iteration'] = current_iteration + 1
        request.metadata['total_cost_cents'] = request.metadata.get('total_cost_cents', 0.0) + critique_cost + rewrite_cost
        
        self._update_review_request(request)
        
        return critique, rewrite, True
    
    def get_version_diff(self, version1_id: str, version2_id: str) -> Dict[str, Any]:
        """Generate diff between two resume versions"""
        try:
            version1 = self._get_resume_version(version1_id)
            version2 = self._get_resume_version(version2_id)
            
            if not version1 or not version2:
                return {'error': 'Version not found'}
            
            # Simple diff implementation (could be enhanced with proper diff library)
            lines1 = version1.content.split('\n')
            lines2 = version2.content.split('\n')
            
            changes = {
                'added_lines': [],
                'removed_lines': [],
                'modified_lines': [],
                'stats': {
                    'lines_added': 0,
                    'lines_removed': 0,
                    'lines_modified': 0
                }
            }
            
            # Basic line-by-line comparison
            max_lines = max(len(lines1), len(lines2))
            for i in range(max_lines):
                line1 = lines1[i] if i < len(lines1) else None
                line2 = lines2[i] if i < len(lines2) else None
                
                if line1 is None:
                    changes['added_lines'].append({'line_num': i+1, 'content': line2})
                    changes['stats']['lines_added'] += 1
                elif line2 is None:
                    changes['removed_lines'].append({'line_num': i+1, 'content': line1})
                    changes['stats']['lines_removed'] += 1
                elif line1 != line2:
                    changes['modified_lines'].append({
                        'line_num': i+1,
                        'old_content': line1,
                        'new_content': line2
                    })
                    changes['stats']['lines_modified'] += 1
            
            return changes
            
        except Exception as e:
            logger.error(f"Failed to generate diff: {e}")
            return {'error': str(e)}
    
    def _get_current_resume_version(self, resume_id: str) -> Optional[ResumeVersion]:
        """Get the current active version of a resume"""
        # Mock implementation - would query database
        return ResumeVersion(
            version_id=f"v_{resume_id}_001",
            resume_id=resume_id,
            content="Sample resume content...",
            version_number=1,
            is_active=True
        )
    
    def _get_resume_version(self, version_id: str) -> Optional[ResumeVersion]:
        """Get a specific resume version"""
        # Mock implementation
        return ResumeVersion(
            version_id=version_id,
            resume_id="sample_resume",
            content="Sample resume content for version...",
            version_number=1
        )
    
    def _get_job_details(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job details"""
        # Mock implementation - would query database
        return {
            'title': 'Software Engineer',
            'company': 'TechCorp',
            'description': 'Looking for a Python developer with 3+ years experience...'
        }
    
    def _save_review_request(self, request: ReviewRequest) -> None:
        """Save review request to database"""
        # Mock implementation - would save to DB
        logger.info(f"Saved review request {request.request_id}")
    
    def _update_review_request(self, request: ReviewRequest) -> None:
        """Update review request in database"""
        # Mock implementation - would update DB
        logger.info(f"Updated review request {request.request_id}")
    
    def _save_resume_version(self, version: ResumeVersion) -> None:
        """Save resume version to database"""
        # Mock implementation - would save to DB
        logger.info(f"Saved resume version {version.version_id}")
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"req_{timestamp}"
    
    def _generate_version_id(self) -> str:
        """Generate unique version ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"ver_{timestamp}"

def create_review_iteration_manager(session) -> ReviewIterationManager:
    """Factory function to create review iteration manager"""
    return ReviewIterationManager(session)