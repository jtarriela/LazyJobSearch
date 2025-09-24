"""Core matching pipeline implementation

Implements the hybrid matching pipeline: FTS → Vector → LLM scoring
as specified in ADR 0003. Handles candidate pruning, scoring orchestration,
and match persistence with cost tracking.
"""
from __future__ import annotations
import logging
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class MatchingStage(Enum):
    """Stages in the matching pipeline"""
    FTS_PREFILTER = "fts"
    VECTOR_SIMILARITY = "vector"
    LLM_SCORING = "llm"
    PERSISTENCE = "persistence"

@dataclass
class JobCandidate:
    """A job candidate in the matching pipeline"""
    job_id: str
    title: str
    company: str
    description: str
    skills: List[str]
    seniority: Optional[str] = None
    location: Optional[str] = None
    url: Optional[str] = None
    fts_score: Optional[float] = None
    vector_score: Optional[float] = None
    llm_score: Optional[int] = None
    llm_reasoning: Optional[str] = None
    final_score: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ResumeProfile:
    """Resume profile for matching"""
    resume_id: str
    fulltext: str
    skills: List[str]
    years_experience: Optional[float] = None
    education_level: Optional[str] = None
    preferred_roles: List[str] = None
    embedding: Optional[List[float]] = None

@dataclass
class MatchingConfig:
    """Configuration for the matching pipeline"""
    fts_limit: int = 1000  # Max candidates from FTS stage
    vector_limit: int = 100  # Max candidates from vector stage
    llm_limit: int = 20  # Max candidates for LLM scoring
    
    fts_min_score: float = 0.1  # Minimum FTS score threshold
    vector_min_score: float = 0.5  # Minimum vector similarity threshold
    llm_min_score: int = 60  # Minimum LLM score threshold
    
    enable_cost_limits: bool = True
    max_cost_cents: float = 50.0  # Max cost per matching run
    
    # Scoring weights for final ranking
    fts_weight: float = 0.2
    vector_weight: float = 0.3
    llm_weight: float = 0.5

@dataclass
class MatchingResult:
    """Result of matching pipeline"""
    resume_id: str
    matches: List[JobCandidate]
    total_candidates: int
    stages_completed: List[MatchingStage]
    total_cost_cents: float
    processing_time_seconds: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class FTSSearcher:
    """Full-text search implementation for job candidates"""
    
    def __init__(self, session):
        self.session = session
    
    async def search_jobs(
        self, 
        query_text: str, 
        limit: int = 1000,
        min_score: float = 0.1
    ) -> List[JobCandidate]:
        """Search jobs using PostgreSQL full-text search
        
        Args:
            query_text: Search query text (typically resume skills/experience)
            limit: Maximum number of results
            min_score: Minimum FTS score threshold
            
        Returns:
            List of job candidates with FTS scores
        """
        try:
            # Build FTS query from resume text
            fts_query = self._build_fts_query(query_text)
            
            # SQL query using tsvector for full-text search with proper PostgreSQL syntax
            from sqlalchemy import text
            sql = text("""
            SELECT 
                j.id,
                j.title,
                c.name as company,
                j.jd_fulltext as description,
                j.jd_skills_csv as skills,
                j.seniority,
                j.location,
                j.url,
                ts_rank(j.jd_tsv, plainto_tsquery(:fts_query)) as fts_score
            FROM jobs j
            JOIN companies c ON j.company_id = c.id
            WHERE j.jd_tsv @@ plainto_tsquery(:fts_query)
            AND ts_rank(j.jd_tsv, plainto_tsquery(:fts_query)) >= :min_score
            ORDER BY fts_score DESC
            LIMIT :limit
            """)
            
            result = self.session.execute(
                sql, 
                {
                    "fts_query": fts_query,
                    "min_score": min_score,
                    "limit": limit
                }
            ).fetchall()
            
            candidates = []
            for row in result:
                skills = row.skills.split(',') if row.skills else []
                
                candidate = JobCandidate(
                    job_id=str(row.id),
                    title=row.title or "",
                    company=row.company or "",
                    description=row.description or "",
                    skills=skills,
                    seniority=row.seniority,
                    location=row.location,
                    url=row.url,
                    fts_score=float(row.fts_score),
                    metadata={'stage': 'fts'}
                )
                candidates.append(candidate)
            
            logger.info(f"FTS search found {len(candidates)} candidates")
            return candidates
            
        except Exception as e:
            logger.error(f"FTS search failed: {e}")
            return []
    
    def _build_fts_query(self, text: str) -> str:
        """Build FTS query from resume text"""
        # Extract important terms (skills, technologies, etc.)
        words = text.lower().split()
        
        # Filter out common words and keep technical terms
        important_words = []
        for word in words:
            if (len(word) > 2 and 
                word not in ['the', 'and', 'or', 'but', 'with', 'for', 'of', 'to', 'in', 'on', 'at']):
                important_words.append(word)
        
        # Take top words by frequency or use all if reasonable count
        if len(important_words) > 20:
            important_words = important_words[:20]
        
        return ' '.join(important_words)

class VectorSearcher:
    """Vector similarity search for job matching"""
    
    def __init__(self, session, embedding_service):
        self.session = session
        self.embedding_service = embedding_service
    
    async def search_similar_jobs(
        self,
        candidates: List[JobCandidate],
        resume_embedding: List[float],
        limit: int = 100,
        min_score: float = 0.5
    ) -> List[JobCandidate]:
        """Search for similar jobs using pgvector cosine distance
        
        Args:
            candidates: Job candidates from FTS stage
            resume_embedding: Resume embedding vector
            limit: Maximum number of results
            min_score: Minimum similarity threshold (converted to distance)
            
        Returns:
            List of job candidates with vector similarity scores
        """
        if not candidates:
            return []
        
        try:
            # Get job IDs for filtering
            candidate_job_ids = [candidate.job_id for candidate in candidates]
            
            # Convert similarity threshold to distance threshold
            # cosine_distance = 1 - cosine_similarity
            max_distance = 1.0 - min_score
            
            # Use pgvector's cosine distance operator (<->) for efficient similarity search
            from sqlalchemy import text
            import json
            sql = text("""
            SELECT 
                jc.job_id,
                1 - (jc.embedding <-> :resume_vector) as similarity_score,
                jc.embedding <-> :resume_vector as distance
            FROM job_chunks jc
            WHERE jc.job_id = ANY(:job_ids)
            AND jc.embedding IS NOT NULL
            AND jc.embedding <-> :resume_vector <= :max_distance
            ORDER BY jc.embedding <-> :resume_vector
            LIMIT :limit
            """)
            
            # Execute query with pgvector distance search
            result = self.session.execute(
                sql,
                {
                    "resume_vector": json.dumps(resume_embedding),
                    "job_ids": candidate_job_ids,
                    "max_distance": max_distance,
                    "limit": limit
                }
            ).fetchall()
            
            # Map results back to candidates
            job_scores = {str(row.job_id): row.similarity_score for row in result}
            
            scored_candidates = []
            for candidate in candidates:
                if candidate.job_id in job_scores:
                    candidate.vector_score = job_scores[candidate.job_id]
                    candidate.metadata['stage'] = 'vector'
                    scored_candidates.append(candidate)
            
            # Sort by vector score (highest first) and limit
            scored_candidates.sort(key=lambda x: x.vector_score or 0, reverse=True)
            result = scored_candidates[:limit]
            
            logger.info(f"Vector search found {len(result)} candidates with similarity >= {min_score}")
            return result
            
        except Exception as e:
            logger.error(f"pgvector search failed: {e}")
            # Fallback to manual cosine similarity calculation
            return await self._fallback_cosine_similarity(candidates, resume_embedding, limit, min_score)
    
    async def _fallback_cosine_similarity(
        self,
        candidates: List[JobCandidate],
        resume_embedding: List[float],
        limit: int,
        min_score: float
    ) -> List[JobCandidate]:
        """Fallback to manual cosine similarity calculation when pgvector fails"""
        try:
            # Get embeddings for job descriptions (from cache or generate)
            job_embeddings = await self._get_job_embeddings(candidates)
            
            # Calculate cosine similarity with resume
            scored_candidates = []
            
            for candidate, job_embedding in zip(candidates, job_embeddings):
                if job_embedding:
                    similarity = self._cosine_similarity(resume_embedding, job_embedding)
                    
                    if similarity >= min_score:
                        candidate.vector_score = similarity
                        candidate.metadata['stage'] = 'vector_fallback'
                        scored_candidates.append(candidate)
            
            # Sort by vector score and limit
            scored_candidates.sort(key=lambda x: x.vector_score or 0, reverse=True)
            return scored_candidates[:limit]
        except Exception as e:
            logger.error(f"Fallback cosine similarity also failed: {e}")
            return candidates[:limit]  # Ultimate fallback to FTS results
    
    async def _get_job_embeddings(self, candidates: List[JobCandidate]) -> List[Optional[List[float]]]:
        """Get embeddings for job descriptions with caching support"""
        embeddings = []
        
        # Check if embeddings already exist in database
        candidate_job_ids = [candidate.job_id for candidate in candidates]
        
        try:
            from sqlalchemy import text
            sql = text("""
            SELECT job_id, embedding 
            FROM job_chunks 
            WHERE job_id = ANY(:job_ids)
            AND embedding IS NOT NULL
            """)
            
            result = self.session.execute(sql, {"job_ids": candidate_job_ids}).fetchall()
            existing_embeddings = {str(row.job_id): row.embedding for row in result}
            
            # Return cached embeddings or None for missing ones
            for candidate in candidates:
                if candidate.job_id in existing_embeddings:
                    embeddings.append(existing_embeddings[candidate.job_id])
                else:
                    # Prepare embedding request for missing embeddings
                    embeddings.append(None)
                    
        except Exception as e:
            logger.warning(f"Could not retrieve cached job embeddings: {e}")
            # Fallback to generating embeddings fresh
            embeddings = await self._generate_job_embeddings(candidates)
        
        return embeddings
    
    async def _generate_job_embeddings(self, candidates: List[JobCandidate]) -> List[Optional[List[float]]]:
        """Generate fresh embeddings for job descriptions"""
        embeddings = []
        
        # Prepare embedding requests
        embedding_requests = []
        for candidate in candidates:
            # Use job title + description for embedding
            embed_text = f"{candidate.title}\n\n{candidate.description[:1000]}"  # Truncate long descriptions
            
            from libs.resume.embedding_service import EmbeddingRequest
            request = EmbeddingRequest(
                text=embed_text,
                model="text-embedding-ada-002",
                text_id=f"job_{candidate.job_id}",
                metadata={'job_id': candidate.job_id}
            )
            embedding_requests.append(request)
        
        # Get embeddings in batch
        try:
            responses = await self.embedding_service.embed_batch(embedding_requests)
            embeddings = [resp.embedding for resp in responses]
        except Exception as e:
            logger.error(f"Failed to generate job embeddings: {e}")
            embeddings = [None] * len(candidates)
        
        return embeddings
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(a * a for a in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)

class LLMScorer:
    """LLM-based job matching scorer"""
    
    def __init__(self):
        self.total_cost = 0.0
        self.requests_made = 0
    
    async def score_matches(
        self,
        candidates: List[JobCandidate],
        resume_profile: ResumeProfile,
        max_cost_cents: float = 50.0
    ) -> List[JobCandidate]:
        """Score job matches using LLM
        
        Args:
            candidates: Job candidates to score
            resume_profile: Resume profile for comparison
            max_cost_cents: Maximum cost limit in cents
            
        Returns:
            List of candidates with LLM scores and reasoning
        """
        if not candidates:
            return []
        
        scored_candidates = []
        current_cost = 0.0
        
        for candidate in candidates:
            # Check cost limit
            if current_cost >= max_cost_cents:
                logger.warning(f"Reached cost limit ${max_cost_cents/100:.2f}, stopping LLM scoring")
                break
            
            try:
                score, reasoning, cost = await self._score_single_match(candidate, resume_profile)
                
                candidate.llm_score = score
                candidate.llm_reasoning = reasoning
                candidate.metadata['llm_cost_cents'] = cost
                candidate.metadata['stage'] = 'llm'
                
                current_cost += cost
                self.total_cost += cost
                self.requests_made += 1
                
                scored_candidates.append(candidate)
                
                # Small delay to avoid rate limits
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"LLM scoring failed for job {candidate.job_id}: {e}")
                # Keep candidate without LLM score
                scored_candidates.append(candidate)
        
        return scored_candidates
    
    async def _score_single_match(
        self,
        candidate: JobCandidate,
        resume_profile: ResumeProfile
    ) -> Tuple[int, str, float]:
        """Score a single job-resume match using LLM
        
        Returns:
            (score, reasoning, cost_cents)
        """
        # Build prompt for LLM scoring
        prompt = self._build_scoring_prompt(candidate, resume_profile)
        
        # Mock LLM implementation (replace with actual LLM call)
        score, reasoning, cost = await self._mock_llm_call(prompt)
        
        return score, reasoning, cost
    
    def _build_scoring_prompt(self, candidate: JobCandidate, resume_profile: ResumeProfile) -> str:
        """Build LLM prompt for job matching"""
        prompt = f"""
You are an expert recruiter evaluating job-candidate fit. Score how well this candidate matches this job on a scale of 0-100.

JOB POSTING:
Title: {candidate.title}
Company: {candidate.company}
Required Skills: {', '.join(candidate.skills)}
Seniority: {candidate.seniority or 'Not specified'}
Description: {candidate.description[:500]}...

CANDIDATE PROFILE:
Skills: {', '.join(resume_profile.skills)}
Experience: {resume_profile.years_experience or 'Not specified'} years
Education: {resume_profile.education_level or 'Not specified'}
Preferred Roles: {', '.join(resume_profile.preferred_roles or [])}

SCORING CRITERIA:
- Skills alignment (40%): How well do candidate skills match job requirements?
- Experience level (30%): Is experience appropriate for seniority level?
- Role fit (20%): Does this match candidate's career direction?
- Culture/company fit (10%): General compatibility

Respond with JSON in this exact format:
{{
    "score": <integer 0-100>,
    "reasoning": "<2-3 sentence explanation of the score>",
    "key_strengths": ["<strength1>", "<strength2>"],
    "key_gaps": ["<gap1>", "<gap2>"]
}}
"""
        return prompt
    
    async def _mock_llm_call(self, prompt: str) -> Tuple[int, str, float]:
        """Make LLM call for job-resume scoring
        
        TODO: Replace with configurable LLM provider (OpenAI, Anthropic, etc.)
        Currently implemented as a mock for development/testing
        """
        # Simulate processing time
        await asyncio.sleep(0.05)
        
        # For now, return mock scores but with more realistic variation
        # In production, this would make actual API calls to OpenAI/Anthropic
        import hashlib
        import re
        
        # Extract some basic features from the prompt to make scoring less random
        prompt_lower = prompt.lower()
        
        # Look for skill matches (basic heuristic)
        skill_keywords = ['python', 'javascript', 'java', 'react', 'sql', 'aws', 'docker', 'kubernetes']
        matched_skills = sum(1 for skill in skill_keywords if skill in prompt_lower)
        
        # Look for experience level indicators
        senior_indicators = ['senior', '5+', 'lead', 'principal', 'architect']
        junior_indicators = ['junior', 'entry', '0-2', 'intern']
        
        is_senior_role = any(indicator in prompt_lower for indicator in senior_indicators)
        is_junior_role = any(indicator in prompt_lower for indicator in junior_indicators)
        
        # Base score on skill matches
        base_score = min(50 + (matched_skills * 10), 90)
        
        # Adjust for experience level mismatch
        if 'experience: 0' in prompt_lower and is_senior_role:
            base_score = max(base_score - 30, 20)  # Penalize junior candidate for senior role
        elif 'experience: ' in prompt_lower:
            exp_match = re.search(r'experience: (\d+)', prompt_lower)
            if exp_match:
                years = int(exp_match.group(1))
                if years >= 5 and is_senior_role:
                    base_score = min(base_score + 10, 95)  # Bonus for senior candidate for senior role
                elif years < 2 and is_junior_role:
                    base_score = min(base_score + 5, 90)   # Small bonus for appropriate level
        
        # Add some deterministic variation based on content hash
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        hash_variation = int(prompt_hash[:2], 16) % 20 - 10  # -10 to +9
        
        final_score = max(0, min(100, base_score + hash_variation))
        
        # Generate more realistic reasoning
        reasoning = f"Skills alignment: {matched_skills}/8 key skills found. "
        if is_senior_role:
            reasoning += "Senior-level role identified. "
        elif is_junior_role:
            reasoning += "Entry-level role identified. "
            
        reasoning += f"Overall compatibility score: {final_score}/100."
        
        # Mock cost based on typical OpenAI pricing (~$0.002 per 1K tokens)
        estimated_tokens = len(prompt.split()) * 1.3  # Rough token estimate
        cost_cents = (estimated_tokens / 1000) * 0.2  # $0.002 per 1K tokens
        
        return final_score, reasoning, cost_cents

class MatchingPipeline:
    """Main matching pipeline orchestrator"""
    
    def __init__(self, session, embedding_service, config: Optional[MatchingConfig] = None):
        self.session = session
        self.embedding_service = embedding_service
        self.config = config or MatchingConfig()
        
        self.fts_searcher = FTSSearcher(session)
        self.vector_searcher = VectorSearcher(session, embedding_service)
        self.llm_scorer = LLMScorer()
    
    async def match_resume_to_jobs(self, resume_profile: ResumeProfile) -> MatchingResult:
        """Run the full matching pipeline for a resume
        
        Args:
            resume_profile: Resume to match against jobs
            
        Returns:
            MatchingResult with ranked job matches
        """
        start_time = time.time()
        stages_completed = []
        total_cost = 0.0
        
        try:
            # Stage 1: FTS Prefilter
            logger.info(f"Starting FTS search for resume {resume_profile.resume_id}")
            candidates = await self.fts_searcher.search_jobs(
                resume_profile.fulltext,
                limit=self.config.fts_limit,
                min_score=self.config.fts_min_score
            )
            stages_completed.append(MatchingStage.FTS_PREFILTER)
            
            if not candidates:
                logger.warning("No candidates found in FTS stage")
                return self._create_empty_result(resume_profile.resume_id, stages_completed, 0.0, time.time() - start_time)
            
            # Stage 2: Vector Similarity
            logger.info(f"Starting vector search with {len(candidates)} candidates")
            
            # Get resume embedding if not provided
            if not resume_profile.embedding:
                resume_profile.embedding = await self._get_resume_embedding(resume_profile)
            
            if resume_profile.embedding:
                candidates = await self.vector_searcher.search_similar_jobs(
                    candidates,
                    resume_profile.embedding,
                    limit=self.config.vector_limit,
                    min_score=self.config.vector_min_score
                )
            stages_completed.append(MatchingStage.VECTOR_SIMILARITY)
            
            if not candidates:
                logger.warning("No candidates passed vector similarity stage")
                return self._create_empty_result(resume_profile.resume_id, stages_completed, 0.0, time.time() - start_time)
            
            # Stage 3: LLM Scoring
            logger.info(f"Starting LLM scoring with {len(candidates)} candidates")
            candidates = candidates[:self.config.llm_limit]  # Limit for cost control
            
            candidates = await self.llm_scorer.score_matches(
                candidates,
                resume_profile,
                max_cost_cents=self.config.max_cost_cents
            )
            stages_completed.append(MatchingStage.LLM_SCORING)
            total_cost = self.llm_scorer.total_cost
            
            # Final ranking
            ranked_candidates = self._calculate_final_scores(candidates)
            
            # Filter by minimum LLM score
            final_matches = [
                c for c in ranked_candidates 
                if (c.llm_score or 0) >= self.config.llm_min_score
            ]
            
            processing_time = time.time() - start_time
            
            logger.info(f"Matching completed: {len(final_matches)} final matches in {processing_time:.2f}s")
            
            return MatchingResult(
                resume_id=resume_profile.resume_id,
                matches=final_matches,
                total_candidates=len(candidates),
                stages_completed=stages_completed,
                total_cost_cents=total_cost,
                processing_time_seconds=processing_time,
                metadata={
                    'fts_candidates': len(candidates) if MatchingStage.FTS_PREFILTER in stages_completed else 0,
                    'vector_candidates': len(candidates) if MatchingStage.VECTOR_SIMILARITY in stages_completed else 0,
                    'llm_candidates': len(candidates) if MatchingStage.LLM_SCORING in stages_completed else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Matching pipeline failed: {e}")
            processing_time = time.time() - start_time
            return self._create_empty_result(resume_profile.resume_id, stages_completed, total_cost, processing_time)
    
    async def _get_resume_embedding(self, resume_profile: ResumeProfile) -> Optional[List[float]]:
        """Get embedding for resume"""
        try:
            response = await self.embedding_service.embed_text(
                resume_profile.fulltext,
                metadata={'resume_id': resume_profile.resume_id}
            )
            return response.embedding
        except Exception as e:
            logger.error(f"Failed to get resume embedding: {e}")
            return None
    
    def _calculate_final_scores(self, candidates: List[JobCandidate]) -> List[JobCandidate]:
        """Calculate final weighted scores and rank candidates"""
        for candidate in candidates:
            fts_score = candidate.fts_score or 0.0
            vector_score = candidate.vector_score or 0.0
            llm_score = (candidate.llm_score or 0) / 100.0  # Normalize to 0-1
            
            # Weighted final score
            final_score = (
                fts_score * self.config.fts_weight +
                vector_score * self.config.vector_weight +
                llm_score * self.config.llm_weight
            )
            
            candidate.final_score = final_score
        
        # Sort by final score
        candidates.sort(key=lambda x: x.final_score or 0, reverse=True)
        return candidates
    
    def _create_empty_result(
        self,
        resume_id: str,
        stages_completed: List[MatchingStage],
        total_cost: float,
        processing_time: float
    ) -> MatchingResult:
        """Create empty result for failed/no-match scenarios"""
        return MatchingResult(
            resume_id=resume_id,
            matches=[],
            total_candidates=0,
            stages_completed=stages_completed,
            total_cost_cents=total_cost,
            processing_time_seconds=processing_time
        )

def create_matching_pipeline(session, embedding_service, config: Optional[MatchingConfig] = None) -> MatchingPipeline:
    """Factory function to create matching pipeline"""
    return MatchingPipeline(session, embedding_service, config)