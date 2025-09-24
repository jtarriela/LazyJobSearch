"""Match persistence service

Handles storing and retrieving match results with deduplication and versioning.
Integrates with the matching pipeline to persist results to the database.
"""
from __future__ import annotations
import logging
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import asdict

from libs.db.models import Match, Job, Company
from libs.matching.pipeline import MatchingResult, JobCandidate

logger = logging.getLogger(__name__)

class MatchPersistenceService:
    """Service for persisting match results to database"""
    
    def __init__(self, session):
        self.session = session
    
    def save_matching_result(self, result: MatchingResult) -> List[str]:
        """Save matching result to database
        
        Args:
            result: MatchingResult to persist
            
        Returns:
            List of match IDs that were created/updated
        """
        match_ids = []
        
        try:
            self.session.begin()
            
            # Clear existing matches for this resume (optional - keeps latest only)
            # self._clear_existing_matches(result.resume_id)
            
            for candidate in result.matches:
                match_id = self._save_single_match(result.resume_id, candidate, result)
                if match_id:
                    match_ids.append(match_id)
            
            self.session.commit()
            logger.info(f"Saved {len(match_ids)} matches for resume {result.resume_id}")
            
        except Exception as e:
            self.session.rollback()
            logger.error(f"Failed to save matching result: {e}")
            raise
        
        return match_ids
    
    def _save_single_match(
        self, 
        resume_id: str, 
        candidate: JobCandidate, 
        result: MatchingResult
    ) -> Optional[str]:
        """Save a single job match"""
        try:
            # Check for existing match to avoid duplicates
            existing_match = self.session.query(Match).filter(
                Match.resume_id == resume_id,
                Match.job_id == candidate.job_id
            ).first()
            
            if existing_match:
                # Update existing match with new scores
                existing_match.vector_score = candidate.vector_score
                existing_match.llm_score = candidate.llm_score
                existing_match.action = self._determine_action(candidate)
                existing_match.reasoning = candidate.llm_reasoning
                existing_match.scored_at = datetime.now()
                
                # Update metadata
                metadata = json.loads(existing_match.metadata or '{}')
                metadata.update({
                    'fts_score': candidate.fts_score,
                    'final_score': candidate.final_score,
                    'pipeline_cost_cents': result.total_cost_cents,
                    'updated_at': datetime.now().isoformat()
                })
                existing_match.metadata = json.dumps(metadata)
                
                return str(existing_match.id)
            else:
                # Create new match
                match = Match(
                    job_id=candidate.job_id,
                    resume_id=resume_id,
                    vector_score=candidate.vector_score,
                    llm_score=candidate.llm_score,
                    action=self._determine_action(candidate),
                    reasoning=candidate.llm_reasoning,
                    llm_model="mock-llm-v1",  # Would be actual model name
                    prompt_hash=self._hash_prompt(candidate, resume_id),
                    scored_at=datetime.now(),
                    metadata=json.dumps({
                        'fts_score': candidate.fts_score,
                        'final_score': candidate.final_score,
                        'stages_completed': [s.value for s in result.stages_completed],
                        'pipeline_cost_cents': result.total_cost_cents,
                        'created_at': datetime.now().isoformat()
                    })
                )
                
                self.session.add(match)
                self.session.flush()  # Get the ID
                
                return str(match.id)
                
        except Exception as e:
            logger.error(f"Failed to save match for job {candidate.job_id}: {e}")
            return None
    
    def _determine_action(self, candidate: JobCandidate) -> str:
        """Determine recommended action based on scores"""
        llm_score = candidate.llm_score or 0
        final_score = candidate.final_score or 0.0
        
        if llm_score >= 85 and final_score >= 0.8:
            return "APPLY_HIGH"
        elif llm_score >= 70 and final_score >= 0.6:
            return "APPLY_MEDIUM"
        elif llm_score >= 60 and final_score >= 0.4:
            return "REVIEW"
        else:
            return "SKIP"
    
    def _hash_prompt(self, candidate: JobCandidate, resume_id: str) -> str:
        """Generate hash for prompt used in LLM scoring"""
        import hashlib
        
        prompt_data = f"{resume_id}:{candidate.job_id}:{candidate.title}:{len(candidate.description)}"
        return hashlib.md5(prompt_data.encode()).hexdigest()[:16]
    
    def get_matches_for_resume(
        self, 
        resume_id: str, 
        limit: int = 50,
        min_score: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get saved matches for a resume
        
        Args:
            resume_id: Resume ID to get matches for
            limit: Maximum number of matches to return
            min_score: Minimum LLM score filter
            
        Returns:
            List of match dictionaries with job details
        """
        try:
            query = self.session.query(Match).filter(Match.resume_id == resume_id)
            
            if min_score is not None:
                query = query.filter(Match.llm_score >= min_score)
            
            matches = query.order_by(Match.llm_score.desc()).limit(limit).all()
            
            result = []
            for match in matches:
                # Get job details
                job = self.session.query(Job).filter(Job.id == match.job_id).first()
                if not job:
                    continue
                
                company = self.session.query(Company).filter(Company.id == job.company_id).first()
                
                match_dict = {
                    'match_id': str(match.id),
                    'job_id': str(match.job_id),
                    'resume_id': str(match.resume_id),
                    'vector_score': match.vector_score,
                    'llm_score': match.llm_score,
                    'action': match.action,
                    'reasoning': match.reasoning,
                    'scored_at': match.scored_at.isoformat() if match.scored_at else None,
                    'job': {
                        'title': job.title,
                        'company': company.name if company else 'Unknown',
                        'location': job.location,
                        'url': job.url,
                        'seniority': job.seniority,
                        'skills': job.jd_skills_csv.split(',') if job.jd_skills_csv else []
                    },
                    'metadata': json.loads(match.metadata) if match.metadata else {}
                }
                result.append(match_dict)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get matches for resume {resume_id}: {e}")
            return []
    
    def get_match_statistics(self, resume_id: str) -> Dict[str, Any]:
        """Get match statistics for a resume"""
        try:
            from sqlalchemy import func
            
            # Get total matches and action breakdown
            stats = self.session.query(
                func.count(Match.id).label('total_matches'),
                func.avg(Match.llm_score).label('avg_score'),
                func.max(Match.llm_score).label('max_score'),
                func.min(Match.llm_score).label('min_score')
            ).filter(Match.resume_id == resume_id).first()
            
            # Get action breakdown
            action_stats = self.session.query(
                Match.action,
                func.count(Match.id).label('count')
            ).filter(Match.resume_id == resume_id).group_by(Match.action).all()
            
            action_breakdown = {action: count for action, count in action_stats}
            
            return {
                'total_matches': stats.total_matches or 0,
                'avg_score': float(stats.avg_score) if stats.avg_score else 0.0,
                'max_score': stats.max_score or 0,
                'min_score': stats.min_score or 0,
                'action_breakdown': action_breakdown,
                'high_quality_matches': action_breakdown.get('APPLY_HIGH', 0) + action_breakdown.get('APPLY_MEDIUM', 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get match statistics: {e}")
            return {'error': str(e)}
    
    def _clear_existing_matches(self, resume_id: str) -> int:
        """Clear existing matches for a resume (optional cleanup)"""
        try:
            deleted_count = self.session.query(Match).filter(
                Match.resume_id == resume_id
            ).delete()
            
            logger.info(f"Cleared {deleted_count} existing matches for resume {resume_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to clear existing matches: {e}")
            return 0

class MatchQueryService:
    """Service for querying and analyzing matches"""
    
    def __init__(self, session):
        self.session = session
    
    def find_similar_matches(
        self, 
        match_id: str, 
        similarity_threshold: float = 0.1,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find matches similar to a given match"""
        try:
            # Get the reference match
            ref_match = self.session.query(Match).filter(Match.id == match_id).first()
            if not ref_match:
                return []
            
            # Find matches with similar vector scores
            similar_matches = self.session.query(Match).filter(
                Match.id != match_id,
                Match.vector_score.between(
                    (ref_match.vector_score or 0) - similarity_threshold,
                    (ref_match.vector_score or 0) + similarity_threshold
                )
            ).limit(limit).all()
            
            return [self._match_to_dict(match) for match in similar_matches]
            
        except Exception as e:
            logger.error(f"Failed to find similar matches: {e}")
            return []
    
    def get_top_matches_across_resumes(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get top matches across all resumes"""
        try:
            top_matches = self.session.query(Match).order_by(
                Match.llm_score.desc()
            ).limit(limit).all()
            
            return [self._match_to_dict(match) for match in top_matches]
            
        except Exception as e:
            logger.error(f"Failed to get top matches: {e}")
            return []
    
    def _match_to_dict(self, match: Match) -> Dict[str, Any]:
        """Convert Match object to dictionary"""
        return {
            'match_id': str(match.id),
            'job_id': str(match.job_id),
            'resume_id': str(match.resume_id),
            'vector_score': match.vector_score,
            'llm_score': match.llm_score,
            'action': match.action,
            'reasoning': match.reasoning,
            'scored_at': match.scored_at.isoformat() if match.scored_at else None,
            'metadata': json.loads(match.metadata) if match.metadata else {}
        }

def create_match_persistence_service(session) -> MatchPersistenceService:
    """Factory function to create match persistence service"""
    return MatchPersistenceService(session)

def create_match_query_service(session) -> MatchQueryService:
    """Factory function to create match query service"""
    return MatchQueryService(session)