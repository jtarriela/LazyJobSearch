"""Main resume processing pipeline."""

import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

from sqlalchemy.orm import Session
from libs.db.models import Resume, ResumeChunk
from libs.db.session import get_session
from libs.nlp.chunkers import SemanticChunker
from libs.nlp.extractors import SkillExtractor, YearsOfExperienceExtractor
from libs.embed import OpenAIEmbeddingProvider
from .parsers import PDFParser, DocxParser


class ResumeProcessor:
    """Process resume files through the complete ingestion pipeline.
    
    Handles parsing, chunking, skill extraction, embedding generation,
    and database storage for resumes.
    """
    
    def __init__(self, embedding_provider: Optional[OpenAIEmbeddingProvider] = None):
        """Initialize processor with dependencies."""
        self.embedding_provider = embedding_provider or OpenAIEmbeddingProvider()
        self.chunker = SemanticChunker(target_chunk_size=400, max_chunk_size=800)
        self.skill_extractor = SkillExtractor()
        self.experience_extractor = YearsOfExperienceExtractor()
        
        # File parsers
        self.parsers = {
            '.pdf': PDFParser(),
            '.docx': DocxParser(),
            '.doc': DocxParser(),  # DocxParser can handle .doc too
        }
    
    def process_resume_file(self, file_path: Path, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a resume file through the complete pipeline.
        
        Args:
            file_path: Path to the resume file
            user_id: Optional user ID to associate with the resume
            
        Returns:
            Dict with processing results and resume ID
        """
        try:
            # Step 1: Parse the resume file
            parsed_content = self._parse_file(file_path)
            
            # Step 2: Extract structured information
            fulltext = parsed_content['fulltext']
            sections = parsed_content['sections']
            
            # Extract skills and experience
            skills = self.skill_extractor.extract_skills(fulltext)
            experience_data = self.experience_extractor.extract_experience(fulltext)
            education_data = self.experience_extractor.extract_education_details(fulltext)
            
            # Step 3: Create resume record
            with get_session() as session:
                resume = Resume(
                    user_id=uuid.UUID(user_id) if user_id else None,
                    fulltext=fulltext,
                    sections_json=sections,
                    skills_csv=self._format_skills_csv(skills),
                    yoe_raw=experience_data['raw_experience'],
                    yoe_adjusted=experience_data['total_experience'],
                    edu_level=education_data.get('highest_degree'),
                    file_url=str(file_path),
                    is_active=True
                )
                
                session.add(resume)
                session.flush()  # Get the ID
                
                # Step 4: Chunk the resume content
                chunks = self._chunk_resume_content(fulltext, sections)
                
                # Step 5: Generate embeddings and store chunks
                chunk_records = []
                for chunk_data in chunks:
                    # Generate embedding
                    embedding = self.embedding_provider.embed_text(chunk_data['text'])
                    
                    chunk_record = ResumeChunk(
                        resume_id=resume.id,
                        chunk_text=chunk_data['text'],
                        embedding=embedding.tolist(),  # Convert numpy array to list
                        token_count=chunk_data.get('token_count', 0),
                        section_type=chunk_data.get('section_type'),
                        chunk_index=chunk_data.get('index', 0)
                    )
                    
                    chunk_records.append(chunk_record)
                
                session.add_all(chunk_records)
                session.commit()
                
                return {
                    'success': True,
                    'resume_id': str(resume.id),
                    'fulltext_length': len(fulltext),
                    'chunks_created': len(chunk_records),
                    'skills_found': len(skills),
                    'experience_years': experience_data['total_experience'],
                    'education_level': education_data.get('highest_degree')
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'resume_id': None
            }
    
    def _parse_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse a resume file based on its extension."""
        extension = file_path.suffix.lower()
        
        if extension not in self.parsers:
            raise ValueError(f"Unsupported file type: {extension}")
        
        parser = self.parsers[extension]
        return parser.parse(file_path)
    
    def _chunk_resume_content(self, fulltext: str, sections: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk resume content with section awareness."""
        # If we have structured sections, chunk each section separately
        if sections and 'sections' in sections:
            all_chunks = []
            
            for section_name, section_content in sections['sections'].items():
                if isinstance(section_content, str) and section_content.strip():
                    section_chunks = self.chunker.chunk_text(
                        section_content,
                        metadata={
                            'section_type': section_name,
                            'doc_type': 'resume'
                        }
                    )
                    all_chunks.extend(section_chunks)
            
            return all_chunks
        else:
            # Fallback to chunking the full text
            return self.chunker.chunk_text(
                fulltext,
                metadata={'doc_type': 'resume'}
            )
    
    def _format_skills_csv(self, skills: List[Any]) -> str:
        """Format extracted skills as CSV string."""
        if not skills:
            return ""
        
        skill_names = [skill.skill for skill in skills]
        return ", ".join(skill_names[:20])  # Limit to top 20 skills
    
    def reprocess_embeddings(self, resume_id: str) -> Dict[str, Any]:
        """Regenerate embeddings for an existing resume.
        
        Useful when switching embedding providers or models.
        """
        try:
            with get_session() as session:
                # Get all chunks for this resume
                chunks = session.query(ResumeChunk).filter(
                    ResumeChunk.resume_id == uuid.UUID(resume_id)
                ).all()
                
                if not chunks:
                    return {'success': False, 'error': 'No chunks found for resume'}
                
                # Regenerate embeddings
                updated_count = 0
                for chunk in chunks:
                    new_embedding = self.embedding_provider.embed_text(chunk.chunk_text)
                    chunk.embedding = new_embedding.tolist()
                    updated_count += 1
                
                session.commit()
                
                return {
                    'success': True,
                    'chunks_updated': updated_count,
                    'resume_id': resume_id
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'resume_id': resume_id
            }