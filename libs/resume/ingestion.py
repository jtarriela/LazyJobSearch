"""Resume ingestion service - orchestrates the complete resume processing pipeline.

This service implements the resume ingest workflow:
1. File parsing (PDF, DOCX, TXT)
2. Content structuring and skills extraction  
3. Text chunking for embeddings
4. Embedding generation
5. Database persistence

Based on requirements from the gap analysis and CLI resume ingest command.
"""
from __future__ import annotations
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid

from libs.observability import get_logger, timer, counter
from libs.db.models import Resume, ResumeChunk
from libs.resume.parser import create_resume_parser, ParsedResume
from libs.resume.chunker import create_resume_chunker
from libs.resume.embedding_service import create_embedding_service, EmbeddingProvider
from libs.embed.versioning import EmbeddingVersionManager

logger = get_logger(__name__)

@dataclass 
class IngestedResume:
    """Result of successful resume ingestion"""
    resume_id: str
    parsed_resume: ParsedResume
    chunks: List[Dict[str, Any]]
    embedding_stats: Dict[str, Any]
    processing_time_ms: float

@dataclass
class IngestionError:
    """Resume ingestion error details"""
    stage: str  # parsing, chunking, embedding, persistence
    error_message: str
    file_path: Optional[str] = None


class ResumeIngestionService:
    """Orchestrates complete resume processing pipeline"""
    
    def __init__(self, db_session, embedding_version_manager: Optional[EmbeddingVersionManager] = None):
        self.db_session = db_session
        self.embedding_version_manager = embedding_version_manager
        
        # Initialize pipeline components
        self.parser = create_resume_parser()
        self.chunker = create_resume_chunker()
        self.embedding_service = None  # Will be initialized with provider
        
    async def ingest_resume_file(self, 
                               file_path: Path, 
                               user_id: Optional[str] = None,
                               embedding_provider: EmbeddingProvider = EmbeddingProvider.MOCK) -> IngestedResume:
        """
        Ingest a resume file through the complete pipeline.
        
        Args:
            file_path: Path to resume file (PDF, DOCX, TXT)
            user_id: Optional user ID to associate with resume
            embedding_provider: Provider for embedding generation
            
        Returns:
            IngestedResume with complete processing results
            
        Raises:
            IngestionError: If any stage of processing fails
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            with timer("resume_ingestion.total"):
                # Initialize embedding service
                if not self.embedding_service:
                    self.embedding_service = create_embedding_service(provider=embedding_provider)
                
                # Stage 1: Parse resume file
                logger.info("Starting resume parsing", file_path=str(file_path))
                parsed_resume = await self._parse_resume_file(file_path)
                counter("resume_ingestion.parse_success")
                
                # Stage 2: Chunk resume content
                logger.info("Starting resume chunking", resume_sections=len(parsed_resume.sections))
                chunks = await self._chunk_resume_content(parsed_resume)
                counter("resume_ingestion.chunk_success")
                
                # Stage 3: Generate embeddings
                logger.info("Starting embedding generation", num_chunks=len(chunks))
                embedding_stats = await self._generate_embeddings(chunks)
                counter("resume_ingestion.embedding_success") 
                
                # Stage 4: Persist to database
                logger.info("Starting database persistence")
                resume_id = await self._persist_resume_data(parsed_resume, chunks, user_id)
                counter("resume_ingestion.persistence_success")
                
                end_time = asyncio.get_event_loop().time()
                processing_time_ms = (end_time - start_time) * 1000
                
                counter("resume_ingestion.total_success")
                logger.info("Resume ingestion completed successfully", 
                           resume_id=resume_id, 
                           processing_time_ms=processing_time_ms,
                           chunks_created=len(chunks))
                
                return IngestedResume(
                    resume_id=resume_id,
                    parsed_resume=parsed_resume,
                    chunks=chunks,
                    embedding_stats=embedding_stats,
                    processing_time_ms=processing_time_ms
                )
                
        except Exception as e:
            counter("resume_ingestion.total_failure")
            logger.error("Resume ingestion failed", error=str(e), file_path=str(file_path))
            raise IngestionError(
                stage="unknown",
                error_message=str(e),
                file_path=str(file_path)
            )
    
    async def _parse_resume_file(self, file_path: Path) -> ParsedResume:
        """Parse resume file and extract structured content"""
        try:
            with timer("resume_ingestion.parsing"):
                parsed_resume = self.parser.parse_file(file_path)
                
                if not parsed_resume.fulltext.strip():
                    raise ValueError("No text content extracted from resume")
                    
                logger.debug("Resume parsing completed",
                           text_length=len(parsed_resume.fulltext),
                           skills_count=len(parsed_resume.skills),
                           sections_count=len(parsed_resume.sections))
                           
                return parsed_resume
                
        except Exception as e:
            counter("resume_ingestion.parse_failure")
            raise IngestionError(
                stage="parsing",
                error_message=f"Failed to parse resume: {e}",
                file_path=str(file_path)
            )
    
    async def _chunk_resume_content(self, parsed_resume: ParsedResume) -> List[Dict[str, Any]]:
        """Chunk resume content for embedding generation"""
        try:
            with timer("resume_ingestion.chunking"):
                chunks = self.chunker.chunk_resume(parsed_resume.fulltext, parsed_resume.sections)
                
                if not chunks:
                    raise ValueError("No chunks generated from resume content")
                
                # Convert to dict format for persistence
                chunk_dicts = []
                for chunk in chunks:
                    chunk_dict = {
                        'chunk_id': chunk.chunk_id,
                        'text': chunk.text,
                        'section': chunk.section,
                        'token_count': chunk.token_count,
                        'embedding': None,  # Will be filled by embedding stage
                        'metadata': chunk.metadata or {}
                    }
                    chunk_dicts.append(chunk_dict)
                
                logger.debug("Resume chunking completed", 
                           chunks_created=len(chunks),
                           total_tokens=sum(c.token_count for c in chunks))
                           
                return chunk_dicts
                
        except Exception as e:
            counter("resume_ingestion.chunk_failure")
            raise IngestionError(
                stage="chunking", 
                error_message=f"Failed to chunk resume: {e}"
            )
    
    async def _generate_embeddings(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate embeddings for resume chunks"""
        try:
            with timer("resume_ingestion.embeddings"):
                # Get current embedding version
                embedding_version = None
                if self.embedding_version_manager:
                    embedding_version = await self.embedding_version_manager.get_active_version()
                
                # Prepare embedding requests
                embedding_requests = []
                for chunk in chunks:
                    request = {
                        "text": chunk["text"],
                        "model": embedding_version.model_name if embedding_version else "text-embedding-ada-002"
                    }
                    embedding_requests.append(request)
                
                # Generate embeddings in batch
                embedding_responses = await self.embedding_service.embed_batch(embedding_requests)
                
                if len(embedding_responses) != len(chunks):
                    raise ValueError(f"Embedding count mismatch: got {len(embedding_responses)}, expected {len(chunks)}")
                
                # Attach embeddings to chunks
                for chunk, response in zip(chunks, embedding_responses):
                    chunk["embedding"] = response.embedding
                    chunk["embedding_version"] = embedding_version.version_id if embedding_version else "v1.0"
                    chunk["embedding_model"] = embedding_version.model_name if embedding_version else "text-embedding-ada-002"
                    
                # Collect embedding statistics
                embedding_stats = self.embedding_service.get_stats()
                embedding_stats["chunks_processed"] = len(chunks)
                embedding_stats["total_tokens"] = sum(c["token_count"] for c in chunks)
                
                logger.debug("Embedding generation completed",
                           chunks_embedded=len(chunks),
                           embedding_stats=embedding_stats)
                           
                return embedding_stats
                
        except Exception as e:
            counter("resume_ingestion.embedding_failure")
            raise IngestionError(
                stage="embedding",
                error_message=f"Failed to generate embeddings: {e}"
            )
    
    async def _persist_resume_data(self, parsed_resume: ParsedResume, chunks: List[Dict[str, Any]], user_id: Optional[str]) -> str:
        """Persist resume and chunks to database"""
        try:
            with timer("resume_ingestion.persistence"):
                # Create resume record
                resume_id = str(uuid.uuid4())
                resume = Resume(
                    id=resume_id,
                    fulltext=parsed_resume.fulltext,
                    sections_json=str(parsed_resume.sections),  # JSON serialize
                    skills_csv=",".join(parsed_resume.skills),
                    yoe_raw=parsed_resume.years_of_experience,
                    yoe_adjusted=parsed_resume.years_of_experience,  # Can be enhanced later
                    edu_level=parsed_resume.education_level or "",
                    file_url=str(parsed_resume.source_file) if parsed_resume.source_file else None,
                    created_at=datetime.utcnow()
                )
                
                self.db_session.add(resume)
                
                # Create resume chunk records
                resume_chunks = []
                for chunk in chunks:
                    resume_chunk = ResumeChunk(
                        id=str(uuid.uuid4()),
                        resume_id=resume_id,
                        chunk_text=chunk["text"],
                        embedding=chunk["embedding"],
                        token_count=chunk["token_count"],
                        embedding_version=chunk.get("embedding_version"),
                        embedding_model=chunk.get("embedding_model"),
                        needs_reembedding=False
                    )
                    resume_chunks.append(resume_chunk)
                    
                self.db_session.add_all(resume_chunks)
                
                # Commit transaction
                await self.db_session.commit()
                
                logger.debug("Database persistence completed",
                           resume_id=resume_id,
                           chunks_persisted=len(resume_chunks))
                           
                return resume_id
                
        except Exception as e:
            counter("resume_ingestion.persistence_failure")
            await self.db_session.rollback()
            raise IngestionError(
                stage="persistence",
                error_message=f"Failed to persist to database: {e}"
            )


def create_resume_ingestion_service(db_session, embedding_version_manager: Optional[EmbeddingVersionManager] = None) -> ResumeIngestionService:
    """Factory function to create resume ingestion service"""
    return ResumeIngestionService(db_session, embedding_version_manager)