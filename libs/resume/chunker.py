"""Resume chunking for embedding and vector search

Handles intelligent chunking of resume content with overlap and token counting.
Designed to work with embedding models and maintain semantic coherence.
"""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)

class ChunkStrategy(Enum):
    """Different chunking strategies for resume content"""
    SECTION_BASED = "section"  # Chunk by resume sections
    SLIDING_WINDOW = "sliding"  # Sliding window with overlap
    SEMANTIC = "semantic"  # Semantic boundary detection
    HYBRID = "hybrid"  # Combination of strategies

@dataclass
class ResumeChunk:
    """Represents a chunk of resume content"""
    chunk_id: str
    text: str
    token_count: int
    start_index: int
    end_index: int
    section: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ChunkingConfig:
    """Configuration for resume chunking"""
    max_tokens: int = 500
    overlap_tokens: int = 50
    min_chunk_tokens: int = 50
    strategy: ChunkStrategy = ChunkStrategy.HYBRID
    preserve_sections: bool = True
    include_section_headers: bool = True

class TokenCounter:
    """Simple token counter for text"""
    
    @staticmethod
    def count_tokens(text: str) -> int:
        """Count approximate tokens in text
        
        This is a simplified implementation. In production, you would use
        a proper tokenizer like tiktoken for OpenAI models or similar.
        """
        # Simple approximation: ~1.3 tokens per word for English text
        word_count = len(text.split())
        return int(word_count * 1.3)
    
    @staticmethod
    def truncate_to_tokens(text: str, max_tokens: int) -> str:
        """Truncate text to approximate token limit"""
        words = text.split()
        # Conservative estimate: keep fewer words to stay under token limit
        max_words = int(max_tokens / 1.5)
        
        if len(words) <= max_words:
            return text
        
        return ' '.join(words[:max_words])

class ResumeChunker:
    """Intelligent chunker for resume content"""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.token_counter = TokenCounter()
    
    def chunk_resume(self, resume_text: str, sections: Optional[Dict[str, str]] = None) -> List[ResumeChunk]:
        """Chunk a resume into overlapping segments
        
        Args:
            resume_text: Full resume text
            sections: Optional dict of section name -> content
            
        Returns:
            List of ResumeChunk objects
        """
        if self.config.strategy == ChunkStrategy.SECTION_BASED:
            return self._chunk_by_sections(resume_text, sections or {})
        elif self.config.strategy == ChunkStrategy.SLIDING_WINDOW:
            return self._chunk_sliding_window(resume_text)
        elif self.config.strategy == ChunkStrategy.SEMANTIC:
            return self._chunk_semantic(resume_text)
        else:  # HYBRID
            return self._chunk_hybrid(resume_text, sections or {})
    
    def _chunk_by_sections(self, resume_text: str, sections: Dict[str, str]) -> List[ResumeChunk]:
        """Chunk resume by sections with intelligent splitting if needed"""
        chunks = []
        
        if not sections:
            # Fallback to sliding window if no sections
            return self._chunk_sliding_window(resume_text)
        
        for section_name, section_content in sections.items():
            section_chunks = self._split_large_section(
                section_content, 
                section_name, 
                resume_text
            )
            chunks.extend(section_chunks)
        
        # Handle any content not captured in sections
        remaining_content = self._get_remaining_content(resume_text, sections)
        if remaining_content:
            remaining_chunks = self._chunk_sliding_window(remaining_content, section="other")
            chunks.extend(remaining_chunks)
        
        return chunks
    
    def _chunk_sliding_window(self, text: str, section: Optional[str] = None) -> List[ResumeChunk]:
        """Create overlapping chunks using sliding window approach"""
        chunks = []
        words = text.split()
        
        if not words:
            return chunks
        
        # Estimate words per chunk based on token limits
        words_per_chunk = int(self.config.max_tokens / 1.3)
        overlap_words = int(self.config.overlap_tokens / 1.3)
        
        start_idx = 0
        chunk_id = 0
        
        while start_idx < len(words):
            end_idx = min(start_idx + words_per_chunk, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            
            # Skip if chunk is too small (unless it's the last chunk)
            if (len(chunk_words) < self.config.min_chunk_tokens / 1.3 and 
                end_idx < len(words)):
                start_idx += max(1, words_per_chunk - overlap_words)
                continue
            
            token_count = self.token_counter.count_tokens(chunk_text)
            
            chunk = ResumeChunk(
                chunk_id=f"chunk_{chunk_id:03d}",
                text=chunk_text,
                token_count=token_count,
                start_index=start_idx,
                end_index=end_idx,
                section=section,
                metadata={
                    'strategy': 'sliding_window',
                    'word_count': len(chunk_words)
                }
            )
            
            chunks.append(chunk)
            chunk_id += 1
            
            # Move window forward
            if end_idx >= len(words):
                break
            
            start_idx += max(1, words_per_chunk - overlap_words)
        
        return chunks
    
    def _chunk_semantic(self, text: str) -> List[ResumeChunk]:
        """Chunk based on semantic boundaries (paragraphs, bullet points, etc.)"""
        chunks = []
        
        # Split by double newlines (paragraphs) and bullet points
        semantic_boundaries = re.split(r'\n\s*\n|(?=^\s*[•\-\*])', text, flags=re.MULTILINE)
        semantic_boundaries = [chunk.strip() for chunk in semantic_boundaries if chunk.strip()]
        
        current_chunk = ""
        current_tokens = 0
        chunk_id = 0
        
        for boundary in semantic_boundaries:
            boundary_tokens = self.token_counter.count_tokens(boundary)
            
            # If adding this boundary would exceed max tokens, finalize current chunk
            if (current_tokens + boundary_tokens > self.config.max_tokens and 
                current_chunk and 
                current_tokens >= self.config.min_chunk_tokens):
                
                chunk = ResumeChunk(
                    chunk_id=f"semantic_{chunk_id:03d}",
                    text=current_chunk.strip(),
                    token_count=current_tokens,
                    start_index=0,  # Would need more complex tracking for exact indices
                    end_index=0,
                    metadata={
                        'strategy': 'semantic',
                        'boundary_count': len(current_chunk.split('\n\n'))
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + boundary if overlap_text else boundary
                current_tokens = self.token_counter.count_tokens(current_chunk)
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + boundary
                else:
                    current_chunk = boundary
                current_tokens += boundary_tokens
        
        # Add final chunk if it exists
        if current_chunk and current_tokens >= self.config.min_chunk_tokens:
            chunk = ResumeChunk(
                chunk_id=f"semantic_{chunk_id:03d}",
                text=current_chunk.strip(),
                token_count=current_tokens,
                start_index=0,
                end_index=0,
                metadata={
                    'strategy': 'semantic',
                    'boundary_count': len(current_chunk.split('\n\n'))
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_hybrid(self, resume_text: str, sections: Dict[str, str]) -> List[ResumeChunk]:
        """Hybrid approach combining section-based and semantic chunking"""
        chunks = []
        
        # First, try section-based chunking for structured sections
        structured_sections = ['experience', 'education', 'projects', 'skills']
        processed_content = ""
        
        for section_name in structured_sections:
            if section_name in sections:
                section_content = sections[section_name]
                if self.config.include_section_headers:
                    section_content = f"{section_name.title()}:\n{section_content}"
                
                # Use semantic chunking within each section
                section_chunks = self._chunk_semantic(section_content)
                for chunk in section_chunks:
                    chunk.section = section_name
                    chunk.chunk_id = f"{section_name}_{chunk.chunk_id}"
                
                chunks.extend(section_chunks)
                processed_content += section_content + "\n\n"
        
        # Handle remaining content with sliding window
        remaining_content = self._get_remaining_content(resume_text, 
                                                      {k: v for k, v in sections.items() 
                                                       if k in structured_sections})
        if remaining_content:
            remaining_chunks = self._chunk_sliding_window(remaining_content, section="other")
            chunks.extend(remaining_chunks)
        
        return chunks
    
    def _split_large_section(self, section_content: str, section_name: str, full_text: str) -> List[ResumeChunk]:
        """Split a large section into multiple chunks if needed"""
        token_count = self.token_counter.count_tokens(section_content)
        
        if token_count <= self.config.max_tokens:
            # Section fits in one chunk
            chunk = ResumeChunk(
                chunk_id=f"{section_name}_001",
                text=section_content,
                token_count=token_count,
                start_index=full_text.find(section_content) if section_content in full_text else 0,
                end_index=full_text.find(section_content) + len(section_content) if section_content in full_text else len(section_content),
                section=section_name,
                metadata={'strategy': 'section_based', 'split': False}
            )
            return [chunk]
        else:
            # Need to split the section
            if section_name in ['experience', 'projects']:
                # Split by job/project entries
                return self._split_by_entries(section_content, section_name)
            else:
                # Use sliding window for other sections
                chunks = self._chunk_sliding_window(section_content, section=section_name)
                # Update chunk IDs to include section name
                for i, chunk in enumerate(chunks):
                    chunk.chunk_id = f"{section_name}_{i+1:03d}"
                    chunk.metadata['strategy'] = 'section_split'
                return chunks
    
    def _split_by_entries(self, content: str, section_name: str) -> List[ResumeChunk]:
        """Split experience or projects section by individual entries"""
        chunks = []
        
        # Look for common entry separators
        entry_patterns = [
            r'\n(?=[A-Z][^a-z]*(?:Engineer|Manager|Developer|Analyst|Specialist|Director|Lead))',  # Job titles
            r'\n(?=\d{4}[\s\-]\d{4})',  # Date ranges
            r'\n(?=[A-Z][a-z]+\s[A-Z][a-z]+)',  # Company names (simple pattern)
        ]
        
        entries = [content]  # Start with full content
        
        for pattern in entry_patterns:
            new_entries = []
            for entry in entries:
                split_entries = re.split(pattern, entry)
                new_entries.extend([e.strip() for e in split_entries if e.strip()])
            entries = new_entries
        
        chunk_id = 1
        for entry in entries:
            token_count = self.token_counter.count_tokens(entry)
            
            if token_count >= self.config.min_chunk_tokens:
                # If entry is still too large, split with sliding window
                if token_count > self.config.max_tokens:
                    sub_chunks = self._chunk_sliding_window(entry, section=section_name)
                    for sub_chunk in sub_chunks:
                        sub_chunk.chunk_id = f"{section_name}_entry_{chunk_id:03d}"
                        chunks.append(sub_chunk)
                        chunk_id += 1
                else:
                    chunk = ResumeChunk(
                        chunk_id=f"{section_name}_entry_{chunk_id:03d}",
                        text=entry,
                        token_count=token_count,
                        start_index=0,
                        end_index=len(entry),
                        section=section_name,
                        metadata={'strategy': 'entry_based'}
                    )
                    chunks.append(chunk)
                    chunk_id += 1
        
        return chunks
    
    def _get_remaining_content(self, full_text: str, processed_sections: Dict[str, str]) -> str:
        """Extract content not captured in processed sections"""
        remaining = full_text
        
        for section_content in processed_sections.values():
            remaining = remaining.replace(section_content, "")
        
        # Clean up extra whitespace
        remaining = re.sub(r'\n\s*\n', '\n\n', remaining).strip()
        
        return remaining if len(remaining) > 50 else ""  # Only return if substantial content
    
    def _get_overlap_text(self, text: str) -> str:
        """Extract overlap text from the end of current chunk"""
        words = text.split()
        overlap_words = int(self.config.overlap_tokens / 1.3)
        
        if len(words) <= overlap_words:
            return text
        
        return ' '.join(words[-overlap_words:])
    
    def merge_small_chunks(self, chunks: List[ResumeChunk]) -> List[ResumeChunk]:
        """Merge chunks that are too small to be useful"""
        if not chunks:
            return chunks
        
        merged_chunks = []
        current_chunk = None
        
        for chunk in chunks:
            if chunk.token_count < self.config.min_chunk_tokens:
                if current_chunk is None:
                    current_chunk = chunk
                else:
                    # Merge with current chunk
                    merged_text = current_chunk.text + "\n\n" + chunk.text
                    merged_tokens = self.token_counter.count_tokens(merged_text)
                    
                    if merged_tokens <= self.config.max_tokens:
                        current_chunk.text = merged_text
                        current_chunk.token_count = merged_tokens
                        current_chunk.end_index = chunk.end_index
                        current_chunk.metadata['merged'] = True
                    else:
                        # Can't merge without exceeding limit, finalize current
                        merged_chunks.append(current_chunk)
                        current_chunk = chunk
            else:
                # Chunk is large enough, finalize any pending merge
                if current_chunk is not None:
                    merged_chunks.append(current_chunk)
                    current_chunk = None
                merged_chunks.append(chunk)
        
        # Add final chunk if exists
        if current_chunk is not None:
            merged_chunks.append(current_chunk)
        
        return merged_chunks

def create_resume_chunker(config: Optional[ChunkingConfig] = None) -> ResumeChunker:
    """Factory function to create a configured resume chunker"""
    return ResumeChunker(config)