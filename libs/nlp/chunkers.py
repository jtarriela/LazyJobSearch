"""Text chunking strategies for optimal embedding."""

import re
from typing import List, Dict, Any
from abc import ABC, abstractmethod


class TextChunker(ABC):
    """Base class for text chunking strategies."""
    
    @abstractmethod
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into semantic chunks.
        
        Returns:
            List of chunk dictionaries with 'text', 'index', and optional metadata.
        """
        pass


class FixedSizeChunker(TextChunker):
    """Simple fixed-size chunker with overlap for context preservation."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into fixed-size chunks with overlap."""
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary if we're not at the end
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                sentence_break = self._find_sentence_break(chunk_text[-100:])
                if sentence_break:
                    chunk_text = chunk_text[:len(chunk_text) - 100 + sentence_break]
            
            chunk = {
                'text': chunk_text.strip(),
                'index': index,
                'char_start': start,
                'char_end': start + len(chunk_text),
                'token_count': self._estimate_tokens(chunk_text)
            }
            
            if metadata:
                chunk.update(metadata)
            
            chunks.append(chunk)
            
            # Move start position with overlap
            start = start + len(chunk_text) - self.overlap
            index += 1
        
        return [chunk for chunk in chunks if chunk['text']]
    
    def _find_sentence_break(self, text: str) -> int:
        """Find the best sentence break point in text."""
        # Look for sentence endings
        for pattern in [r'\.[\s\n]', r'![\s\n]', r'\?[\s\n]', r'\.[\"\'][\s\n]']:
            matches = list(re.finditer(pattern, text))
            if matches:
                return matches[-1].end() - 1  # Position after the punctuation
        
        # Fallback to paragraph or line breaks
        for pattern in [r'\n\n', r'\n']:
            matches = list(re.finditer(pattern, text))
            if matches:
                return matches[-1].end()
        
        return None
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters for English)."""
        return max(1, len(text) // 4)


class SemanticChunker(TextChunker):
    """Semantic chunker that preserves document structure.
    
    This chunker is aware of resume/job description structure and tries
    to keep related content together (e.g., a complete experience entry).
    """
    
    def __init__(self, target_chunk_size: int = 400, max_chunk_size: int = 800):
        """Initialize semantic chunker.
        
        Args:
            target_chunk_size: Preferred chunk size in characters
            max_chunk_size: Maximum allowed chunk size
        """
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text semantically based on document structure."""
        if not text.strip():
            return []
        
        # Detect document type for specialized handling
        doc_type = self._detect_document_type(text, metadata)
        
        if doc_type == 'resume':
            return self._chunk_resume(text, metadata)
        elif doc_type == 'job_description':
            return self._chunk_job_description(text, metadata)
        else:
            # Fallback to paragraph-based chunking
            return self._chunk_by_paragraphs(text, metadata)
    
    def _detect_document_type(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """Detect whether this is a resume, job description, or other."""
        if metadata and 'doc_type' in metadata:
            return metadata['doc_type']
        
        # Simple heuristics based on common patterns
        text_lower = text.lower()
        
        resume_indicators = ['experience', 'education', 'skills', 'objective', 'summary']
        job_indicators = ['responsibilities', 'requirements', 'qualifications', 'we are looking']
        
        resume_score = sum(1 for indicator in resume_indicators if indicator in text_lower)
        job_score = sum(1 for indicator in job_indicators if indicator in text_lower)
        
        return 'resume' if resume_score > job_score else 'job_description'
    
    def _chunk_resume(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Chunk resume text by sections and experience entries."""
        chunks = []
        
        # Split by common resume section headers
        section_patterns = [
            r'\n(?:EXPERIENCE|WORK EXPERIENCE|PROFESSIONAL EXPERIENCE)\n',
            r'\n(?:EDUCATION|ACADEMIC BACKGROUND)\n',
            r'\n(?:SKILLS|TECHNICAL SKILLS|CORE COMPETENCIES)\n',
            r'\n(?:PROJECTS|KEY PROJECTS)\n',
            r'\n(?:SUMMARY|PROFESSIONAL SUMMARY|OBJECTIVE)\n'
        ]
        
        sections = self._split_by_patterns(text, section_patterns)
        
        for i, section in enumerate(sections):
            section_chunks = self._chunk_section(section, i, metadata)
            chunks.extend(section_chunks)
        
        return chunks
    
    def _chunk_job_description(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Chunk job description by responsibilities and requirements."""
        chunks = []
        
        # Common JD section patterns
        section_patterns = [
            r'\n(?:RESPONSIBILITIES|KEY RESPONSIBILITIES|DUTIES)\n',
            r'\n(?:REQUIREMENTS|QUALIFICATIONS|REQUIRED QUALIFICATIONS)\n',
            r'\n(?:PREFERRED|NICE TO HAVE|BONUS)\n',
            r'\n(?:BENEFITS|COMPENSATION|PERKS)\n',
            r'\n(?:ABOUT|COMPANY|ROLE)\n'
        ]
        
        sections = self._split_by_patterns(text, section_patterns)
        
        for i, section in enumerate(sections):
            section_chunks = self._chunk_section(section, i, metadata)
            chunks.extend(section_chunks)
        
        return chunks
    
    def _chunk_by_paragraphs(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Fallback chunking by paragraphs."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            # If adding this paragraph would exceed target size, start new chunk
            if current_chunk and len(current_chunk) + len(para) > self.target_chunk_size:
                chunks.append(self._make_chunk(current_chunk, chunk_index, metadata))
                current_chunk = para
                chunk_index += 1
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para
            
            # If current chunk exceeds max size, force a break
            if len(current_chunk) > self.max_chunk_size:
                chunks.append(self._make_chunk(current_chunk, chunk_index, metadata))
                current_chunk = ""
                chunk_index += 1
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._make_chunk(current_chunk, chunk_index, metadata))
        
        return chunks
    
    def _split_by_patterns(self, text: str, patterns: List[str]) -> List[str]:
        """Split text by regex patterns."""
        import re
        
        # Combine all patterns into one
        combined_pattern = '|'.join(f'({pattern})' for pattern in patterns)
        sections = re.split(combined_pattern, text, flags=re.IGNORECASE)
        
        # Filter out empty sections and pattern matches
        return [section.strip() for section in sections if section and section.strip()]
    
    def _chunk_section(self, section: str, section_index: int, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Chunk a document section."""
        if len(section) <= self.target_chunk_size:
            return [self._make_chunk(section, 0, metadata, section_index)]
        
        # For longer sections, break by sentences or bullet points
        chunks = []
        sentences = self._split_sentences(section)
        
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            if current_chunk and len(current_chunk) + len(sentence) > self.target_chunk_size:
                chunks.append(self._make_chunk(current_chunk, chunk_index, metadata, section_index))
                current_chunk = sentence
                chunk_index += 1
            else:
                current_chunk = current_chunk + " " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(self._make_chunk(current_chunk, chunk_index, metadata, section_index))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling bullet points."""
        # Split by bullet points first
        bullet_pattern = r'[\n\r][\s]*[•\-\*]\s*'
        parts = re.split(bullet_pattern, text)
        
        sentences = []
        for part in parts:
            # Further split by sentence endings
            part_sentences = re.split(r'[.!?]+\s+', part.strip())
            sentences.extend([s.strip() for s in part_sentences if s.strip()])
        
        return sentences
    
    def _make_chunk(self, text: str, index: int, metadata: Dict[str, Any] = None, section_index: int = None) -> Dict[str, Any]:
        """Create a chunk dictionary."""
        chunk = {
            'text': text.strip(),
            'index': index,
            'token_count': len(text) // 4,  # Rough estimate
        }
        
        if section_index is not None:
            chunk['section_index'] = section_index
        
        if metadata:
            chunk.update(metadata)
        
        return chunk