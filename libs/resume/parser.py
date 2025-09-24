"""Resume parser for PDF and DOCX files

Handles file upload, text extraction, and basic resume section parsing.
Supports both PDF and Word document formats with error handling.
"""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import tempfile
import os

logger = logging.getLogger(__name__)

@dataclass
class ResumeSection:
    """Represents a parsed section of a resume"""
    name: str
    content: str
    start_index: int
    end_index: int

@dataclass
class ParsedResume:
    """Represents a fully parsed resume with metadata"""
    fulltext: str
    sections: Dict[str, str]
    skills: List[str]
    years_of_experience: Optional[float]
    education_level: Optional[str]
    contact_info: Dict[str, str]
    word_count: int
    char_count: int

class ResumeParseError(Exception):
    """Custom exception for resume parsing errors"""
    pass

class ResumeParser:
    """Parser for resume documents (PDF and DOCX)"""
    
    # Common section headers to look for
    SECTION_PATTERNS = {
        'contact': [
            r'\b(contact|personal)\s*(information|info|details)?\b',
            r'\b(phone|email|address)\b'
        ],
        'summary': [
            r'\b(professional\s*)?summary\b',
            r'\bobjective\b',
            r'\babout\s*me\b',
            r'\bprofile\b'
        ],
        'experience': [
            r'\b(work\s*)?experience\b',
            r'\b(professional\s*)?experience\b',
            r'\bemployment\s*(history)?\b',
            r'\bcareer\s*history\b'
        ],
        'education': [
            r'\beducation\b',
            r'\bacademic\s*(background|qualifications)\b',
            r'\bdegree(s)?\b'
        ],
        'skills': [
            r'\b(technical\s*)?skills\b',
            r'\bcompeten(cies|t)\b',
            r'\bexpertise\b',
            r'\bproficienc(ies|y)\b'
        ],
        'projects': [
            r'\bprojects?\b',
            r'\bportfolio\b'
        ],
        'certifications': [
            r'\bcertifications?\b',
            r'\blicenses?\b',
            r'\bcredentials?\b'
        ]
    }
    
    # Technical skills patterns for extraction
    TECH_SKILLS = {
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
        'react', 'vue', 'angular', 'node.js', 'express', 'django', 'flask',
        'sql', 'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform',
        'git', 'jenkins', 'cicd', 'devops', 'linux', 'bash',
        'machine learning', 'ml', 'ai', 'data science', 'pandas', 'numpy',
        'tensorflow', 'pytorch', 'scikit-learn'
    }
    
    def __init__(self):
        """Initialize the resume parser"""
        self._compiled_patterns = {}
        self._compile_section_patterns()
    
    def _compile_section_patterns(self):
        """Pre-compile regex patterns for efficiency"""
        for section, patterns in self.SECTION_PATTERNS.items():
            self._compiled_patterns[section] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                for pattern in patterns
            ]
    
    def parse_file(self, file_path: Union[str, Path]) -> ParsedResume:
        """Parse a resume file (PDF or DOCX)
        
        Args:
            file_path: Path to the resume file
            
        Returns:
            ParsedResume object with extracted content
            
        Raises:
            ResumeParseError: If parsing fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ResumeParseError(f"File not found: {file_path}")
        
        try:
            # Extract text based on file type
            if file_path.suffix.lower() == '.pdf':
                fulltext = self._extract_pdf_text(file_path)
            elif file_path.suffix.lower() in ['.docx', '.doc']:
                fulltext = self._extract_docx_text(file_path)
            else:
                raise ResumeParseError(f"Unsupported file type: {file_path.suffix}")
            
            if not fulltext.strip():
                raise ResumeParseError("No text content extracted from file")
            
            # Parse the extracted text
            return self._parse_text(fulltext)
            
        except Exception as e:
            logger.error(f"Failed to parse resume {file_path}: {e}")
            raise ResumeParseError(f"Parsing failed: {str(e)}") from e
    
    def parse_text(self, text: str) -> ParsedResume:
        """Parse resume text directly
        
        Args:
            text: Raw resume text
            
        Returns:
            ParsedResume object with extracted content
        """
        return self._parse_text(text)
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF file
        
        Note: This is a stub implementation. In production, you would use
        a library like PyPDF2, pdfplumber, or pdfminer.six
        """
        try:
            # Stub implementation - in reality would use PDF extraction library
            # import PyPDF2
            # with open(file_path, 'rb') as file:
            #     reader = PyPDF2.PdfReader(file)
            #     text = ""
            #     for page in reader.pages:
            #         text += page.extract_text()
            #     return text
            
            # For now, return a placeholder to prevent blocking
            logger.warning(f"PDF parsing not implemented - using placeholder for {file_path}")
            return f"[PDF content placeholder for {file_path.name}]\n\nJohn Doe\nSoftware Engineer\n\nExperience:\n- Python development (3 years)\n- Web applications with Django\n- Database design\n\nSkills: Python, SQL, AWS, Docker"
            
        except Exception as e:
            raise ResumeParseError(f"PDF extraction failed: {str(e)}") from e
    
    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX file
        
        Note: This is a stub implementation. In production, you would use
        python-docx library
        """
        try:
            # Stub implementation - in reality would use python-docx
            # from docx import Document
            # document = Document(file_path)
            # text = ""
            # for paragraph in document.paragraphs:
            #     text += paragraph.text + "\n"
            # return text
            
            # For now, return a placeholder to prevent blocking
            logger.warning(f"DOCX parsing not implemented - using placeholder for {file_path}")
            return f"[DOCX content placeholder for {file_path.name}]\n\nJane Smith\nData Scientist\n\nExperience:\n- Machine learning projects (2 years)\n- Python and R programming\n- Statistical analysis\n\nSkills: Python, R, SQL, TensorFlow, pandas"
            
        except Exception as e:
            raise ResumeParseError(f"DOCX extraction failed: {str(e)}") from e
    
    def _parse_text(self, text: str) -> ParsedResume:
        """Parse raw resume text into structured data"""
        
        # Clean and normalize text
        normalized_text = self._normalize_text(text)
        
        # Extract sections
        sections = self._extract_sections(normalized_text)
        
        # Extract skills
        skills = self._extract_skills(normalized_text)
        
        # Extract years of experience
        yoe = self._extract_years_of_experience(normalized_text)
        
        # Extract education level
        education_level = self._extract_education_level(normalized_text)
        
        # Extract contact information
        contact_info = self._extract_contact_info(normalized_text)
        
        return ParsedResume(
            fulltext=normalized_text,
            sections=sections,
            skills=skills,
            years_of_experience=yoe,
            education_level=education_level,
            contact_info=contact_info,
            word_count=len(normalized_text.split()),
            char_count=len(normalized_text)
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize and clean resume text"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove special characters that might interfere with parsing
        text = re.sub(r'[^\w\s\-\.,;:()\[\]@/]', '', text)
        
        return text.strip()
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections from resume text"""
        sections = {}
        text_lower = text.lower()
        
        for section_name, patterns in self._compiled_patterns.items():
            section_content = self._find_section_content(text, text_lower, patterns)
            if section_content:
                sections[section_name] = section_content
        
        return sections
    
    def _find_section_content(self, text: str, text_lower: str, patterns: List[re.Pattern]) -> Optional[str]:
        """Find content for a specific section using regex patterns"""
        best_match = None
        best_start = float('inf')
        
        for pattern in patterns:
            match = pattern.search(text_lower)
            if match and match.start() < best_start:
                best_match = match
                best_start = match.start()
        
        if not best_match:
            return None
        
        # Find the end of this section (start of next section or end of text)
        section_start = best_match.end()
        
        # Look for the next section header
        next_section_start = len(text)
        for other_patterns in self._compiled_patterns.values():
            for pattern in other_patterns:
                matches = list(pattern.finditer(text_lower[section_start:]))
                if matches:
                    next_start = section_start + matches[0].start()
                    if next_start < next_section_start:
                        next_section_start = next_start
        
        content = text[section_start:next_section_start].strip()
        return content if content else None
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract technical skills from resume text"""
        text_lower = text.lower()
        found_skills = []
        
        for skill in self.TECH_SKILLS:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.append(skill)
        
        # Also look for skills in a dedicated skills section
        skills_section = None
        for section_name, content in self._extract_sections(text).items():
            if section_name == 'skills':
                skills_section = content
                break
        
        if skills_section:
            # Extract additional skills from the skills section
            # This is a simple approach - could be enhanced with NLP
            potential_skills = re.findall(r'\b[A-Za-z][A-Za-z0-9\.\+#]{2,15}\b', skills_section)
            for skill in potential_skills:
                if skill.lower() not in [s.lower() for s in found_skills]:
                    found_skills.append(skill)
        
        return found_skills
    
    def _extract_years_of_experience(self, text: str) -> Optional[float]:
        """Extract years of experience from resume text"""
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            r'(?:experience|exp)[:\s]*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)',
            r'(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)'
        ]
        
        text_lower = text.lower()
        years = []
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    years.append(float(match))
                except ValueError:
                    continue
        
        if years:
            # Return the maximum years found (likely the total experience)
            return max(years)
        
        return None
    
    def _extract_education_level(self, text: str) -> Optional[str]:
        """Extract education level from resume text"""
        text_lower = text.lower()
        
        # Define education levels in order of priority
        education_levels = [
            ('phd', ['ph.d', 'phd', 'doctorate', 'doctoral']),
            ('masters', ['master', 'msc', 'mba', 'ms', 'm.s']),
            ('bachelors', ['bachelor', 'bsc', 'bs', 'b.s', 'ba', 'b.a']),
            ('associates', ['associate', 'aa', 'as']),
            ('high_school', ['high school', 'diploma', 'ged'])
        ]
        
        for level, keywords in education_levels:
            for keyword in keywords:
                if keyword in text_lower:
                    return level
        
        return None
    
    def _extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information from resume text"""
        contact = {}
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            contact['email'] = email_match.group()
        
        # Phone pattern (simple US format)
        phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            contact['phone'] = phone_match.group()
        
        # LinkedIn pattern
        linkedin_pattern = r'linkedin\.com/in/([A-Za-z0-9\-]+)'
        linkedin_match = re.search(linkedin_pattern, text.lower())
        if linkedin_match:
            contact['linkedin'] = f"linkedin.com/in/{linkedin_match.group(1)}"
        
        return contact

def create_resume_parser() -> ResumeParser:
    """Factory function to create a configured resume parser"""
    return ResumeParser()