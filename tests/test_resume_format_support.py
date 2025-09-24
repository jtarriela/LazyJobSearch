"""Tests for resume parser file format support"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

from libs.resume.parser import ResumeParser, ResumeParseError


class TestResumeParserFormats:
    """Test file format parsing functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.parser = ResumeParser()
    
    def test_txt_parsing(self):
        """Test TXT file parsing"""
        # Create temporary TXT file
        test_content = """
        John Doe
        Software Engineer
        Email: john@example.com
        Phone: (555) 123-4567
        
        Experience:
        - Python development (3 years)
        - Django web applications
        
        Skills: Python, SQL, Django, AWS
        
        Education:
        Bachelor's degree in Computer Science
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = Path(f.name)
        
        try:
            result = self.parser.parse_file(temp_path)
            
            assert result.fulltext
            assert len(result.skills) > 0
            assert 'python' in [s.lower() for s in result.skills]
            assert result.contact_info.get('email') == 'john@example.com'
            assert result.contact_info.get('phone') == '555) 123-4567'
            assert result.education_level == 'bachelors'
            
        finally:
            temp_path.unlink()  # Clean up
    
    @patch('pdfplumber.open')
    def test_pdf_parsing_success(self, mock_pdfplumber):
        """Test successful PDF parsing with mocked pdfplumber"""
        # Mock PDF content
        mock_page = Mock()
        mock_page.extract_text.return_value = """
        Jane Smith
        Data Scientist
        jane@example.com
        
        Experience:
        - Machine learning projects (2 years)
        - Python and R programming
        
        Skills: Python, R, SQL, TensorFlow
        """
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.return_value.__enter__.return_value = mock_pdf
        
        # Create temporary PDF file (content doesn't matter since we're mocking)
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            result = self.parser.parse_file(temp_path)
            
            assert result.fulltext
            assert 'jane smith' in result.fulltext.lower()
            assert len(result.skills) > 0
            assert 'python' in [s.lower() for s in result.skills]
            assert result.contact_info.get('email') == 'jane@example.com'
            
        finally:
            temp_path.unlink()
    
    @patch('pdfplumber.open')
    def test_pdf_parsing_no_text(self, mock_pdfplumber):
        """Test PDF parsing when no text is extracted"""
        mock_page = Mock()
        mock_page.extract_text.return_value = None
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.return_value.__enter__.return_value = mock_pdf
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ResumeParseError, match="No text content extracted"):
                self.parser.parse_file(temp_path)
            
        finally:
            temp_path.unlink()
    
    @patch('pdfplumber.open')
    def test_pdf_parsing_error(self, mock_pdfplumber):
        """Test PDF parsing error handling"""
        mock_pdfplumber.side_effect = Exception("PDF parsing error")
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ResumeParseError, match="PDF extraction failed"):
                self.parser.parse_file(temp_path)
                
        finally:
            temp_path.unlink()
    
    @patch('docx.Document')
    def test_docx_parsing_success(self, mock_document):
        """Test successful DOCX parsing with mocked python-docx"""
        # Mock DOCX content
        mock_paragraph1 = Mock()
        mock_paragraph1.text = "John Smith"
        mock_paragraph2 = Mock()
        mock_paragraph2.text = "Software Engineer"
        mock_paragraph3 = Mock()
        mock_paragraph3.text = "Email: john.smith@example.com"
        mock_paragraph4 = Mock()
        mock_paragraph4.text = "Skills: Python, Java, React"
        
        mock_doc = Mock()
        mock_doc.paragraphs = [mock_paragraph1, mock_paragraph2, mock_paragraph3, mock_paragraph4]
        mock_doc.tables = []  # No tables
        mock_document.return_value = mock_doc
        
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            result = self.parser.parse_file(temp_path)
            
            assert result.fulltext
            assert 'john smith' in result.fulltext.lower()
            assert len(result.skills) > 0
            assert 'python' in [s.lower() for s in result.skills]
            assert result.contact_info.get('email') == 'john.smith@example.com'
            
        finally:
            temp_path.unlink()
    
    @patch('docx.Document')
    def test_docx_parsing_with_tables(self, mock_document):
        """Test DOCX parsing with tables"""
        # Mock paragraphs
        mock_paragraph = Mock()
        mock_paragraph.text = "Resume Header"
        
        # Mock table
        mock_cell1 = Mock()
        mock_cell1.text = "Skills"
        mock_cell2 = Mock()
        mock_cell2.text = "Python, SQL, Docker"
        
        mock_row = Mock()
        mock_row.cells = [mock_cell1, mock_cell2]
        
        mock_table = Mock()
        mock_table.rows = [mock_row]
        
        mock_doc = Mock()
        mock_doc.paragraphs = [mock_paragraph]
        mock_doc.tables = [mock_table]
        mock_document.return_value = mock_doc
        
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            result = self.parser.parse_file(temp_path)
            
            assert result.fulltext
            assert 'skills' in result.fulltext.lower()
            assert 'python' in result.fulltext.lower()
            assert len(result.skills) > 0
            
        finally:
            temp_path.unlink()
    
    @patch('docx.Document')
    def test_docx_parsing_error(self, mock_document):
        """Test DOCX parsing error handling"""
        mock_document.side_effect = Exception("DOCX parsing error")
        
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ResumeParseError, match="DOCX extraction failed"):
                self.parser.parse_file(temp_path)
                
        finally:
            temp_path.unlink()
    
    def test_unsupported_format(self):
        """Test unsupported file format"""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ResumeParseError, match="Unsupported file type"):
                self.parser.parse_file(temp_path)
                
        finally:
            temp_path.unlink()
    
    def test_file_not_found(self):
        """Test file not found error"""
        non_existent_path = Path("/tmp/nonexistent_file.pdf")
        
        with pytest.raises(ResumeParseError, match="File not found"):
            self.parser.parse_file(non_existent_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])