"""Tests for PII redaction functionality"""
import pytest
from unittest.mock import Mock

from libs.resume.ingestion import ResumeIngestionService


class TestPIIRedaction:
    """Test PII redaction in logs and error messages"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_db_session = Mock()
        self.service = ResumeIngestionService(self.mock_db_session)
    
    def test_redact_email(self):
        """Test email redaction"""
        text = "Contact me at john.doe@example.com for questions."
        result = self.service._redact_pii(text)
        
        assert "[EMAIL]" in result
        assert "john.doe@example.com" not in result
        assert "Contact me at [EMAIL] for questions." == result
    
    def test_redact_multiple_emails(self):
        """Test multiple email redaction"""
        text = "Primary: john@company.com, Secondary: jane.smith@backup.org"
        result = self.service._redact_pii(text)
        
        assert result.count("[EMAIL]") == 2
        assert "john@company.com" not in result
        assert "jane.smith@backup.org" not in result
        assert "Primary: [EMAIL], Secondary: [EMAIL]" == result
    
    def test_redact_phone_standard(self):
        """Test standard phone number redaction"""
        text = "Call me at (555) 123-4567 anytime."
        result = self.service._redact_pii(text)
        
        assert "[PHONE]" in result
        assert "(555) 123-4567" not in result
        assert "Call me at [PHONE] anytime." == result
    
    def test_redact_phone_variations(self):
        """Test various phone number format redaction"""
        test_cases = [
            ("555-123-4567", "555-123-4567"),
            ("555.123.4567", "555.123.4567"), 
            ("555 123 4567", "555 123 4567"),
            ("(555)123-4567", "(555)123-4567"),
        ]
        
        for original, pattern in test_cases:
            text = f"Phone: {original}"
            result = self.service._redact_pii(text)
            assert "[PHONE]" in result
            assert original not in result
    
    def test_redact_ssn(self):
        """Test SSN redaction"""
        text = "SSN: 123-45-6789"
        result = self.service._redact_pii(text)
        
        assert "[SSN]" in result
        assert "123-45-6789" not in result
        assert "SSN: [SSN]" == result
    
    def test_redact_address(self):
        """Test street address redaction"""
        test_cases = [
            "123 Main Street",
            "456 Oak Ave",
            "789 First Rd",
            "321 Elm Drive",
            "654 Park Lane",
            "987 Broadway Blvd"
        ]
        
        for address in test_cases:
            text = f"Address: {address}, City"
            result = self.service._redact_pii(text)
            assert "[ADDRESS]" in result
            assert address not in result
    
    def test_redact_mixed_pii(self):
        """Test redaction of multiple PII types"""
        text = """
        John Doe
        Email: john.doe@company.com
        Phone: (555) 123-4567
        Address: 123 Main Street, Anytown
        SSN: 123-45-6789
        """
        
        result = self.service._redact_pii(text)
        
        assert "[EMAIL]" in result
        assert "[PHONE]" in result  
        assert "[ADDRESS]" in result
        assert "[SSN]" in result
        
        # Verify original PII is not present
        assert "john.doe@company.com" not in result
        assert "(555) 123-4567" not in result
        assert "123 Main Street" not in result
        assert "123-45-6789" not in result
    
    def test_redact_no_pii(self):
        """Test text with no PII remains unchanged"""
        text = "This is just regular text with no sensitive information."
        result = self.service._redact_pii(text)
        
        assert result == text
    
    def test_redact_empty_text(self):
        """Test redaction handles empty/None text"""
        assert self.service._redact_pii("") == ""
        assert self.service._redact_pii(None) is None
    
    def test_redact_case_insensitive(self):
        """Test redaction is case insensitive for formats"""
        text = "Email: JOHN.DOE@COMPANY.COM and phone: (555) 123-4567"
        result = self.service._redact_pii(text)
        
        assert "[EMAIL]" in result
        assert "[PHONE]" in result
        assert "JOHN.DOE@COMPANY.COM" not in result
    
    def test_redact_preserves_structure(self):
        """Test redaction preserves text structure"""
        text = """Line 1: john@example.com
Line 2: Some other content
Line 3: Phone (555) 123-4567"""
        
        result = self.service._redact_pii(text)
        
        # Should preserve line structure
        lines = result.split('\n')
        assert len(lines) == 3
        assert "[EMAIL]" in lines[0]
        assert "Some other content" in lines[1]
        assert "[PHONE]" in lines[2]
    
    def test_redact_partial_matches(self):
        """Test redaction doesn't over-match partial patterns"""
        # These should NOT be redacted as they're not complete PII
        text = "Visit example.com or call extension 1234"
        result = self.service._redact_pii(text)
        
        # Should remain unchanged as these aren't complete email/phone patterns
        assert result == text
        assert "[EMAIL]" not in result
        assert "[PHONE]" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])