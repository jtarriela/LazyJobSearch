import re
from .models import PortalType

class PortalDetectionService:
    """Service for detecting job portal types from HTML content."""

    # These patterns find unique footprints of each ATS in a page's HTML source
    PORTAL_PATTERNS = {
        PortalType.GREENHOUSE: [r'boards\.greenhouse\.io/|gh_src|greenhouse\.io/embed'],
        PortalType.LEVER: [r'jobs\.lever\.co/|lever\.co/careers'],
        PortalType.WORKDAY: [r'myworkdayjobs\.com/|wd1\.myworkdayjobs\.com'],
        PortalType.ASHBY: [r'jobs\.ashbyhq\.com/'],
        PortalType.ICIMS: [r'careers\.icims\.com/'],
    }

    def detect_portal(self, html_content: str) -> PortalType:
        """Analyzes HTML to identify the underlying job platform."""
        html_lower = html_content.lower()
        for portal_type, patterns in self.PORTAL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, html_lower):
                    return portal_type # Return the first match
        return PortalType.CUSTOM # Default to custom if no pattern matches