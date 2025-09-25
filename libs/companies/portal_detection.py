# libs/companies/portal_detection.py
import re
from typing import Optional, Tuple
from .models import PortalType, PortalConfig

class PortalDetectionService:
    """Service for detecting job portal types from HTML content"""

    # Patterns to find footprints of each ATS in the page's HTML
    PORTAL_PATTERNS = {
        PortalType.GREENHOUSE: [r'boards\.greenhouse\.io/|gh_src|greenhouse\.io/embed'],
        PortalType.LEVER: [r'jobs\.lever\.co/|lever\.co/careers'],
        PortalType.WORKDAY: [r'myworkdayjobs\.com/|wd1\.myworkdayjobs\.com'],
        # ... add more patterns for other ATS ...
    }

    def detect_portal(self, html_content: str, base_url: str) -> Tuple[PortalType, float]:
        """Detects the portal type and provides a confidence score."""
        html_lower = html_content.lower()
        best_match = PortalType.CUSTOM
        best_confidence = 0.1 # Default low confidence

        for portal_type, patterns in self.PORTAL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, html_lower):
                    # A simple match gives medium confidence. More advanced checks
                    # (like looking for specific script URLs) would increase this.
                    confidence = 0.8
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = portal_type

        return best_match, best_confidence