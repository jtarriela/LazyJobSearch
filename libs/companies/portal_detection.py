"""
Portal Detection Service

This module implements automatic detection of job portals/ATS systems
by analyzing HTML content of careers pages.
"""
from __future__ import annotations
import re
import logging
from typing import Optional, Tuple, Dict, Any
from urllib.parse import urlparse, urljoin

from .models import PortalType, PortalConfig

logger = logging.getLogger(__name__)


class PortalDetectionService:
    """Service for detecting job portal types from HTML content"""
    
    # Portal detection patterns - each pattern maps to (portal_type, company_id_regex)
    PORTAL_PATTERNS = {
        PortalType.GREENHOUSE: [
            (r'boards\.greenhouse\.io/([a-zA-Z0-9_-]+)', r'boards\.greenhouse\.io/([a-zA-Z0-9_-]+)'),
            (r'greenhouse\.io/embed', r'greenhouse\.io/embed/job'),
            (r'api\.greenhouse\.io', None),
        ],
        PortalType.LEVER: [
            (r'jobs\.lever\.co/([a-zA-Z0-9_-]+)', r'jobs\.lever\.co/([a-zA-Z0-9_-]+)'),
            (r'lever\.co/careers', r'lever\.co/careers'),
        ],
        PortalType.WORKDAY: [
            (r'([a-zA-Z0-9_-]+)\.myworkdayjobs\.com', r'([a-zA-Z0-9_-]+)\.myworkdayjobs\.com'),
            (r'myworkdayjobs\.com/([a-zA-Z0-9_-]+)', r'myworkdayjobs\.com/([a-zA-Z0-9_-]+)'),
        ],
        PortalType.ASHBY: [
            (r'jobs\.ashbyhq\.com/([a-zA-Z0-9_-]+)', r'jobs\.ashbyhq\.com/([a-zA-Z0-9_-]+)'),
        ],
        PortalType.JOBVITE: [
            (r'jobs\.jobvite\.com/([a-zA-Z0-9_-]+)', r'jobs\.jobvite\.com/([a-zA-Z0-9_-]+)'),
            (r'jobvite\.com/careers', None),
        ],
        PortalType.SMARTRECRUITERS: [
            (r'jobs\.smartrecruiters\.com/([a-zA-Z0-9_-]+)', r'jobs\.smartrecruiters\.com/([a-zA-Z0-9_-]+)'),
        ],
        PortalType.WORKABLE: [
            (r'([a-zA-Z0-9_-]+)\.workable\.com', r'([a-zA-Z0-9_-]+)\.workable\.com'),
            (r'apply\.workable\.com/([a-zA-Z0-9_-]+)', r'apply\.workable\.com/([a-zA-Z0-9_-]+)'),
        ],
        PortalType.ICIMS: [
            (r'careers\.icims\.com/([a-zA-Z0-9_-]+)', r'careers\.icims\.com/([a-zA-Z0-9_-]+)'),
            (r'icims\.com/jobs', None),
        ],
        PortalType.BAMBOOHR: [
            (r'([a-zA-Z0-9_-]+)\.bamboohr\.com', r'([a-zA-Z0-9_-]+)\.bamboohr\.com'),
        ]
    }
    
    def __init__(self):
        """Initialize portal detection service"""
        pass
    
    def detect_portal(self, html_content: str, base_url: str) -> Tuple[Optional[PortalType], Optional[PortalConfig], float]:
        """
        Detect portal type from HTML content
        
        Args:
            html_content: Raw HTML content of the careers page
            base_url: Base URL of the page being analyzed
            
        Returns:
            Tuple of (portal_type, portal_config, confidence_score)
        """
        logger.info(f"Detecting portal type for {base_url}")
        
        best_match = None
        best_confidence = 0.0
        best_config = None
        
        for portal_type, patterns in self.PORTAL_PATTERNS.items():
            for pattern, company_id_pattern in patterns:
                # Search for pattern in HTML content
                match = re.search(pattern, html_content, re.IGNORECASE)
                if match:
                    confidence = self._calculate_confidence(portal_type, match, html_content)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = portal_type
                        
                        # Extract company ID if possible
                        company_id = None
                        if company_id_pattern and match.groups():
                            company_id = match.group(1)
                        
                        # Create portal config
                        best_config = PortalConfig(
                            company_id=company_id,
                            base_url=self._extract_base_url(match.group(0)) if match.groups() else None
                        )
                        
                        logger.info(f"Found {portal_type.value} portal with confidence {confidence:.2f}, company_id: {company_id}")
        
        if best_match is None:
            logger.info("No known portal detected, using custom")
            return PortalType.CUSTOM, PortalConfig(), 0.1
        
        return best_match, best_config, best_confidence
    
    def _calculate_confidence(self, portal_type: PortalType, match: re.Match, html_content: str) -> float:
        """Calculate confidence score for a portal match"""
        base_confidence = 0.7
        
        # Boost confidence based on multiple indicators
        confidence_boosts = {
            # Multiple occurrences of portal domain
            'multiple_occurrences': min(0.15, (html_content.lower().count(match.group(0).lower()) - 1) * 0.05),
            
            # Portal-specific keywords in HTML
            'portal_keywords': self._count_portal_keywords(portal_type, html_content) * 0.02,
            
            # Job-related content indicators
            'job_content': min(0.1, self._count_job_indicators(html_content) * 0.01),
        }
        
        total_confidence = base_confidence + sum(confidence_boosts.values())
        return min(0.95, total_confidence)  # Cap at 95%
    
    def _count_portal_keywords(self, portal_type: PortalType, html_content: str) -> int:
        """Count portal-specific keywords in HTML"""
        keywords = {
            PortalType.GREENHOUSE: ['greenhouse', 'apply now', 'job board'],
            PortalType.LEVER: ['lever', 'apply for this job'],
            PortalType.WORKDAY: ['workday', 'find jobs', 'search jobs'],
            PortalType.ASHBY: ['ashby', 'apply'],
            PortalType.JOBVITE: ['jobvite', 'apply online'],
        }
        
        if portal_type not in keywords:
            return 0
        
        content_lower = html_content.lower()
        return sum(1 for keyword in keywords[portal_type] if keyword in content_lower)
    
    def _count_job_indicators(self, html_content: str) -> int:
        """Count general job-related indicators"""
        indicators = ['apply', 'position', 'role', 'opening', 'career', 'job', 'employment', 'hiring']
        content_lower = html_content.lower()
        return sum(1 for indicator in indicators if indicator in content_lower)
    
    def _extract_base_url(self, matched_text: str) -> str:
        """Extract base URL from matched text"""
        if '://' not in matched_text:
            matched_text = f'https://{matched_text}'
        
        parsed = urlparse(matched_text)
        return f"{parsed.scheme}://{parsed.netloc}"
    
    def get_adapter_name(self, portal_type: PortalType) -> str:
        """Get the adapter name for a portal type"""
        adapter_mapping = {
            PortalType.GREENHOUSE: "greenhouse_v1",
            PortalType.LEVER: "lever_v1", 
            PortalType.WORKDAY: "workday_v1",
            PortalType.ASHBY: "ashby_v1",
            PortalType.JOBVITE: "jobvite_v1",
            PortalType.SMARTRECRUITERS: "smartrecruiters_v1",
            PortalType.WORKABLE: "workable_v1",
            PortalType.ICIMS: "icims_v1",
            PortalType.BAMBOOHR: "bamboohr_v1",
            PortalType.CUSTOM: "custom_v1",
        }
        return adapter_mapping.get(portal_type, "custom_v1")