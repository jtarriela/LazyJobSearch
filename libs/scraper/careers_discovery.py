"""
Careers Page Discovery Service

This module provides automatic discovery of careers pages from base company domains.
Given a company website (e.g., company.com), it probes common paths and parses 
internal links to locate careers portals.

Features:
- Probes common career page paths (/careers, /jobs, /join-us, etc.)
- Parses homepage for career-related links
- Respects robots.txt before crawling
- Scores candidates based on URL patterns and content analysis
- Returns the most likely careers URL
"""
from __future__ import annotations
import logging
import re
import time
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class CareersDiscoveryService:
    """Service for automatically discovering careers pages from company domains"""
    
    # Common careers page paths to probe
    COMMON_CAREER_PATHS = [
        '/careers',
        '/jobs',
        '/job-opportunities', 
        '/join-us',
        '/work-with-us',
        '/opportunities',
        '/employment',
        '/hiring',
        '/openings',
        '/positions',
        '/talent',
        '/recruitment',
        '/career',
        '/job',
        '/join',
        '/work'
    ]
    
    # Keywords that indicate career-related content
    CAREER_KEYWORDS = [
        'career', 'careers', 'job', 'jobs', 'hiring', 'employment', 
        'opportunities', 'join', 'work', 'talent', 'recruitment',
        'positions', 'openings', 'apply'
    ]
    
    def __init__(self, session: Optional[requests.Session] = None, timeout: int = 10):
        """Initialize the discovery service
        
        Args:
            session: Optional requests session for connection pooling
            timeout: Request timeout in seconds
        """
        self.session = session or requests.Session()
        self.timeout = timeout
        
        # Set user agent to appear more like a regular browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def discover_careers_url(self, base_url: str) -> Optional[str]:
        """Discover the careers page URL for a given company website
        
        Args:
            base_url: Base company URL (e.g., 'https://company.com' or 'company.com')
            
        Returns:
            The discovered careers page URL, or None if not found
        """
        logger.info(f"Starting careers page discovery for {base_url}")
        
        # Normalize the base URL
        if not base_url.startswith(('http://', 'https://')):
            base_url = f'https://{base_url}'
        
        try:
            # Check robots.txt first
            if not self._check_robots_txt(base_url):
                logger.warning(f"Robots.txt disallows crawling for {base_url}")
                return None
            
            # Step 1: Probe common career paths
            direct_candidates = self._probe_common_paths(base_url)
            if direct_candidates:
                logger.info(f"Found direct path candidates: {direct_candidates}")
                return self._select_best_candidate(direct_candidates)
            
            # Step 2: Parse homepage for career links
            homepage_candidates = self._parse_homepage_links(base_url)
            if homepage_candidates:
                logger.info(f"Found homepage link candidates: {homepage_candidates}")
                return self._select_best_candidate(homepage_candidates)
            
            logger.warning(f"No careers page found for {base_url}")
            return None
            
        except Exception as e:
            logger.error(f"Error discovering careers page for {base_url}: {e}")
            return None
    
    def _check_robots_txt(self, base_url: str) -> bool:
        """Check if robots.txt allows crawling
        
        Args:
            base_url: Base URL to check
            
        Returns:
            True if crawling is allowed, False otherwise
        """
        try:
            rp = RobotFileParser()
            rp.set_url(urljoin(base_url, '/robots.txt'))
            rp.read()
            
            # Check if our user agent can fetch the homepage
            return rp.can_fetch(self.session.headers.get('User-Agent', '*'), base_url)
        except Exception as e:
            logger.debug(f"Error checking robots.txt for {base_url}: {e}")
            # If we can't check robots.txt, assume it's okay to proceed
            return True
    
    def _probe_common_paths(self, base_url: str) -> List[Tuple[str, float]]:
        """Probe common career page paths
        
        Args:
            base_url: Base URL to probe
            
        Returns:
            List of (url, score) tuples for found pages
        """
        candidates = []
        
        for path in self.COMMON_CAREER_PATHS:
            try:
                candidate_url = urljoin(base_url, path)
                logger.debug(f"Probing {candidate_url}")
                
                response = self._safe_head_request(candidate_url)
                if response and response.status_code == 200:
                    score = self._score_career_url(candidate_url, response)
                    candidates.append((candidate_url, score))
                    logger.debug(f"Found candidate: {candidate_url} (score: {score})")
                
                # Rate limiting to be respectful
                time.sleep(0.5)
                
            except Exception as e:
                logger.debug(f"Error probing {candidate_url}: {e}")
                continue
        
        return candidates
    
    def _parse_homepage_links(self, base_url: str) -> List[Tuple[str, float]]:
        """Parse homepage for career-related links
        
        Args:
            base_url: Base URL to parse
            
        Returns:
            List of (url, score) tuples for found links
        """
        try:
            response = self.session.get(base_url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            candidates = []
            
            # Find all links
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if not href:
                    continue
                
                # Make relative URLs absolute
                absolute_url = urljoin(base_url, href)
                
                # Skip external links (different domain)
                if not self._is_same_domain(base_url, absolute_url):
                    continue
                
                # Check if the link text or URL contains career keywords
                link_text = link.get_text(strip=True).lower()
                url_path = urlparse(absolute_url).path.lower()
                
                score = self._score_career_link(link_text, url_path, href)
                if score > 0:
                    candidates.append((absolute_url, score))
                    logger.debug(f"Found homepage candidate: {absolute_url} (score: {score})")
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error parsing homepage {base_url}: {e}")
            return []
    
    def _safe_head_request(self, url: str) -> Optional[requests.Response]:
        """Make a safe HEAD request to check if URL exists
        
        Args:
            url: URL to check
            
        Returns:
            Response object if successful, None otherwise
        """
        try:
            return self.session.head(url, timeout=self.timeout, allow_redirects=True)
        except Exception as e:
            logger.debug(f"HEAD request failed for {url}: {e}")
            return None
    
    def _score_career_url(self, url: str, response: requests.Response) -> float:
        """Score a career URL candidate based on URL pattern and response
        
        Args:
            url: The candidate URL
            response: HTTP response from the URL
            
        Returns:
            Score from 0.0 to 1.0 (higher is better)
        """
        score = 0.0
        url_lower = url.lower()
        
        # Score based on URL path
        if '/careers' in url_lower:
            score += 0.9
        elif '/jobs' in url_lower:
            score += 0.8
        elif any(keyword in url_lower for keyword in ['/hiring', '/employment', '/opportunities']):
            score += 0.7
        elif any(keyword in url_lower for keyword in ['/join', '/work', '/talent']):
            score += 0.6
        
        # Bonus for exact matches
        if url_lower.endswith('/careers') or url_lower.endswith('/jobs'):
            score += 0.1
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' in content_type:
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_career_link(self, link_text: str, url_path: str, href: str) -> float:
        """Score a career link found on homepage
        
        Args:
            link_text: Text content of the link
            url_path: URL path component
            href: Original href attribute
            
        Returns:
            Score from 0.0 to 1.0 (higher is better)
        """
        score = 0.0
        
        # Score based on link text
        for keyword in self.CAREER_KEYWORDS:
            if keyword in link_text:
                if keyword in ['careers', 'jobs']:
                    score += 0.3
                elif keyword in ['career', 'job']:
                    score += 0.25
                else:
                    score += 0.2
                break
        
        # Score based on URL path
        for keyword in self.CAREER_KEYWORDS:
            if keyword in url_path:
                if keyword in ['careers', 'jobs']:
                    score += 0.4
                elif keyword in ['career', 'job']:
                    score += 0.35
                else:
                    score += 0.3
                break
        
        # Penalty for very long URLs (likely not main careers page)
        if len(href) > 100:
            score *= 0.8
        
        # Only return positive scores for likely candidates
        return score if score > 0.2 else 0.0
    
    def _select_best_candidate(self, candidates: List[Tuple[str, float]]) -> Optional[str]:
        """Select the best candidate from a list of scored URLs
        
        Args:
            candidates: List of (url, score) tuples
            
        Returns:
            The URL with the highest score, or None if no good candidates
        """
        if not candidates:
            return None
        
        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        best_url, best_score = candidates[0]
        
        # Only return if score is above threshold
        if best_score >= 0.5:
            logger.info(f"Selected best candidate: {best_url} (score: {best_score})")
            return best_url
        
        logger.warning(f"No candidate above threshold (best: {best_url}, score: {best_score})")
        return None
    
    def _is_same_domain(self, base_url: str, candidate_url: str) -> bool:
        """Check if two URLs are from the same domain
        
        Args:
            base_url: Base URL to compare against
            candidate_url: Candidate URL to check
            
        Returns:
            True if same domain, False otherwise
        """
        try:
            base_domain = urlparse(base_url).netloc.lower()
            candidate_domain = urlparse(candidate_url).netloc.lower()
            
            # Handle www. prefixes
            base_domain = base_domain.replace('www.', '')
            candidate_domain = candidate_domain.replace('www.', '')
            
            return base_domain == candidate_domain
        except Exception:
            return False