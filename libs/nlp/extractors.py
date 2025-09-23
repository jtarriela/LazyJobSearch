"""Text extractors for skills, experience, and other structured data."""

import re
from typing import List, Dict, Set, Optional
from dataclasses import dataclass


@dataclass
class SkillMatch:
    """Represents a matched skill with context."""
    skill: str
    category: str
    confidence: float
    context: str


class SkillExtractor:
    """Extract technical skills and competencies from text.
    
    Uses a combination of curated skill lists and pattern matching
    to identify relevant skills in resumes and job descriptions.
    """
    
    def __init__(self):
        """Initialize with curated skill dictionaries."""
        self.programming_languages = {
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
            'scala', 'kotlin', 'swift', 'php', 'ruby', 'r', 'matlab', 'sql',
            'bash', 'powershell', 'perl', 'html', 'css', 'sass', 'less'
        }
        
        self.frameworks = {
            'react', 'angular', 'vue', 'svelte', 'django', 'flask', 'fastapi',
            'spring', 'express', 'nodejs', 'nextjs', 'nuxtjs', 'gatsby',
            'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
            'react native', 'flutter', 'xamarin', 'ionic'
        }
        
        self.databases = {
            'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch',
            'cassandra', 'dynamodb', 'sqlite', 'oracle', 'sql server',
            'neo4j', 'influxdb', 'clickhouse', 'snowflake', 'bigquery'
        }
        
        self.cloud_platforms = {
            'aws', 'azure', 'gcp', 'google cloud', 'heroku', 'vercel',
            'netlify', 'digitalocean', 'linode', 'cloudflare'
        }
        
        self.tools = {
            'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins',
            'gitlab ci', 'github actions', 'circleci', 'jira', 'confluence',
            'git', 'svn', 'mercurial', 'postman', 'insomnia', 'datadog',
            'prometheus', 'grafana', 'elk stack', 'splunk'
        }
        
        # Combine all skill categories
        self.all_skills = {
            **{skill: 'programming' for skill in self.programming_languages},
            **{skill: 'framework' for skill in self.frameworks},
            **{skill: 'database' for skill in self.databases},
            **{skill: 'cloud' for skill in self.cloud_platforms},
            **{skill: 'tool' for skill in self.tools}
        }
    
    def extract_skills(self, text: str, min_confidence: float = 0.7) -> List[SkillMatch]:
        """Extract skills from text with confidence scoring."""
        text_lower = text.lower()
        matches = []
        
        for skill, category in self.all_skills.items():
            confidence, context = self._match_skill(skill, text_lower)
            
            if confidence >= min_confidence:
                matches.append(SkillMatch(
                    skill=skill,
                    category=category,
                    confidence=confidence,
                    context=context
                ))
        
        # Sort by confidence and remove duplicates
        matches.sort(key=lambda x: x.confidence, reverse=True)
        seen_skills = set()
        unique_matches = []
        
        for match in matches:
            if match.skill not in seen_skills:
                unique_matches.append(match)
                seen_skills.add(match.skill)
        
        return unique_matches
    
    def _match_skill(self, skill: str, text: str) -> tuple[float, str]:
        """Match a skill in text and return confidence score and context."""
        # Exact word boundary match gets highest confidence
        word_pattern = r'\b' + re.escape(skill) + r'\b'
        exact_matches = list(re.finditer(word_pattern, text, re.IGNORECASE))
        
        if exact_matches:
            match = exact_matches[0]
            context = self._extract_context(text, match.start(), match.end())
            
            # Higher confidence for skills in relevant contexts
            confidence = 0.9
            context_lower = context.lower()
            
            # Boost confidence for skills in technical contexts
            tech_indicators = ['experience', 'proficient', 'expert', 'years', 'skilled']
            if any(indicator in context_lower for indicator in tech_indicators):
                confidence = min(1.0, confidence + 0.1)
            
            return confidence, context
        
        # Partial matches get lower confidence
        if skill in text:
            # Find the position for context
            pos = text.find(skill)
            context = self._extract_context(text, pos, pos + len(skill))
            return 0.7, context
        
        return 0.0, ""
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Extract surrounding context for a match."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()
    
    def get_skill_summary(self, matches: List[SkillMatch]) -> Dict[str, List[str]]:
        """Group skills by category for summary."""
        summary = {}
        for match in matches:
            if match.category not in summary:
                summary[match.category] = []
            summary[match.category].append(match.skill)
        return summary


class YearsOfExperienceExtractor:
    """Extract years of experience from resume text.
    
    Handles various formats like "5 years", "2+ years", "3-5 years", etc.
    and applies education bonus rules as per architecture specs.
    """
    
    def __init__(self):
        """Initialize with experience patterns."""
        self.experience_patterns = [
            # Direct years mentions
            r'(\d+)\s*\+?\s*years?\s+(?:of\s+)?(?:experience|exp)',
            r'(\d+)\s*\+?\s*yrs?\s+(?:of\s+)?(?:experience|exp)',
            
            # Range patterns
            r'(\d+)\s*[-–]\s*(\d+)\s*years?\s+(?:of\s+)?(?:experience|exp)',
            r'(\d+)\s*[-–]\s*(\d+)\s*yrs?\s+(?:of\s+)?(?:experience|exp)',
            
            # Over/more than patterns
            r'(?:over|more than|>\s*)(\d+)\s*years?\s+(?:of\s+)?(?:experience|exp)',
            r'(?:over|more than|>\s*)(\d+)\s*yrs?\s+(?:of\s+)?(?:experience|exp)',
        ]
        
        self.education_levels = {
            'phd': 4,
            'doctorate': 4,
            'doctoral': 4,
            'ph.d': 4,
            'master': 2,
            'masters': 2,
            "master's": 2,
            'mba': 2,
            'ms': 2,
            'ma': 2,
            'bachelor': 0,
            'bachelors': 0,
            "bachelor's": 0,
            'bs': 0,
            'ba': 0,
            'undergraduate': 0,
        }
    
    def extract_experience(self, text: str) -> Dict[str, float]:
        """Extract years of experience with education bonus.
        
        Returns:
            Dict with 'raw_experience', 'education_bonus', 'total_experience'
        """
        text_lower = text.lower()
        
        # Extract raw experience years
        raw_years = self._extract_raw_years(text_lower)
        
        # Extract education level
        education_bonus = self._calculate_education_bonus(text_lower)
        
        # Calculate total adjusted experience
        total_years = raw_years + education_bonus
        
        return {
            'raw_experience': raw_years,
            'education_bonus': education_bonus,
            'total_experience': total_years
        }
    
    def _extract_raw_years(self, text: str) -> float:
        """Extract raw years of experience from text."""
        years = []
        
        for pattern in self.experience_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                groups = match.groups()
                
                if len(groups) == 1:
                    # Single number
                    years.append(float(groups[0]))
                elif len(groups) == 2:
                    # Range - take the average
                    start, end = float(groups[0]), float(groups[1])
                    years.append((start + end) / 2)
        
        # Also look for employment date ranges
        date_years = self._extract_from_employment_dates(text)
        years.extend(date_years)
        
        # Return the maximum found (most comprehensive experience claim)
        return max(years) if years else 0.0
    
    def _extract_from_employment_dates(self, text: str) -> List[float]:
        """Extract experience from employment date ranges."""
        # Pattern for date ranges like "2019-2023", "Jan 2020 - Present"
        date_patterns = [
            r'(\d{4})\s*[-–]\s*(\d{4})',  # 2019-2023
            r'(\d{4})\s*[-–]\s*(?:present|current|now)',  # 2019-Present
        ]
        
        years = []
        current_year = 2024  # Update this as needed
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                groups = match.groups()
                start_year = int(groups[0])
                
                if len(groups) == 2 and groups[1].isdigit():
                    end_year = int(groups[1])
                else:
                    end_year = current_year  # Present/Current
                
                experience = max(0, end_year - start_year)
                years.append(experience)
        
        return years
    
    def _calculate_education_bonus(self, text: str) -> float:
        """Calculate education bonus based on highest degree."""
        max_bonus = 0.0
        
        for degree, bonus in self.education_levels.items():
            if degree in text:
                max_bonus = max(max_bonus, bonus)
        
        return max_bonus
    
    def extract_education_details(self, text: str) -> Dict[str, any]:
        """Extract detailed education information."""
        text_lower = text.lower()
        
        # Find highest degree
        highest_degree = None
        max_bonus = 0.0
        
        for degree, bonus in self.education_levels.items():
            if degree in text_lower:
                if bonus > max_bonus:
                    max_bonus = bonus
                    highest_degree = degree
        
        # Extract graduation years
        grad_years = []
        grad_pattern = r'(?:graduated|graduation|class of|\')\s*(\d{4})'
        matches = re.finditer(grad_pattern, text_lower)
        
        for match in matches:
            grad_years.append(int(match.group(1)))
        
        return {
            'highest_degree': highest_degree,
            'education_bonus': max_bonus,
            'graduation_years': sorted(grad_years)
        }