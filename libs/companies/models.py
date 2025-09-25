from __future__ import annotations
from pydantic import BaseModel, HttpUrl, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import re


class PortalType(str, Enum):
    """Supported portal types"""
    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    WORKDAY = "workday"
    ASHBY = "ashby"
    CUSTOM = "custom"


class Confidence(BaseModel):
    """Confidence scores for automated detection"""
    careers_url: float = Field(..., ge=0.0, le=1.0)
    portal_detection: float = Field(..., ge=0.0, le=1.0)


class PortalConfig(BaseModel):
    """Portal-specific configuration"""
    company_id: Optional[str] = None
    job_path_template: Optional[str] = None
    base_url: Optional[str] = None


class Portal(BaseModel):
    """Portal information and configuration"""
    type: PortalType
    adapter: Optional[str] = None
    portal_config: Optional[PortalConfig] = None


class Careers(BaseModel):
    """Careers page information"""
    primary_url: HttpUrl
    discovered_alternatives: List[HttpUrl] = Field(default_factory=list)


class CrawlerConfig(BaseModel):
    """Crawler-specific configuration"""
    enabled: bool = True
    start_urls: List[HttpUrl] = Field(default_factory=list)
    heuristics: Dict[str, Any] = Field(default_factory=dict)


class CompanySeed(BaseModel):
    """Complete company seed configuration"""
    id: str = Field(..., description="Company slug identifier")
    name: str = Field(..., description="Company display name")
    domain: str = Field(..., description="Company primary domain")
    careers: Careers
    portal: Portal
    crawler: CrawlerConfig = Field(default_factory=CrawlerConfig)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None

    def model_post_init(self, __context):
        """Set default metadata after initialization"""
        if not self.metadata.get('created_at'):
            self.metadata['created_at'] = datetime.utcnow().isoformat()
        if not self.crawler.start_urls:
            self.crawler.start_urls = [self.careers.primary_url]


def generate_slug(company_name: str) -> str:
    """Generate a slug from company name"""
    slug = re.sub(r'[^\w\s-]', '', company_name.lower())
    slug = re.sub(r'[-\s]+', '-', slug)
    slug = re.sub(r'-(inc|corp|llc|ltd|company)$', '', slug)
    return slug.strip('-')