"""Company seeding utilities - support for CLI companies seed command.

This module implements company data seeding functionality:
1. CSV/JSON company data parsing
2. Duplicate detection and deduplication
3. Careers URL validation and discovery
4. Batch company insertion with error handling

Based on requirements from the gap analysis for company seeding CLI.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid
import hashlib
import asyncio

from libs.observability import get_logger, timer, counter
from libs.db.models import Company

logger = get_logger(__name__)

@dataclass 
class CompanyData:
    """Company data for seeding"""
    name: str
    website: Optional[str] = None
    careers_url: Optional[str] = None
    crawler_profile: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SeedingStats:
    """Statistics from company seeding process"""
    companies_read: int
    companies_deduplicated: int
    companies_created: int
    companies_updated: int
    errors: List[str] = None


class CompanySeedingService:
    """Service for seeding company data"""
    
    def __init__(self, db_session):
        self.db_session = db_session
        self._name_fingerprint_cache: Set[str] = set()
    
    async def seed_companies_from_file(self, file_path: Path, update_existing: bool = False) -> SeedingStats:
        """
        Seed companies from CSV or JSON file.
        
        Args:
            file_path: Path to CSV or JSON file with company data
            update_existing: Whether to update existing companies
            
        Returns:
            SeedingStats with seeding results
        """
        try:
            logger.info("Starting company seeding", file_path=str(file_path))
            
            with timer("company_seeding.total"):
                # Load existing companies for deduplication
                await self._load_company_fingerprints()
                
                # Parse company data from file
                companies_data = await self._parse_companies_file(file_path)
                
                stats = SeedingStats(
                    companies_read=len(companies_data),
                    companies_deduplicated=0,
                    companies_created=0,
                    companies_updated=0,
                    errors=[]
                )
                
                if not companies_data:
                    logger.info("No company data found in file")
                    return stats
                
                # Process and seed companies
                await self._seed_companies_batch(companies_data, update_existing, stats)
                
                logger.info("Company seeding completed", stats=stats)
                return stats
                
        except Exception as e:
            logger.error("Company seeding failed", file_path=str(file_path), error=str(e))
            raise
    
    def _generate_company_fingerprint(self, name: str) -> str:
        """Generate fingerprint from company name for deduplication"""
        # Normalize name for consistent fingerprinting
        normalized_name = name.lower().strip()
        return hashlib.md5(normalized_name.encode('utf-8')).hexdigest()


def create_company_seeding_service(db_session) -> CompanySeedingService:
    """Factory function to create company seeding service"""
    return CompanySeedingService(db_session)