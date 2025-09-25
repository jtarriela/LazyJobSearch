# libs/companies/seeding.py

from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid
import hashlib

from libs.observability import get_logger, timer
from libs.db.models import Company

logger = get_logger(__name__)

@dataclass
class CompanyData:
    name: str
    website: Optional[str] = None
    careers_url: Optional[str] = None
    crawler_profile: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SeedingStats:
    companies_read: int = 0
    companies_deduplicated: int = 0
    companies_created: int = 0
    companies_updated: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class CompanySeedingService:
    def __init__(self, db_session):
        self.db_session = db_session
        self._name_fingerprint_cache: Set[str] = set()

    def _load_company_fingerprints(self):
        try:
            companies = self.db_session.query(Company.name).all()
            for (name,) in companies:
                fingerprint = self._generate_company_fingerprint(name)
                self._name_fingerprint_cache.add(fingerprint)
            logger.info(f"Loaded {len(self._name_fingerprint_cache)} existing company fingerprints.")
        except Exception as e:
            logger.error(f"Failed to load company fingerprints: {e}")
            self._name_fingerprint_cache = set()

    def _parse_companies_file(self, file_path: Path) -> List[CompanyData]:
        if file_path.suffix.lower() == ".csv":
            return self._parse_csv(file_path)
        elif file_path.suffix.lower() == ".json":
            return self._parse_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

    def _parse_csv(self, file_path: Path) -> List[CompanyData]:
        companies = []
        with open(file_path, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                companies.append(CompanyData(
                    name=row.get('name'),
                    website=row.get('domain') or row.get('website'),
                    careers_url=row.get('careers_url')
                ))
        return companies

    def _parse_json(self, file_path: Path) -> List[CompanyData]:
        companies = []
        with open(file_path, mode='r', encoding='utf-8') as infile:
            data = json.load(infile)
            company_list = data.get('companies') if isinstance(data, dict) else data
            for item in company_list:
                companies.append(CompanyData(
                    name=item.get('name'),
                    website=item.get('domain') or item.get('website'),
                    careers_url=item.get('careers_url')
                ))
        return companies

    def _seed_companies_batch(self, companies_data: List[CompanyData], update_existing: bool, stats: SeedingStats):
        new_companies = []
        for company_data in companies_data:
            if not company_data.name:
                stats.errors.append("Skipping row with missing company name.")
                continue

            fingerprint = self._generate_company_fingerprint(company_data.name)
            if fingerprint in self._name_fingerprint_cache:
                stats.companies_deduplicated += 1
                continue

            new_company = Company(
                id=str(uuid.uuid4()),
                name=company_data.name,
                website=company_data.website,
                careers_url=company_data.careers_url,
                crawler_profile_json=json.dumps(company_data.crawler_profile or {})
            )
            new_companies.append(new_company)
            self._name_fingerprint_cache.add(fingerprint)
            stats.companies_created += 1

        if new_companies:
            try:
                self.db_session.add_all(new_companies)
                # The session is committed by the context manager in the CLI
                logger.info(f"Prepared {len(new_companies)} new companies for seeding.")
            except Exception as e:
                logger.error(f"Database error during batch add: {e}")
                stats.errors.append(f"Batch add failed: {e}")
                stats.companies_created -= len(new_companies)


    def seed_companies_from_file(self, file_path: Path, update_existing: bool = False) -> SeedingStats:
        try:
            logger.info("Starting company seeding", file_path=str(file_path))
            with timer("company_seeding.total"):
                self._load_company_fingerprints()
                companies_data = self._parse_companies_file(file_path)

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

                self._seed_companies_batch(companies_data, update_existing, stats)
                logger.info("Company seeding completed", stats=asdict(stats))
                return stats
        except Exception as e:
            logger.error("Company seeding failed", file_path=str(file_path), error=str(e))
            raise

    def _generate_company_fingerprint(self, name: str) -> str:
        normalized_name = name.lower().strip()
        return hashlib.md5(normalized_name.encode('utf-8')).hexdigest()

def create_company_seeding_service(db_session) -> CompanySeedingService:
    return CompanySeedingService(db_session)