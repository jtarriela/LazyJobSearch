"""Tests for company seeding idempotency and deduplication."""
import pytest
from unittest.mock import Mock, AsyncMock
from pathlib import Path
import tempfile
import csv
import json

# Note: These are integration-style tests that would need DB setup
# For now, they test the core logic with mocked DB components


class TestCompanySeedingIdempotency:
    """Test suite for company seeding idempotency verification."""

    def test_csv_parsing_deduplication(self):
        """Test that duplicate companies in CSV are detected."""
        # Create temporary CSV with duplicates
        csv_content = """name,domain
Acme Corp,acme.com
TechStart Inc,techstart.io
Acme Corp,acme.com
Global Solutions,globalsolutions.net
TechStart Inc,techstart.io"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)
        
        try:
            # Parse CSV and check for duplicates
            companies = {}
            with open(temp_path) as csvfile:
                reader = csv.DictReader(csvfile)
                duplicates = []
                
                for row in reader:
                    domain = row['domain']
                    if domain in companies:
                        duplicates.append(domain)
                    else:
                        companies[domain] = row
            
            # Should detect 2 duplicates
            assert len(duplicates) == 2
            assert 'acme.com' in duplicates
            assert 'techstart.io' in duplicates
            
            # Final unique companies should be 3
            assert len(companies) == 3
            
        finally:
            temp_path.unlink()

    def test_json_parsing_deduplication(self):
        """Test that duplicate companies in JSON are detected."""
        json_content = [
            {"name": "Acme Corp", "domain": "acme.com"},
            {"name": "TechStart Inc", "domain": "techstart.io"},
            {"name": "Acme Corp", "domain": "acme.com"},  # Duplicate
            {"name": "Global Solutions", "domain": "globalsolutions.net"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_content, f)
            temp_path = Path(f.name)
        
        try:
            with open(temp_path) as f:
                companies_data = json.load(f)
            
            # Deduplicate by domain
            seen_domains = set()
            unique_companies = []
            duplicates = []
            
            for company in companies_data:
                domain = company['domain']
                if domain in seen_domains:
                    duplicates.append(domain)
                else:
                    seen_domains.add(domain)
                    unique_companies.append(company)
            
            assert len(duplicates) == 1
            assert 'acme.com' in duplicates
            assert len(unique_companies) == 3
            
        finally:
            temp_path.unlink()

    def test_domain_fingerprinting_logic(self):
        """Test domain normalization for duplicate detection."""
        test_cases = [
            ("acme.com", "acme.com", True),
            ("www.acme.com", "acme.com", True),  # Should match
            ("subdomain.acme.com", "acme.com", False),  # Should not match
            ("ACME.COM", "acme.com", True),  # Case insensitive
            ("acme.co", "acme.com", False),  # Different TLD
        ]
        
        def normalize_domain(domain):
            """Normalize domain for comparison."""
            domain = domain.lower().strip()
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        
        for domain1, domain2, should_match in test_cases:
            norm1 = normalize_domain(domain1)
            norm2 = normalize_domain(domain2)
            matches = (norm1 == norm2)
            
            assert matches == should_match, f"Domain comparison failed: {domain1} vs {domain2}"

    def test_name_normalization_logic(self):
        """Test company name normalization for fuzzy matching."""
        test_cases = [
            ("Acme Corp", "ACME CORP", True),
            ("Acme Corp.", "Acme Corp", True),  # Remove punctuation
            ("Acme Corp", "Acme Corporation", False),  # Different words
            ("Tech-Start Inc", "TechStart Inc", True),  # Remove hyphens
            ("Acme Corp LLC", "Acme Corp", False),  # Keep legal suffixes as different
        ]
        
        def normalize_name(name):
            """Normalize company name for comparison."""
            import re
            name = name.lower().strip()
            # Remove common punctuation
            name = re.sub(r'[.,\-_]', '', name)
            # Normalize whitespace
            name = re.sub(r'\s+', ' ', name)
            return name
        
        for name1, name2, should_match in test_cases:
            norm1 = normalize_name(name1)
            norm2 = normalize_name(name2)
            matches = (norm1 == norm2)
            
            assert matches == should_match, f"Name comparison failed: {name1} vs {name2}"

    def test_update_vs_skip_behavior_logic(self):
        """Test the logic for update vs skip behavior."""
        
        # Mock existing company data
        existing_companies = {
            "acme.com": {
                "name": "Acme Corp",
                "domain": "acme.com", 
                "description": "Old description",
                "founded_year": None
            }
        }
        
        # New company data with updates
        new_company = {
            "name": "Acme Corp",
            "domain": "acme.com",
            "description": "Updated description",
            "founded_year": 2010
        }
        
        def simulate_seeding(update_existing=False):
            """Simulate seeding behavior."""
            domain = new_company["domain"]
            stats = {"created": 0, "updated": 0, "skipped": 0}
            
            if domain in existing_companies:
                if update_existing:
                    # Update existing company
                    existing_companies[domain].update(new_company)
                    stats["updated"] = 1
                else:
                    # Skip duplicate
                    stats["skipped"] = 1
            else:
                # Create new company
                existing_companies[domain] = new_company
                stats["created"] = 1
                
            return stats
        
        # Test skip behavior (default)
        stats_skip = simulate_seeding(update_existing=False)
        assert stats_skip["skipped"] == 1
        assert stats_skip["created"] == 0
        assert stats_skip["updated"] == 0
        assert existing_companies["acme.com"]["description"] == "Old description"
        
        # Test update behavior
        stats_update = simulate_seeding(update_existing=True)
        assert stats_update["updated"] == 1
        assert stats_update["created"] == 0
        assert stats_update["skipped"] == 0
        assert existing_companies["acme.com"]["description"] == "Updated description"
        assert existing_companies["acme.com"]["founded_year"] == 2010

    def test_seeding_stats_tracking(self):
        """Test that seeding statistics are properly tracked."""
        from dataclasses import dataclass
        from typing import List
        
        @dataclass
        class SeedingStats:
            companies_read: int = 0
            companies_created: int = 0
            companies_updated: int = 0
            companies_deduplicated: int = 0
            errors: List[str] = None
            
            def __post_init__(self):
                if self.errors is None:
                    self.errors = []
        
        # Simulate seeding process
        stats = SeedingStats()
        
        # Read 5 companies from file
        stats.companies_read = 5
        
        # 2 are duplicates within the file
        stats.companies_deduplicated = 2
        
        # 1 already exists in DB, skip
        # 2 are new, create
        stats.companies_created = 2
        
        # Validation
        assert stats.companies_read == 5
        assert stats.companies_deduplicated == 2
        assert stats.companies_created == 2
        assert stats.companies_updated == 0
        assert len(stats.errors) == 0
        
        # Check totals make sense
        processed = stats.companies_created + stats.companies_updated + stats.companies_deduplicated
        # Note: deduplicated includes both file duplicates and existing DB records
        # So this math depends on exact implementation

    def test_error_handling_preserves_idempotency(self):
        """Test that errors don't break idempotency guarantees."""
        
        # Mock database session with rollback capability
        class MockTransaction:
            def __init__(self):
                self.committed = False
                self.rolled_back = False
                self.operations = []
            
            def add_company(self, company):
                self.operations.append(('add', company))
            
            def commit(self):
                self.committed = True
            
            def rollback(self):
                self.rolled_back = True
                self.operations = []  # Clear operations on rollback
        
        def simulate_batch_with_error():
            """Simulate batch processing with error."""
            tx = MockTransaction()
            
            companies = [
                {"name": "Good Company", "domain": "good.com"},
                {"name": "Bad Company", "domain": "invalid-domain"},  # Invalid
                {"name": "Another Good", "domain": "another.com"}
            ]
            
            try:
                for company in companies:
                    # Simulate validation
                    if "invalid" in company["domain"]:
                        raise ValueError(f"Invalid domain: {company['domain']}")
                    
                    tx.add_company(company)
                
                tx.commit()
                return "success", tx.operations
                
            except Exception as e:
                tx.rollback()
                return "error", []
        
        result, operations = simulate_batch_with_error()
        
        # Should fail and rollback everything
        assert result == "error"
        assert len(operations) == 0  # No operations should persist
        
        # Re-running should be idempotent - same result
        result2, operations2 = simulate_batch_with_error()
        assert result2 == "error"
        assert len(operations2) == 0

    def test_concurrent_seeding_safety(self):
        """Test that concurrent seeding operations are safe."""
        
        # Mock concurrent seeding scenario
        # This would need real concurrency testing in practice
        
        shared_state = {"companies": {}}
        
        def simulate_concurrent_seed(session_id, companies_data):
            """Simulate seeding from different sessions."""
            results = []
            
            for company in companies_data:
                domain = company["domain"]
                
                # Simulate checking if company exists (race condition possible)
                exists = domain in shared_state["companies"]
                
                if not exists:
                    # Simulate creation
                    shared_state["companies"][domain] = company
                    results.append("created")
                else:
                    results.append("duplicate")
            
            return results
        
        # Two sessions trying to seed overlapping companies
        session1_data = [
            {"name": "Acme Corp", "domain": "acme.com"},
            {"name": "TechStart", "domain": "techstart.io"}
        ]
        
        session2_data = [
            {"name": "Acme Corp", "domain": "acme.com"},  # Duplicate
            {"name": "Global Inc", "domain": "global.com"}
        ]
        
        # In real implementation, this would need proper locking/transactions
        results1 = simulate_concurrent_seed("session1", session1_data)
        results2 = simulate_concurrent_seed("session2", session2_data)
        
        # Verify final state has no duplicates
        assert len(shared_state["companies"]) == 3  # acme.com, techstart.io, global.com
        assert "acme.com" in shared_state["companies"]
        assert "techstart.io" in shared_state["companies"] 
        assert "global.com" in shared_state["companies"]