#!/usr/bin/env python3
"""
Demonstration of the Company Auto-Discovery functionality

This script shows how the new company seed generation features work
by creating mock company seeds and demonstrating the CLI workflow.
"""

import tempfile
from pathlib import Path
from libs.companies.models import CompanySeed, PortalType, generate_slug
from libs.companies.yaml_writer import YamlWriterService

def demo_company_seed_creation():
    """Demonstrate creating and managing company seeds"""
    
    print("🚀 Company Auto-Discovery Feature Demonstration\n")
    
    # Use temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Initialize YAML writer service
        yaml_writer = YamlWriterService(Path(temp_dir))
        print("✅ Initialized YAML writer service\n")
        
        # Create sample company seeds
        companies = [
            {
                "name": "Anduril Industries",
                "domain": "anduril.com", 
                "careers_url": "https://anduril.com/careers",
                "portal_type": PortalType.GREENHOUSE,
                "company_id": "anduril"
            },
            {
                "name": "Palantir Technologies",
                "domain": "palantir.com",
                "careers_url": "https://www.palantir.com/careers", 
                "portal_type": PortalType.LEVER,
                "company_id": "palantir"
            },
            {
                "name": "SpaceX",
                "domain": "spacex.com",
                "careers_url": "https://www.spacex.com/careers",
                "portal_type": PortalType.WORKDAY,
                "company_id": "spacex"
            }
        ]
        
        print("📝 Creating company seeds...")
        for company_info in companies:
            # Generate company seed
            seed = CompanySeed(
                id=generate_slug(company_info["name"]),
                name=company_info["name"],
                domain=company_info["domain"],
                careers={
                    "primary_url": company_info["careers_url"],
                    "discovered_alternatives": []
                },
                portal={
                    "type": company_info["portal_type"],
                    "adapter": f"{company_info['portal_type'].value}_v1",
                    "portal_config": {"company_id": company_info["company_id"]}
                },
                metadata={
                    "confidence": {
                        "careers_url": 0.95,
                        "portal_detection": 0.89
                    },
                    "discovery_method": "demo"
                },
                notes="Demo company seed for testing"
            )
            
            # Write to YAML
            file_path = yaml_writer.write_company_seed(seed)
            print(f"  ✅ Created {seed.name} -> {file_path.name}")
        
        print(f"\n📋 Company Seeds Directory Contents:")
        for file_path in yaml_writer.companies_dir.glob("*.yaml"):
            print(f"  📄 {file_path.name}")
        
        # Show company list 
        print(f"\n📊 Company Seeds Index:")
        companies_index = yaml_writer.list_company_seeds()
        for company_id, info in companies_index.items():
            print(f"  🏢 {company_id}:")
            print(f"     Name: {info['name']}")
            print(f"     Domain: {info['domain']}")  
            print(f"     Portal: {info['portal_type']}")
            print(f"     Careers URL: {info['careers_url']}")
        
        # Demo reading a seed back
        print(f"\n🔍 Reading back a company seed...")
        seed = yaml_writer.read_company_seed("anduril-industries")
        if seed:
            print(f"  ✅ Successfully read: {seed.name}")
            print(f"     Domain: {seed.domain}")
            print(f"     Portal: {seed.portal.type.value}")
            print(f"     Confidence: {seed.metadata.get('confidence', {})}")
        
        # Demo dry-run YAML generation
        print(f"\n📋 Sample Generated YAML:")
        print("=" * 50)
        sample_seed = yaml_writer.read_company_seed("spacex")
        if sample_seed:
            yaml_content = yaml_writer.generate_dry_run_yaml(sample_seed)
            print(yaml_content)
        print("=" * 50)

if __name__ == "__main__":
    demo_company_seed_creation()