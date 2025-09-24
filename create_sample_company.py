#!/usr/bin/env python3
"""
Create a sample company seed for CLI demonstration
"""

from libs.companies.models import CompanySeed, PortalType, generate_slug
from libs.companies.yaml_writer import YamlWriterService

def create_sample_company():
    """Create a sample company seed to demonstrate CLI functionality"""
    
    # Create sample seed
    seed = CompanySeed(
        id=generate_slug("Example Tech Company"),
        name="Example Tech Company",
        domain="example-tech.com",
        careers={
            "primary_url": "https://example-tech.com/careers",
            "discovered_alternatives": ["https://example-tech.com/jobs"]
        },
        portal={
            "type": PortalType.GREENHOUSE,
            "adapter": "greenhouse_v1",
            "portal_config": {"company_id": "example-tech"}
        },
        metadata={
            "confidence": {
                "careers_url": 0.92,
                "portal_detection": 0.88
            },
            "discovery_method": "manual_demo"
        },
        notes="Sample company seed for CLI demonstration"
    )
    
    # Write to default location
    yaml_writer = YamlWriterService()
    file_path = yaml_writer.write_company_seed(seed)
    
    print(f"✅ Created sample company seed: {file_path}")
    print(f"🏢 Company ID: {seed.id}")
    print(f"📝 Name: {seed.name}")
    print(f"🌐 Domain: {seed.domain}")
    print(f"🔗 Careers URL: {seed.careers.primary_url}")
    print(f"🏭 Portal: {seed.portal.type.value}")
    
    return seed.id

if __name__ == "__main__":
    company_id = create_sample_company()
    print(f"\n💡 Try these commands:")
    print(f"   python -m cli.ljs companies list")
    print(f"   python -m cli.ljs companies show {company_id}")
    print(f"   python -m cli.ljs companies select {company_id}")