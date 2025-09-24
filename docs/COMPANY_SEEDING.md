# Company Seeding Guide

This document describes the company seeding functionality for LazyJobSearch, which allows bulk import of company data from external sources.

## Overview

Company seeding enables users to populate the LazyJobSearch database with company information from CSV or JSON files. The seeding process includes:

- Company data parsing and validation
- Duplicate detection and deduplication
- Careers URL validation and discovery
- Batch insertion with comprehensive error handling
- Idempotent operations (safe to run multiple times)

## Supported Formats

### CSV Format

CSV files must include a header row with the following required and optional columns:

**Required Columns:**
- `name` - Company name (string)
- `domain` - Primary company domain (string, e.g., "example.com")

**Optional Columns:**
- `careers_url` - Direct link to careers page (string URL)
- `description` - Company description (string)
- `industry` - Industry category (string)
- `size` - Company size category (string: "startup", "small", "medium", "large", "enterprise")
- `location` - Primary location (string)
- `founded_year` - Year founded (integer)
- `stock_symbol` - Stock ticker symbol (string)
- `linkedin_url` - LinkedIn company page URL (string)
- `twitter_handle` - Twitter handle without @ (string)
- `tags` - Comma-separated tags (string, e.g., "tech,saas,remote-friendly")

**Example CSV:**
```csv
name,domain,careers_url,description,industry,size,location,founded_year,stock_symbol,linkedin_url,twitter_handle,tags
Acme Corp,acme.com,https://acme.com/careers,Leading software company,Technology,medium,San Francisco CA,2010,ACME,https://linkedin.com/company/acme,acmecorp,tech,saas
TechStart Inc,techstart.io,,Innovative startup in AI,Technology,startup,Austin TX,2020,,,techstartio,ai,startup,remote-friendly
```

### JSON Format

JSON files can contain either:

1. **Array of company objects:**
```json
[
  {
    "name": "Acme Corp",
    "domain": "acme.com", 
    "careers_url": "https://acme.com/careers",
    "description": "Leading software company",
    "industry": "Technology",
    "size": "medium",
    "location": "San Francisco, CA",
    "founded_year": 2010,
    "stock_symbol": "ACME",
    "linkedin_url": "https://linkedin.com/company/acme",
    "twitter_handle": "acmecorp",
    "tags": ["tech", "saas", "remote-friendly"]
  },
  {
    "name": "TechStart Inc",
    "domain": "techstart.io",
    "description": "Innovative startup in AI",
    "industry": "Technology", 
    "size": "startup",
    "location": "Austin, TX",
    "founded_year": 2020,
    "tags": ["ai", "startup", "remote-friendly"]
  }
]
```

2. **Object with companies array:**
```json
{
  "version": "1.0",
  "source": "manual_research_2024",
  "companies": [
    {
      "name": "Acme Corp",
      "domain": "acme.com",
      "careers_url": "https://acme.com/careers"
    }
  ]
}
```

## Usage

### CLI Command

```bash
# Seed from CSV file
ljs companies seed --file companies.csv

# Seed from JSON file  
ljs companies seed --file companies.json

# Update existing companies (default: skip duplicates)
ljs companies seed --file companies.csv --update-existing

# Dry run to preview changes
ljs companies seed --file companies.csv --dry-run
```

### Programmatic Usage

```python
from libs.companies.seeding import create_company_seeding_service
from pathlib import Path

# Create seeding service
seeding_service = create_company_seeding_service(db_session)

# Seed companies from file
stats = await seeding_service.seed_companies_from_file(
    Path("companies.csv"),
    update_existing=False
)

print(f"Created: {stats.companies_created}")
print(f"Updated: {stats.companies_updated}")  
print(f"Deduplicated: {stats.companies_deduplicated}")
print(f"Errors: {len(stats.errors)}")
```

## Deduplication Logic

The seeding service prevents duplicate companies using multiple strategies:

1. **Exact domain match** - Companies with identical domains are considered duplicates
2. **Name normalization** - Company names are normalized (lowercased, special characters removed) for fuzzy matching
3. **Domain fingerprinting** - Similar domains (e.g., "acme.com" vs "www.acme.com") are detected

### Deduplication Behavior

- **Default mode (`update_existing=False`)**: Skip inserting duplicates, increment deduplicated counter
- **Update mode (`update_existing=True`)**: Update existing company record with new data, increment updated counter

## Error Handling

The seeding service provides comprehensive error handling:

### File-level Errors
- Invalid file format or encoding
- Missing required columns (CSV)  
- Invalid JSON structure

### Record-level Errors
- Missing required fields (`name`, `domain`)
- Invalid domain format
- Invalid URL format for careers_url, linkedin_url
- Invalid data types (e.g., non-integer founded_year)

### Recovery Strategies
- Continue processing remaining records after individual record failures
- Collect all errors for batch reporting
- Rollback transaction on critical failures

## Careers URL Discovery

When `careers_url` is not provided, the seeding service can attempt automatic discovery:

1. **Common patterns**: Check `/careers`, `/jobs`, `/opportunities`
2. **Subdomain patterns**: Check `careers.domain.com`, `jobs.domain.com`  
3. **HTTP/HTTPS detection**: Try both protocols
4. **Response validation**: Ensure discovered URLs return valid responses

*Note: Automatic discovery is disabled by default to avoid rate limiting. Enable with `--discover-careers` flag.*

## Performance Considerations

### Batch Processing
- Companies are processed in batches of 100 by default
- Configurable batch size via `--batch-size` parameter
- Transaction per batch for faster rollback on errors

### Rate Limiting  
- Built-in throttling for careers URL validation (1 request per second)
- Configurable via `--request-delay` parameter
- Respects robots.txt when available

### Memory Usage
- Streaming JSON/CSV parsing for large files
- Limited in-memory company data (1000 records buffered)
- Automatic memory cleanup between batches

## Example Files

See the `docs/examples/seed_data/` directory for sample files:

- `companies_basic.csv` - Basic CSV format with required fields only
- `companies_full.csv` - Full CSV format with all optional fields
- `companies_basic.json` - Basic JSON array format
- `companies_full.json` - Full JSON format with metadata

## Validation Schema

All seeded company data is validated against the following schema:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["name", "domain"],
  "properties": {
    "name": {
      "type": "string",
      "minLength": 1,
      "maxLength": 255
    },
    "domain": {
      "type": "string", 
      "pattern": "^[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]*\\.[a-zA-Z]{2,}$"
    },
    "careers_url": {
      "type": "string",
      "format": "uri"
    },
    "description": {
      "type": "string",
      "maxLength": 2000
    },
    "industry": {
      "type": "string",
      "maxLength": 100
    },
    "size": {
      "type": "string", 
      "enum": ["startup", "small", "medium", "large", "enterprise"]
    },
    "location": {
      "type": "string",
      "maxLength": 255
    },
    "founded_year": {
      "type": "integer",
      "minimum": 1800,
      "maximum": 2030
    },
    "stock_symbol": {
      "type": "string",
      "maxLength": 10
    },
    "linkedin_url": {
      "type": "string",
      "format": "uri"
    },
    "twitter_handle": {
      "type": "string",
      "pattern": "^[a-zA-Z0-9_]{1,15}$"
    },
    "tags": {
      "type": "array",
      "items": {
        "type": "string",
        "maxLength": 50
      },
      "maxItems": 20
    }
  }
}
```