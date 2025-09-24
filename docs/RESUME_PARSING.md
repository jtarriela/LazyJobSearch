# Resume Parsing Documentation

## Overview

The LazyJobSearch resume parsing system uses an **LLM-first approach** with structured schema validation, retry logic for missing fields, and fallback mechanisms. This ensures high-quality extraction of resume data with proper error handling.

## Architecture

### Core Components

1. **ParsedResumeData**: Pydantic models with strict validation
2. **LLMResumeParser**: Main parsing engine with retry logic  
3. **LLMClient**: Abstracted client interface for multiple providers
4. **LLMConfig**: Environment-based configuration

### Key Features

- **Strict Schema Validation**: All resume data is validated against pydantic models
- **Targeted Retries**: Missing fields are specifically requested in follow-up attempts
- **PDF + Text Support**: Providers can use PDF bytes directly or fallback to text
- **JSON Extraction**: Robust parsing of LLM responses that may contain extra text
- **Merge Logic**: Data from multiple attempts is intelligently merged

## Schema Definition

### Required Fields

The system enforces these required fields for complete resume parsing:

```python
REQUIRED_FIELDS = [
    "full_name",      # Person's full name
    "email",          # Primary email address  
    "phone",          # Phone number
    "skills",         # List of all skills mentioned
    "experience",     # Work experience entries
    "education",      # Educational background
    "full_text"       # Complete resume text for chunking
]
```

### Data Models

#### ParsedResumeData
Main container for resume data with validation:

```python
class ParsedResumeData(BaseModel):
    # Core required fields
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience: List[ExperienceItem] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    full_text: Optional[str] = None
    
    # Optional fields
    summary: Optional[str] = None
    certifications: List[CertificationItem] = Field(default_factory=list)
    projects: List[ProjectItem] = Field(default_factory=list)
    links: Links = Field(default_factory=Links)
    skills_structured: Skills = Field(default_factory=Skills)
    years_of_experience: Optional[float] = None
    education_level: Optional[str] = None
```

#### Sub-Models

- **ExperienceItem**: Job title, company, duration, description, location
- **EducationItem**: Degree, field, institution, year, GPA
- **CertificationItem**: Name, issuer, date
- **ProjectItem**: Name, description, technologies
- **Links**: LinkedIn, GitHub, portfolio, other URLs
- **Skills**: Technical, soft, languages, tools (categorized)

## Prompting Strategy

### System Prompt
Standard instruction emphasizing accuracy and JSON-only output:

```
You are an expert resume parser. Your task is to extract structured information from resume text and return it as valid JSON.

Rules:
- Return ONLY a single JSON object, no additional text or explanation
- If a field is not found, use null for strings/numbers or empty array [] for lists  
- Do not hallucinate information that is not present
- Extract ALL skills mentioned, including technical and soft skills
- Include ALL work experience and education entries
- Be thorough and accurate
```

### Initial User Prompt
Provides resume text and complete schema structure with examples.

### Retry Prompt (Targeted)
When fields are missing after initial attempt:

```
I previously parsed this resume but some fields were missing. Please focus on extracting the missing information.

Resume: {resume_text}
Missing fields: {missing_fields}
Current data: {current_extracted_data}

Please return a JSON object with ONLY the missing fields filled in.
```

## Retry Logic and Merge

### Retry Flow

1. **Attempt 1**: Full parsing with initial prompt
2. **Validation**: Check for missing required fields
3. **Attempt N**: Targeted retry focusing on missing fields only
4. **Merge**: Combine new data with existing, preferring non-empty values
5. **Early Exit**: Stop when all required fields are present

### Merge Strategy

```python
def _merge_parsed_data(existing, new):
    # Prefer non-empty values from new data
    # For lists, use longer list if new has more items
    # Preserve existing data when new is empty/null
```

## Configuration

### Environment Variables

```bash
# Provider configuration
LJS_LLM_PROVIDER=mock|openai|anthropic  # Default: mock
LJS_LLM_MODEL=gpt-3.5-turbo             # Default: gpt-3.5-turbo
LJS_LLM_TIMEOUT=30                      # Request timeout in seconds
LJS_LLM_MAX_TOKENS=2000                 # Maximum tokens per request

# Provider-specific API keys
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
```

### Usage

```python
from libs.resume.llm_service import LLMResumeParser, LLMConfig

# Use environment config
config = LLMConfig()
parser = LLMResumeParser(config)

# Parse with PDF bytes (preferred)
result, responses = await parser.parse_resume(
    pdf_bytes=pdf_data,
    fallback_text=extracted_text
)

# Parse with text only
result, responses = await parser.parse_resume(
    fallback_text=resume_text
)
```

## JSON Extraction

The system includes a robust JSON extraction utility that handles LLM responses containing extra text:

```python
def _extract_json(self, text: str) -> str:
    """Extract JSON from LLM response that may contain extra text"""
    # Handles responses like:
    # "Here's the parsed resume data: {json...} Hope this helps!"
    # Returns clean JSON object for parsing
```

## Error Handling

### Validation Errors
- Pydantic validation failures create minimal required objects
- Invalid emails/phones are set to `None`
- Malformed lists are normalized to empty lists

### LLM Request Failures
- Exponential backoff between retry attempts
- Graceful degradation with partial data
- Final fallback returns minimal valid object

### JSON Parsing Errors
- Robust extraction handles malformed responses
- Detailed error logging with response content
- Fallback to minimal data structure

## Testing

### Unit Test Coverage
- Configuration validation and defaults
- Schema validation and missing field detection
- Mock LLM client functionality
- JSON extraction from messy responses
- Data merging across retry attempts

### Test Example
```python
@pytest.mark.asyncio
async def test_llm_resume_parser_basic():
    parser = LLMResumeParser()
    result, responses = await parser.parse_resume(
        fallback_text="John Doe\nSoftware Engineer\nEmail: john@example.com"
    )
    assert isinstance(result, ParsedResumeData)
    assert result.full_text
    assert len(responses) > 0
```

## Integration with Ingestion Pipeline

The LLM parser integrates with the existing resume ingestion pipeline:

1. **File Processing**: PDF/DOCX text extraction
2. **LLM Parsing**: Structured data extraction with retries
3. **Chunking**: Uses `parsed.full_text` for embedding chunks
4. **Embedding Generation**: Creates vectors from chunks
5. **Persistence**: Saves parsed data and embeddings to database

## Provider Support

### Implemented
- **Mock**: For testing and development
- **OpenAI**: Placeholder (TODO: implement API calls)
- **Anthropic**: Placeholder (TODO: implement API calls)

### Adding New Providers

1. Implement `LLMClient` interface
2. Add to `LLMConfig` validation
3. Update `create_llm_client` factory
4. Add provider-specific environment variables

## Monitoring and Observability

### Logging
- Structured logging with attempt numbers
- Missing fields tracking
- Cost and token usage
- Error details with context

### Metrics
- Parsing success/failure rates
- Average attempts per resume
- Token usage and costs per provider
- Field completion rates

## Best Practices

### For Resume Quality
- Ensure PDF text extraction is clean
- Provide fallback text when PDF parsing fails  
- Use appropriate timeout values for provider
- Monitor token costs and usage

### For Development
- Use mock provider for testing
- Implement provider-specific error handling
- Add comprehensive test coverage
- Monitor parsing quality metrics

### for Production
- Set reasonable retry limits (3 attempts recommended)
- Implement proper API key rotation
- Monitor provider rate limits
- Use structured logging for debugging