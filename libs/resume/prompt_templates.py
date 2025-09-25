"""
Enhanced prompt templates for comprehensive resume parsing.
Ensures complete extraction of all resume information.
"""

from __future__ import annotations
from typing import Dict, Any, Iterable, Tuple, Optional
import json

# Comprehensive schema for resume extraction
COMPREHENSIVE_SCHEMA = """{
  "full_name": "string",
  "email": "string", 
  "phone": "string",
  "location": "string or null",
  "summary": "professional summary or objective - full text",
  "titles": ["list of all professional titles mentioned"],
  "skills": [
    "COMPLETE list of EVERY skill, technology, tool, framework, library, platform, language, etc.",
    "Include items from parenthetical lists (e.g. 'Python (pandas, numpy)' extracts as Python, pandas, numpy)",
    "Include all acronyms AND their expansions if mentioned"
  ],
  "skills_structured": {
    "programming": ["all programming languages"],
    "frameworks": ["all frameworks and libraries"], 
    "tools": ["all tools, IDEs, and development platforms"],
    "databases": ["all database technologies"],
    "cloud": ["all cloud platforms and services"],
    "ml_ai": ["all ML/AI/Data Science technologies"],
    "other": ["any other technical skills"],
    "all": ["complete combined list of everything"]
  },
  "experience": [
    {
      "title": "exact job title",
      "company": "company name",
      "duration": "complete date range as shown",
      "location": "location if mentioned",
      "description": "role overview if provided",
      "bullets": [
        "complete text of each bullet point",
        "preserve all metrics, percentages, and achievements",
        "do not summarize or shorten"
      ],
      "technologies": ["technologies specifically used in this role"]
    }
  ],
  "education": [
    {
      "degree": "degree type (B.S., M.S., Ph.D., etc.)",
      "field": "field of study",
      "institution": "complete institution name",
      "year": "graduation year or date range",
      "gpa": "GPA if mentioned",
      "specialization": "any specialization, concentration, or thesis topic"
    }
  ],
  "certifications": [
    {"name": "certification name", "issuer": "issuing organization", "date": "date obtained"}
  ],
  "projects": [
    {
      "name": "project name",
      "description": "complete project description",
      "technologies": ["technologies used"],
      "link": "project URL if provided"
    }
  ],
  "awards": [
    "complete text of each award, honor, or achievement"
  ],
  "publications": [
    "complete citation or description of each publication"
  ],
  "links": {
    "linkedin": "LinkedIn URL",
    "github": "GitHub URL", 
    "portfolio": "portfolio/website URL",
    "other": ["any other URLs"]
  },
  "years_of_experience": number or null,
  "education_level": "highest degree achieved",
  "clearance_level": "security clearance level if mentioned (e.g., 'ACTIVE CLEARANCE', 'Secret', 'Top Secret')"
}"""

def build_parse_prompt(
    text: str,
    hints: Optional[Dict[str, Any]] = None,
    schema_hint: Optional[str] = None,
) -> Tuple[str, str]:
    """Build comprehensive extraction prompt for first-pass parsing."""
    
    system = """You are an expert resume parser with deep knowledge of technical roles and terminology.
Your task is to extract EVERY piece of information from the resume with 100% accuracy.

Critical extraction rules:
1. Extract ALL skills, technologies, tools, and platforms mentioned ANYWHERE in the resume
2. When skills are listed with examples in parentheses (e.g., "Python (pandas, numpy, scikit-learn)"), extract BOTH the main item AND each item in parentheses as separate skills
3. Include all acronyms AND their expansions if both are mentioned
4. Extract complete text of all bullet points - do not summarize
5. Preserve all metrics, percentages, and quantified achievements exactly as stated
6. Include all dates, durations, and time periods exactly as shown
7. Extract security clearance information if mentioned
8. Do NOT invent or infer information not explicitly stated

Return ONLY valid JSON matching the schema. No markdown, no commentary."""

    schema = schema_hint if schema_hint else COMPREHENSIVE_SCHEMA
    
    user = f"""Extract ALL information from this resume into the exact JSON structure.

CRITICAL INSTRUCTIONS:
- Extract EVERY skill, technology, tool mentioned (including from parenthetical lists)
- Include complete text of all experience bullets and descriptions
- Do not skip or summarize any information
- If parsing "Python (pandas, numpy, scikit-learn, LightGBM, RAPIDS, ...)" extract: Python, pandas, numpy, scikit-learn, LightGBM, RAPIDS
- Look for skills in: summary, experience bullets, education, projects, and dedicated skills sections
- Include both general categories (e.g., "Machine Learning") and specific tools (e.g., "TensorFlow", "PyTorch")

SCHEMA:
{schema}

RESUME TEXT:
{text}

Return ONLY the JSON object. No additional text."""
    
    return system, user

def build_retry_prompt(
    text: str,
    missing_fields: Iterable[str],
    current_json: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Build retry prompt for missing fields with emphasis on completeness."""
    
    system = """You are an expert resume parser. 
Some critical information was not extracted in the first pass.
Your task is to carefully re-read the resume and extract ALL missing information.

Remember:
- Skills can appear anywhere in the resume (summary, experience, education, projects)
- Extract items from parenthetical lists as separate skills
- Include all variations of technologies mentioned
- Do not skip any sections of the resume"""

    missing_str = ", ".join(str(f) for f in missing_fields)
    current_str = json.dumps(current_json or {}, indent=2)
    
    user = f"""The following fields are incomplete or missing. Extract them comprehensively.

MISSING/INCOMPLETE FIELDS:
{missing_str}

CURRENT EXTRACTED DATA:
{current_str}

Special attention to skills extraction:
- If current skills list has fewer than 20 items for a technical resume, you likely missed many
- Check experience bullets for technologies used
- Check education section for relevant coursework or tools
- Look for skills mentioned in context, not just in skills sections

RESUME TEXT:
{text}

Return JSON with the complete missing information. Be exhaustive in your extraction."""
    
    return system, user

def build_skill_enhancement_prompt(text: str, current_skills: List[str]) -> Tuple[str, str]:
    """Special prompt for comprehensive skill extraction when initial extraction is insufficient."""
    
    system = """You are a technical recruiter and resume expert specializing in identifying technical skills.
Your task is to identify EVERY technical skill, tool, technology, framework, library, and platform mentioned in this resume.

Include:
- Programming languages
- Frameworks and libraries  
- Development tools and IDEs
- Databases and data stores
- Cloud platforms and services
- ML/AI technologies
- DevOps and CI/CD tools
- Testing frameworks
- Containerization and orchestration
- Version control systems
- Operating systems
- Methodologies and practices
- Any other technical competency"""
    
    current_skills_str = ", ".join(current_skills) if current_skills else "none found yet"
    
    user = f"""Extract ALL technical skills from this resume. The current extraction only found: {current_skills_str}

This appears incomplete. Carefully read through:
1. The skills section (including items in parentheses)
2. Each job's responsibilities and achievements
3. Education and certifications
4. Projects and publications
5. Any technical terms mentioned anywhere

For parenthetical lists like "Python (pandas, numpy, scikit-learn)", extract each item separately.

RESUME TEXT:
{text}

Return a JSON object with a single key "skills" containing a comprehensive array of ALL technical skills found."""
    
    return system, user

def build_requirements_prompt(pdf_text: str) -> Tuple[str, str]:
    """Build prompt for inferring requirements from PDF text."""
    
    system = """You are a document analysis expert.
Given PDF text containing a form or document with fields to fill, identify all the input fields being requested.
Return ONLY a JSON array describing these fields."""
    
    user = f"""Analyze this PDF text and identify all fields/inputs being requested.

Return a JSON array where each object has:
- name: field name
- type: string|number|date|email|phone|url|enum|address|textarea  
- required: boolean
- description: string or null
- options: array of options for enum types, null otherwise
- regex_hint: validation pattern hint if applicable
- example: example value if provided

PDF TEXT:
{pdf_text}

Return ONLY the JSON array."""
    
    return system, user