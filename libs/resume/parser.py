# Factory for CLI/ingestion compatibility
def create_resume_parser(use_llm: bool = True):
    return ResumeParser(use_llm=use_llm)
# direct_openai_fix.py
# Add this to libs/resume/parser.py to force OpenAI usage

import os
from dotenv import load_dotenv

# Force load .env from project root
import pathlib
env_path = pathlib.Path.cwd() / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded .env from {env_path}")

class ResumeParser:
    """Parser for resume documents (PDF and DOCX)"""
    
    def __init__(self, use_llm: bool = True):
        """Initialize the resume parser"""
        self.use_llm = use_llm
        self._compiled_patterns = {}
        self._compile_section_patterns()
        
        if self.use_llm:
            import logging
            logger = logging.getLogger(__name__)
            
            # Force check for API keys
            openai_key = os.getenv('OPENAI_API_KEY')
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            gemini_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
            
            # Determine which provider to use based on available keys
            if openai_key:
                provider_name = 'openai'
                model = os.getenv('LJS_LLM_MODEL', 'gpt-4-turbo-preview')
                logger.info(f"Using OpenAI with model: {model}")
            elif anthropic_key:
                provider_name = 'anthropic'  
                model = os.getenv('LJS_LLM_MODEL', 'claude-3-opus-20240229')
                logger.info(f"Using Anthropic with model: {model}")
            elif gemini_key:
                provider_name = 'gemini'
                model = os.getenv('LJS_LLM_MODEL', 'gemini-pro')
                logger.info(f"Using Gemini with model: {model}")
            else:
                raise ValueError(
                    "No LLM API key found! Please set one of the following in your .env file:\n"
                    "  - OPENAI_API_KEY=sk-...\n"
                    "  - ANTHROPIC_API_KEY=sk-ant-...\n"
                    "  - GOOGLE_API_KEY=...\n"
                    f"Current working directory: {os.getcwd()}\n"
                    f"Looking for .env at: {env_path}"
                )
            
            # Override any config with detected provider
            os.environ['LJS_LLM_PROVIDER'] = provider_name
            os.environ['LJS_LLM_MODEL'] = model
            
            from .llm_service import create_llm_service, LLMProvider
            
            # Map to enum
            provider_map = {
                'openai': LLMProvider.OPENAI,
                'anthropic': LLMProvider.ANTHROPIC,
                'gemini': LLMProvider.GEMINI
            }
            
            provider_enum = provider_map[provider_name]
            self.llm_service = create_llm_service(provider=provider_enum, model=model)
            logger.info(f"LLM Service initialized with {provider_name}")
