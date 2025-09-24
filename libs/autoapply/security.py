"""Security utilities for template DSL execution.

Provides input sanitization, validation, and sandboxing capabilities
for portal template execution to prevent injection attacks and ensure
safe template variable substitution.
"""
from __future__ import annotations
import html
import re
import urllib.parse
from typing import Any, Dict, Set, Optional
import logging

logger = logging.getLogger(__name__)


class TemplateSanitizer:
    """Sanitizes template variables to prevent injection attacks."""
    
    # Allowed variable patterns - only profile.*, files.*, and answers.*  
    # but exclude sensitive profile fields
    ALLOWED_VARIABLE_PATTERNS = {
        r'^profile\.(?!password|secret|token)[a-zA-Z_][a-zA-Z0-9_]*$',
        r'^files\.[a-zA-Z_][a-zA-Z0-9_]*$', 
        r'^answers\.[a-zA-Z_][a-zA-Z0-9_]*$',
        r'^company_[a-zA-Z_][a-zA-Z0-9_]*$'  # Company-specific variables
    }
    
    # Dangerous patterns that should never appear in template values
    DANGEROUS_PATTERNS = {
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',                # JavaScript URLs
        r'data:text/html',            # Data URLs with HTML
        r'on\w+\s*=',                 # Event handlers (onclick, onload, etc.)
        r'expression\s*\(',           # CSS expressions
        r'@import',                   # CSS imports
        r'<iframe[^>]*>',             # Iframe tags
        r'<object[^>]*>',             # Object tags
        r'<embed[^>]*>',              # Embed tags
    }
    
    def __init__(self):
        self._compiled_allowed_patterns = [
            re.compile(pattern) for pattern in self.ALLOWED_VARIABLE_PATTERNS
        ]
        self._compiled_dangerous_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.DANGEROUS_PATTERNS
        ]
    
    def validate_template_variable(self, variable_name: str) -> bool:
        """
        Validate that a template variable name follows allowed patterns.
        
        Args:
            variable_name: The variable name to validate (e.g., 'profile.email')
            
        Returns:
            True if variable is allowed, False otherwise
        """
        return any(
            pattern.match(variable_name) 
            for pattern in self._compiled_allowed_patterns
        )
    
    def sanitize_template_value(self, value: Any, context: str = "text") -> str:
        """
        Sanitize a template variable value based on context.
        
        Args:
            value: The value to sanitize
            context: The context where value will be used ('text', 'attribute', 'url')
            
        Returns:
            Sanitized string value
            
        Raises:
            ValueError: If value contains dangerous patterns
        """
        if value is None:
            return ""
        
        str_value = str(value)
        
        # Check for dangerous patterns
        for pattern in self._compiled_dangerous_patterns:
            if pattern.search(str_value):
                dangerous_match = pattern.search(str_value).group()
                logger.warning(f"Dangerous pattern detected in template value: {dangerous_match}")
                raise ValueError(f"Template value contains dangerous pattern: {dangerous_match}")
        
        # Sanitize based on context
        if context == "attribute":
            # For HTML attributes, escape quotes and special chars
            sanitized = html.escape(str_value, quote=True)
        elif context == "url":
            # For URLs, ensure proper encoding but preserve safe characters
            sanitized = urllib.parse.quote(str_value, safe=':/?#[]@!$&\'()*+,;=.')
        else:  # text context
            # For text content, escape HTML entities
            sanitized = html.escape(str_value)
        
        return sanitized
    
    def extract_template_variables(self, template_string: str) -> Set[str]:
        """
        Extract all template variable references from a string.
        
        Args:
            template_string: String that may contain {{variable}} references
            
        Returns:
            Set of variable names found in the template
        """
        pattern = r'\{\{\s*([^}]+)\s*\}\}'
        matches = re.findall(pattern, template_string)
        return {match.strip() for match in matches}
    
    def substitute_template_variables(
        self, 
        template_string: str, 
        variables: Dict[str, Any],
        context: str = "text"
    ) -> str:
        """
        Safely substitute template variables with sanitized values.
        
        Args:
            template_string: String with {{variable}} placeholders
            variables: Dict of variable names to values
            context: Context for sanitization ('text', 'attribute', 'url')
            
        Returns:
            String with variables substituted and sanitized
            
        Raises:
            ValueError: If template contains disallowed variables or dangerous values
        """
        # Extract all variables from template
        template_vars = self.extract_template_variables(template_string)
        
        # Validate all variables are allowed
        for var_name in template_vars:
            if not self.validate_template_variable(var_name):
                raise ValueError(f"Disallowed template variable: {var_name}")
        
        # Substitute variables
        result = template_string
        for var_name in template_vars:
            if var_name in variables:
                sanitized_value = self.sanitize_template_value(
                    variables[var_name], 
                    context
                )
                result = result.replace(f"{{{{{var_name}}}}}", sanitized_value)
            else:
                logger.warning(f"Template variable not found in data: {var_name}")
                # Leave placeholder for missing variables
                pass
        
        return result


class TemplateExecutionSandbox:
    """Provides execution sandboxing for portal templates."""
    
    # Allowed domains for navigation and form submission
    ALLOWED_DOMAINS = {
        'greenhouse.io',
        'lever.co', 
        'workday.com',
        'myworkdayjobs.com',
        'bamboohr.com',
        'successfactors.com',
        'oracle.com',
        'taleo.net'
    }
    
    # Maximum execution time per template (in seconds)
    MAX_EXECUTION_TIME = 300  # 5 minutes
    
    # Maximum number of steps per template
    MAX_STEPS = 100
    
    # Maximum file upload size (in bytes)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    def __init__(self):
        self.sanitizer = TemplateSanitizer()
    
    def validate_template_security(self, template: Dict[str, Any]) -> List[str]:
        """
        Validate template security constraints.
        
        Args:
            template: Template dictionary to validate
            
        Returns:
            List of security violation messages (empty if valid)
        """
        violations = []
        
        # Check step count
        steps = template.get("steps", [])
        if len(steps) > self.MAX_STEPS:
            violations.append(f"Template has {len(steps)} steps, max allowed is {self.MAX_STEPS}")
        
        # Check start URL domain
        start_url = template.get("start", {}).get("url", "")
        if start_url and not self._is_url_allowed(start_url):
            violations.append(f"Start URL domain not in allowed list: {start_url}")
        
        # Check all template variables in steps
        for i, step in enumerate(steps):
            step_violations = self._validate_step_security(step, i)
            violations.extend(step_violations)
        
        return violations
    
    def _is_url_allowed(self, url: str) -> bool:
        """Check if a URL's domain is in the allowed list."""
        # Check if URL contains template variables - allow these
        if "{{" in url and "}}" in url:
            return True
        
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove 'www.' prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
            
            return any(
                domain == allowed or domain.endswith(f'.{allowed}')
                for allowed in self.ALLOWED_DOMAINS
            )
        except Exception:
            return False
    
    def _validate_step_security(self, step: Dict[str, Any], step_index: int) -> List[str]:
        """Validate security of a single template step."""
        violations = []
        
        # Check for script execution
        if step.get("action") == "script":
            script_content = step.get("script", "")
            if self._contains_dangerous_script(script_content):
                violations.append(f"Step {step_index}: Dangerous script content detected")
        
        # Check template variables in step values
        for field in ["value", "selector", "script"]:
            if field in step:
                field_value = str(step[field])
                template_vars = self.sanitizer.extract_template_variables(field_value)
                
                for var_name in template_vars:
                    if not self.sanitizer.validate_template_variable(var_name):
                        violations.append(f"Step {step_index}: Disallowed variable {var_name} in {field}")
        
        return violations
    
    def _contains_dangerous_script(self, script: str) -> bool:
        """Check if a script contains dangerous JavaScript patterns."""
        dangerous_js_patterns = [
            r'eval\s*\(',
            r'Function\s*\(',
            r'document\.write',
            r'innerHTML\s*=',
            r'outerHTML\s*=', 
            r'window\.',
            r'location\.',
            r'XMLHttpRequest',
            r'fetch\s*\(',
            r'import\s*\(',
            r'require\s*\(',
        ]
        
        for pattern in dangerous_js_patterns:
            if re.search(pattern, script, re.IGNORECASE):
                return True
        
        return False


# Factory functions
def create_template_sanitizer() -> TemplateSanitizer:
    """Create a new template sanitizer instance."""
    return TemplateSanitizer()


def create_execution_sandbox() -> TemplateExecutionSandbox:
    """Create a new template execution sandbox instance."""
    return TemplateExecutionSandbox()