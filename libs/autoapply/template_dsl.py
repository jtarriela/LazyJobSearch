"""Template DSL for auto-apply system

Defines a domain-specific language for describing job application forms
and mapping candidate data to form fields across different portals.
"""
from __future__ import annotations
import logging
import json
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum

logger = logging.getLogger(__name__)

class FieldType(Enum):
    """Types of form fields"""
    TEXT = "text"
    EMAIL = "email"
    PHONE = "phone"
    SELECT = "select"
    RADIO = "radio"
    CHECKBOX = "checkbox"
    TEXTAREA = "textarea"
    FILE_UPLOAD = "file_upload"
    DATE = "date"
    URL = "url"
    HIDDEN = "hidden"

class ValidationRule(Enum):
    """Validation rules for form fields"""
    REQUIRED = "required"
    EMAIL_FORMAT = "email_format"
    PHONE_FORMAT = "phone_format"
    URL_FORMAT = "url_format"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    REGEX = "regex"

@dataclass
class FieldMapping:
    """Maps a candidate data field to a form field"""
    source_field: str  # Field name in candidate profile
    transform: Optional[str] = None  # Transformation to apply
    default_value: Optional[str] = None  # Default if source is empty
    condition: Optional[str] = None  # Condition for when to use this mapping

@dataclass
class FormField:
    """Definition of a form field"""
    field_id: str
    field_name: str
    field_type: FieldType
    label: str
    is_required: bool = False
    options: Optional[List[str]] = None  # For select/radio fields
    validation_rules: List[ValidationRule] = None
    mapping: Optional[FieldMapping] = None
    css_selector: Optional[str] = None
    xpath: Optional[str] = None
    placeholder: Optional[str] = None
    help_text: Optional[str] = None
    
    def __post_init__(self):
        if self.validation_rules is None:
            self.validation_rules = []

@dataclass
class FormSection:
    """A section of a job application form"""
    section_id: str
    section_name: str
    fields: List[FormField]
    is_optional: bool = False
    condition: Optional[str] = None  # When this section should be filled

@dataclass
class ApplicationTemplate:
    """Complete template for a job application portal"""
    template_id: str
    portal_name: str
    portal_version: str
    form_sections: List[FormSection]
    submission_config: Dict[str, Any]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class DSLValidator:
    """Validates template DSL syntax and structure"""
    
    def validate_template(self, template: ApplicationTemplate) -> List[str]:
        """Validate an application template
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Basic template validation
        if not template.template_id:
            errors.append("Template ID is required")
        
        if not template.portal_name:
            errors.append("Portal name is required")
        
        if not template.form_sections:
            errors.append("Template must have at least one form section")
        
        # Validate each section
        for section in template.form_sections:
            section_errors = self._validate_section(section)
            errors.extend([f"Section '{section.section_id}': {err}" for err in section_errors])
        
        # Check for duplicate field IDs
        all_field_ids = []
        for section in template.form_sections:
            for field in section.fields:
                if field.field_id in all_field_ids:
                    errors.append(f"Duplicate field ID: {field.field_id}")
                all_field_ids.append(field.field_id)
        
        return errors
    
    def _validate_section(self, section: FormSection) -> List[str]:
        """Validate a form section"""
        errors = []
        
        if not section.section_id:
            errors.append("Section ID is required")
        
        if not section.fields:
            errors.append("Section must have at least one field")
        
        # Validate each field
        for field in section.fields:
            field_errors = self._validate_field(field)
            errors.extend([f"Field '{field.field_id}': {err}" for err in field_errors])
        
        return errors
    
    def _validate_field(self, field: FormField) -> List[str]:
        """Validate a form field"""
        errors = []
        
        if not field.field_id:
            errors.append("Field ID is required")
        
        if not field.field_name:
            errors.append("Field name is required") 
        
        # Validate field type specific requirements
        if field.field_type in [FieldType.SELECT, FieldType.RADIO]:
            if not field.options:
                errors.append(f"{field.field_type.value} field must have options")
        
        # Validate selector presence
        if not field.css_selector and not field.xpath:
            errors.append("Field must have either css_selector or xpath")
        
        # Validate mapping if present
        if field.mapping:
            mapping_errors = self._validate_mapping(field.mapping)
            errors.extend([f"Mapping: {err}" for err in mapping_errors])
        
        return errors
    
    def _validate_mapping(self, mapping: FieldMapping) -> List[str]:
        """Validate a field mapping"""
        errors = []
        
        if not mapping.source_field:
            errors.append("Source field is required")
        
        # Validate transform syntax if present
        if mapping.transform:
            if not self._is_valid_transform(mapping.transform):
                errors.append(f"Invalid transform: {mapping.transform}")
        
        return errors
    
    def _is_valid_transform(self, transform: str) -> bool:
        """Check if transform syntax is valid"""
        # Basic validation - in practice, would be more sophisticated
        valid_transforms = ['upper', 'lower', 'title', 'strip', 'format_phone', 'format_date']
        return any(t in transform for t in valid_transforms)

class FieldMapper:
    """Maps candidate data to form fields using the DSL"""
    
    def __init__(self):
        self.transforms = {
            'upper': lambda x: str(x).upper(),
            'lower': lambda x: str(x).lower(), 
            'title': lambda x: str(x).title(),
            'strip': lambda x: str(x).strip(),
            'format_phone': self._format_phone,
            'format_date': self._format_date,
        }
    
    def map_candidate_to_form(
        self,
        candidate_data: Dict[str, Any],
        template: ApplicationTemplate
    ) -> Dict[str, Any]:
        """Map candidate data to form fields using template
        
        Args:
            candidate_data: Candidate profile data
            template: Application template
            
        Returns:
            Dictionary of field_id -> value mappings
        """
        mapped_data = {}
        
        for section in template.form_sections:
            # Check section condition
            if section.condition and not self._evaluate_condition(section.condition, candidate_data):
                continue
            
            for field in section.fields:
                # Skip if field has no mapping
                if not field.mapping:
                    continue
                
                # Check field condition
                if field.mapping.condition and not self._evaluate_condition(field.mapping.condition, candidate_data):
                    continue
                
                # Get mapped value
                value = self._map_field_value(field, candidate_data)
                if value is not None:
                    mapped_data[field.field_id] = value
        
        return mapped_data
    
    def _map_field_value(self, field: FormField, candidate_data: Dict[str, Any]) -> Any:
        """Map a single field value"""
        mapping = field.mapping
        
        # Get source value
        source_value = self._get_nested_value(candidate_data, mapping.source_field)
        
        # Use default if source is empty
        if not source_value and mapping.default_value:
            source_value = mapping.default_value
        
        # Apply transformation if specified
        if source_value and mapping.transform:
            source_value = self._apply_transform(source_value, mapping.transform)
        
        # Validate against field type and rules
        validated_value = self._validate_field_value(source_value, field)
        
        return validated_value
    
    def _get_nested_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = field_path.split('.')
        value = data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _apply_transform(self, value: Any, transform: str) -> Any:
        """Apply transformation to value"""
        try:
            # Handle simple transforms
            if transform in self.transforms:
                return self.transforms[transform](value)
            
            # Handle transforms with parameters (e.g., "format({template})")
            if '(' in transform:
                func_name = transform.split('(')[0]
                if func_name in self.transforms:
                    return self.transforms[func_name](value)
            
            # Fallback to original value
            return value
            
        except Exception as e:
            logger.warning(f"Transform '{transform}' failed: {e}")
            return value
    
    def _evaluate_condition(self, condition: str, data: Dict[str, Any]) -> bool:
        """Evaluate a condition expression"""
        try:
            # Simple condition evaluation - in practice, would use safe eval
            # For now, support basic existence checks
            if condition.startswith('has_'):
                field_name = condition[4:]  # Remove 'has_' prefix
                return self._get_nested_value(data, field_name) is not None
            
            if '==' in condition:
                field, expected = condition.split('==', 1)
                field = field.strip()
                expected = expected.strip().strip('"\'')
                actual = self._get_nested_value(data, field)
                return str(actual) == expected
            
            # Default to true for unknown conditions
            return True
            
        except Exception as e:
            logger.warning(f"Condition evaluation failed '{condition}': {e}")
            return True
    
    def _validate_field_value(self, value: Any, field: FormField) -> Any:
        """Validate value against field rules"""
        if value is None:
            return None
        
        # Convert to string for most validations
        str_value = str(value)
        
        # Apply validation rules
        for rule in field.validation_rules:
            if rule == ValidationRule.REQUIRED and not str_value:
                logger.warning(f"Required field {field.field_id} is empty")
                return None
            
            elif rule == ValidationRule.EMAIL_FORMAT:
                if not self._is_valid_email(str_value):
                    logger.warning(f"Invalid email format for field {field.field_id}: {str_value}")
                    return None
            
            elif rule == ValidationRule.PHONE_FORMAT:
                if not self._is_valid_phone(str_value):
                    logger.warning(f"Invalid phone format for field {field.field_id}: {str_value}")
                    return None
        
        # Handle field type specific validation
        if field.field_type == FieldType.SELECT and field.options:
            if str_value not in field.options:
                # Try to find closest match
                closest_match = self._find_closest_option(str_value, field.options)
                if closest_match:
                    logger.info(f"Mapped '{str_value}' to closest option '{closest_match}' for field {field.field_id}")
                    return closest_match
                else:
                    logger.warning(f"Value '{str_value}' not in options for field {field.field_id}")
                    return None
        
        return value
    
    def _format_phone(self, phone: str) -> str:
        """Format phone number"""
        # Simple US phone formatting
        digits = re.sub(r'[^\d]', '', str(phone))
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits[0] == '1':
            return f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        return phone
    
    def _format_date(self, date_str: str) -> str:
        """Format date string"""
        # Basic date formatting - would use proper date parsing in practice
        return str(date_str)
    
    def _is_valid_email(self, email: str) -> bool:
        """Basic email validation"""
        email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return bool(re.match(email_pattern, email))
    
    def _is_valid_phone(self, phone: str) -> bool:
        """Basic phone validation"""
        digits = re.sub(r'[^\d]', '', phone)
        return len(digits) in [10, 11]
    
    def _find_closest_option(self, value: str, options: List[str]) -> Optional[str]:
        """Find closest matching option"""
        value_lower = value.lower()
        
        # Exact match (case insensitive)
        for option in options:
            if option.lower() == value_lower:
                return option
        
        # Contains match
        for option in options:
            if value_lower in option.lower() or option.lower() in value_lower:
                return option
        
        return None

class TemplateBuilder:
    """Builder for creating application templates"""
    
    def __init__(self):
        self.template = None
        self.current_section = None
    
    def create_template(self, template_id: str, portal_name: str, portal_version: str = "1.0") -> 'TemplateBuilder':
        """Start building a new template"""
        self.template = ApplicationTemplate(
            template_id=template_id,
            portal_name=portal_name,
            portal_version=portal_version,
            form_sections=[],
            submission_config={}
        )
        return self
    
    def add_section(self, section_id: str, section_name: str, is_optional: bool = False) -> 'TemplateBuilder':
        """Add a form section"""
        if not self.template:
            raise ValueError("Must create template first")
        
        self.current_section = FormSection(
            section_id=section_id,
            section_name=section_name,
            fields=[],
            is_optional=is_optional
        )
        self.template.form_sections.append(self.current_section)
        return self
    
    def add_field(
        self,
        field_id: str,
        field_name: str,
        field_type: FieldType,
        css_selector: str,
        **kwargs
    ) -> 'TemplateBuilder':
        """Add a field to the current section"""
        if not self.current_section:
            raise ValueError("Must add section first")
        
        field = FormField(
            field_id=field_id,
            field_name=field_name,
            field_type=field_type,
            css_selector=css_selector,
            label=kwargs.get('label', field_name),
            is_required=kwargs.get('is_required', False),
            options=kwargs.get('options'),
            validation_rules=kwargs.get('validation_rules', []),
            mapping=kwargs.get('mapping'),
            xpath=kwargs.get('xpath'),
            placeholder=kwargs.get('placeholder'),
            help_text=kwargs.get('help_text')
        )
        
        self.current_section.fields.append(field)
        return self
    
    def set_submission_config(self, config: Dict[str, Any]) -> 'TemplateBuilder':
        """Set submission configuration"""
        if not self.template:
            raise ValueError("Must create template first")
        
        self.template.submission_config = config
        return self
    
    def build(self) -> ApplicationTemplate:
        """Build and validate the template"""
        if not self.template:
            raise ValueError("No template to build")
        
        # Validate template
        validator = DSLValidator()
        errors = validator.validate_template(self.template)
        
        if errors:
            raise ValueError(f"Template validation failed: {'; '.join(errors)}")
        
        return self.template

def create_template_builder() -> TemplateBuilder:
    """Factory function to create template builder"""
    return TemplateBuilder()

def create_field_mapper() -> FieldMapper:
    """Factory function to create field mapper"""
    return FieldMapper()