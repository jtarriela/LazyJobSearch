"""Tests for portal template validation and schema compliance."""
import json
import pytest
from pathlib import Path
from jsonschema import validate, ValidationError


class TestPortalTemplateValidation:
    """Test suite for portal template DSL validation."""

    @pytest.fixture
    def schema(self):
        """Load the portal template DSL schema."""
        schema_path = Path(__file__).parent.parent / "docs" / "portal_template_dsl.schema.json"
        with open(schema_path) as f:
            return json.load(f)

    @pytest.fixture
    def template_dir(self):
        """Get the portal templates directory."""
        return Path(__file__).parent.parent / "docs" / "examples" / "portal_templates"

    def test_greenhouse_template_validates(self, schema, template_dir):
        """Test that the Greenhouse template validates against the schema."""
        template_path = template_dir / "greenhouse_basic.json"
        with open(template_path) as f:
            template = json.load(f)
        
        # Should not raise ValidationError
        validate(template, schema)

    def test_lever_template_validates(self, schema, template_dir):
        """Test that the Lever template validates against the schema.""" 
        template_path = template_dir / "lever_basic.json"
        with open(template_path) as f:
            template = json.load(f)
        
        # Should not raise ValidationError
        validate(template, schema)

    def test_workday_template_validates(self, schema, template_dir):
        """Test that the Workday template validates against the schema."""
        template_path = template_dir / "workday_basic.json" 
        with open(template_path) as f:
            template = json.load(f)
        
        # Should not raise ValidationError
        validate(template, schema)

    def test_all_example_templates_validate(self, schema, template_dir):
        """Test that all example templates validate against the schema."""
        template_files = list(template_dir.glob("*.json"))
        assert len(template_files) >= 3, "Expected at least 3 template examples"
        
        for template_file in template_files:
            with open(template_file) as f:
                template = json.load(f)
            
            # Should not raise ValidationError
            validate(template, schema)

    def test_invalid_template_missing_required_fields(self, schema):
        """Test that templates missing required fields fail validation."""
        invalid_template = {
            "version": 1,
            "meta": {"portal": "test"},
            # Missing required 'start' and 'steps' fields
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate(invalid_template, schema)
        
        error_message = str(exc_info.value)
        assert "'start' is a required property" in error_message or "'steps' is a required property" in error_message

    def test_invalid_template_wrong_action_type(self, schema):
        """Test that templates with invalid action types fail validation."""
        invalid_template = {
            "start": {"url": "https://example.com"},
            "steps": [
                {
                    "action": "invalid_action",  # Not in allowed enum
                    "selector": ".test"
                }
            ]
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate(invalid_template, schema)
        
        error_message = str(exc_info.value)
        assert "invalid_action" in error_message

    def test_invalid_template_malformed_steps(self, schema):
        """Test that templates with malformed steps fail validation."""
        invalid_template = {
            "start": {"url": "https://example.com"},
            "steps": "not-an-array"  # Should be array
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate(invalid_template, schema)
        
        error_message = str(exc_info.value)
        assert "array" in error_message.lower() or "type" in error_message.lower()

    def test_template_validation_error_messages_are_actionable(self, schema):
        """Test that validation error messages provide actionable feedback.""" 
        invalid_template = {
            "start": {"url": "https://example.com"},
            "steps": [
                {
                    # Missing required 'action' field
                    "selector": ".test",
                    "value": "test"
                }
            ]
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate(invalid_template, schema)
        
        error_message = str(exc_info.value)
        # Should clearly indicate what's wrong and where
        assert "'action' is a required property" in error_message
        assert "steps" in error_message

    def test_template_with_conditional_logic_validates(self, schema):
        """Test that templates with conditional logic (skipIf) validate correctly."""
        template_with_conditionals = {
            "start": {"url": "https://example.com"},
            "steps": [
                {
                    "action": "type",
                    "selector": "input[name='optional']",
                    "value": "{{profile.optional_field}}",
                    "skipIf": [
                        {
                            "selector": "input[name='optional']",
                            "predicate": "notExists"
                        }
                    ]
                }
            ]
        }
        
        # Should not raise ValidationError
        validate(template_with_conditionals, schema)

    def test_template_with_error_recovery_validates(self, schema):
        """Test that templates with error recovery strategies validate correctly."""
        template_with_recovery = {
            "start": {"url": "https://example.com"},
            "steps": [
                {
                    "action": "click", 
                    "selector": ".submit"
                }
            ],
            "errorRecovery": {
                "validation_error": {
                    "steps": [
                        {
                            "action": "wait",
                            "waitFor": "timeout",
                            "timeoutMs": 2000
                        }
                    ],
                    "maxAttempts": 2
                }
            }
        }
        
        # Should not raise ValidationError
        validate(template_with_recovery, schema)

    def test_template_variable_substitution_patterns(self, schema):
        """Test that templates with various variable substitution patterns validate."""
        template_with_variables = {
            "start": {"url": "https://example.com/{{company_slug}}"},
            "steps": [
                {
                    "action": "type",
                    "selector": "input[name='email']",
                    "value": "{{profile.email}}"
                },
                {
                    "action": "upload",
                    "selector": "input[type='file']",
                    "file": "{{files.resume}}"
                },
                {
                    "action": "select",
                    "selector": "select[name='status']",
                    "value": "{{answers.veteran_status}}"
                }
            ]
        }
        
        # Should not raise ValidationError
        validate(template_with_variables, schema)