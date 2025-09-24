# Portal Template Version Management

This document outlines the strategy for managing portal template versions, upgrades, rollbacks, and maintaining backward compatibility in the LazyJobSearch auto-apply system.

## Overview

Portal templates evolve over time as ATS platforms change their UI, add new features, or modify their application flows. Effective version management ensures:

- **Backward Compatibility**: Existing applications continue to work
- **Smooth Upgrades**: New features can be adopted gradually
- **Rollback Safety**: Problems can be quickly reverted
- **Testing Isolation**: New templates can be tested before deployment

## Version Schema

### Template Version Format

Templates use semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes that require user action
- **MINOR**: New features, backward compatible  
- **PATCH**: Bug fixes and minor improvements

Example: `"version": "2.1.0"`

### Version Metadata

Each template includes version metadata:

```json
{
  "version": "2.1.0",
  "meta": {
    "portal": "greenhouse",
    "variant": "basic",
    "created_at": "2024-01-15T10:00:00Z",
    "deprecated": false,
    "supported_until": "2025-01-15T00:00:00Z",
    "changelog": "Added support for optional cover letter field",
    "breaking_changes": [],
    "migration_guide": ""
  }
}
```

## Database Schema

### Portal Templates Table

```sql
CREATE TABLE portal_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portal_id UUID REFERENCES portals(id),
    template_name TEXT NOT NULL,
    template_json JSONB NOT NULL,
    version TEXT NOT NULL,
    is_active BOOLEAN DEFAULT true,
    is_deprecated BOOLEAN DEFAULT false,
    supported_until TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    
    -- Ensure only one active version per portal variant
    UNIQUE(portal_id, template_name, version),
    
    -- Index for efficient version queries
    INDEX idx_portal_templates_version ON (portal_id, template_name, version),
    INDEX idx_portal_templates_active ON (portal_id, is_active) WHERE is_active = true
);
```

### Template Versions Table

```sql 
CREATE TABLE template_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_id UUID REFERENCES portal_templates(id),
    version TEXT NOT NULL,
    template_json JSONB NOT NULL,
    changelog TEXT,
    breaking_changes TEXT[],
    created_at TIMESTAMPTZ DEFAULT now(),
    created_by UUID, -- Admin user who created version
    
    UNIQUE(template_id, version)
);
```

## Version Management Operations

### 1. Template Creation

```python
from libs.autoapply.template_versioning import TemplateVersionManager

manager = TemplateVersionManager()

# Create new template
template_id = manager.create_template(
    portal_id="greenhouse-uuid",
    name="greenhouse_basic",
    template_json=template_data,
    version="1.0.0"
)
```

### 2. Version Upgrade

```python
# Minor version upgrade (backward compatible)
manager.upgrade_template(
    template_id=template_id,
    new_version="1.1.0", 
    template_json=updated_template,
    changelog="Added support for LinkedIn profile field",
    breaking_changes=[]
)

# Major version upgrade (breaking changes)
manager.upgrade_template(
    template_id=template_id,
    new_version="2.0.0",
    template_json=new_template,
    changelog="Restructured step flow for new Greenhouse UI",
    breaking_changes=[
        "Changed selector for email field from #email to input[name='email']",
        "Removed deprecated 'next' action, use 'click' on specific buttons"
    ]
)
```

### 3. Template Selection

```python
# Get latest active version
template = manager.get_active_template("greenhouse", "basic")

# Get specific version
template = manager.get_template_version("greenhouse", "basic", "1.2.0")

# Get all versions for template
versions = manager.list_template_versions("greenhouse", "basic")
```

### 4. Rollback Operations

```python
# Rollback to previous version
manager.rollback_template(template_id, target_version="1.2.0")

# Emergency rollback (deactivate current, activate previous)
manager.emergency_rollback(template_id)
```

## Upgrade Strategies

### 1. Blue-Green Deployment

- **Blue**: Current active template version
- **Green**: New template version being tested
- Switch traffic gradually: 10% → 50% → 100%

```python
# Gradual rollout
manager.set_traffic_split(
    template_id=template_id,
    versions={
        "1.2.0": 80,  # 80% of traffic
        "1.3.0": 20   # 20% of traffic (new version)
    }
)
```

### 2. Canary Releases

Deploy new version to subset of companies first:

```python
# Deploy to specific companies only
manager.deploy_canary(
    template_id=template_id,
    version="2.0.0",
    company_ids=["company-1", "company-2"],
    duration_hours=24
)
```

### 3. Feature Flags

Use template conditionals for gradual feature rollout:

```json
{
  "steps": [
    {
      "id": "new_field",
      "action": "type",
      "selector": "input[name='new_field']",
      "value": "{{profile.new_field}}",
      "skipIf": [
        {
          "selector": "body[data-feature-flag='new_field']",
          "predicate": "notExists"
        }
      ]
    }
  ]
}
```

## Testing Framework

### 1. Template Compatibility Testing

```python
class TemplateCompatibilityTest:
    def test_version_upgrade_compatibility(self):
        """Test that new version is compatible with existing data"""
        old_template = self.load_template("1.2.0")
        new_template = self.load_template("1.3.0") 
        
        # Test same profile data works with both versions
        self.assert_compatible(old_template, new_template, test_profile)
    
    def test_breaking_changes_detected(self):
        """Test that breaking changes are properly identified"""
        changes = self.detect_breaking_changes("1.2.0", "2.0.0")
        assert len(changes) > 0
```

### 2. Regression Testing

```bash
# Run regression tests against multiple template versions
ljs template test --version 1.2.0 --profile test_data/profile.json
ljs template test --version 1.3.0 --profile test_data/profile.json

# Compare results
ljs template compare --versions 1.2.0,1.3.0 --metrics success_rate,completion_time
```

## Monitoring & Alerting

### 1. Template Performance Metrics

Track key metrics per template version:

- **Success Rate**: % of successful applications
- **Error Rate**: % of applications with errors
- **Completion Time**: Average time per application
- **Selector Drift**: Rate of selector match failures

```python
class TemplateMetrics:
    def record_application(self, template_version, status, duration, errors):
        """Record metrics for template execution"""
        self.metrics.increment(f"template.{template_version}.attempts")
        
        if status == "success":
            self.metrics.increment(f"template.{template_version}.success")
        else:
            self.metrics.increment(f"template.{template_version}.errors")
            
        self.metrics.histogram(f"template.{template_version}.duration", duration)
```

### 2. Automated Rollback Triggers

```python
class AutoRollbackManager:
    def check_rollback_conditions(self, template_version):
        """Check if template should be auto-rolled back"""
        metrics = self.get_recent_metrics(template_version, hours=1)
        
        if metrics.error_rate > 0.20:  # 20% error rate
            self.trigger_rollback(template_version, reason="High error rate")
            
        if metrics.selector_failures > 0.50:  # 50% selector failures  
            self.trigger_rollback(template_version, reason="Selector drift detected")
```

## CLI Commands

### Version Management Commands

```bash
# List all template versions
ljs template versions --portal greenhouse --name basic

# Create new template version
ljs template create --portal greenhouse --name basic --version 1.3.0 --file template.json

# Upgrade existing template
ljs template upgrade --template-id uuid --version 1.3.0 --file updated_template.json

# Rollback template
ljs template rollback --template-id uuid --target-version 1.2.0

# Set active version
ljs template activate --template-id uuid --version 1.3.0

# Deprecate old version
ljs template deprecate --template-id uuid --version 1.1.0 --until 2024-12-31
```

### Testing Commands

```bash
# Test template version
ljs template test --version 1.3.0 --dry-run --profile test_profile.json

# Compare template versions
ljs template compare --versions 1.2.0,1.3.0 --show-diff

# Validate template upgrade
ljs template validate-upgrade --from 1.2.0 --to 1.3.0
```

## Migration Guide

### For Template Authors

1. **Version Increments**:
   - Patch: Bug fixes, selector updates
   - Minor: New optional fields, enhanced error handling  
   - Major: Changed required fields, restructured flow

2. **Breaking Change Documentation**:
   - Document all breaking changes in changelog
   - Provide migration steps for users
   - Include compatibility matrix

3. **Testing Requirements**:
   - Test against multiple company portals
   - Validate with various profile data sets
   - Verify error handling and recovery

### For System Administrators

1. **Deployment Process**:
   - Stage new templates in test environment
   - Run automated compatibility tests
   - Deploy with traffic splitting for gradual rollout

2. **Monitoring Checklist**:
   - Watch error rates for first 24 hours
   - Monitor selector match failures
   - Check application completion rates

3. **Rollback Procedures**:
   - Have rollback plan ready before deployment
   - Document rollback triggers and thresholds
   - Test rollback procedures in staging

## Future Enhancements

1. **Automated Template Updates**: AI-powered template maintenance
2. **Cross-Portal Learning**: Apply successful patterns across portals
3. **A/B Testing Framework**: Built-in experiment management
4. **Template Marketplace**: Community-contributed templates
5. **Visual Template Editor**: GUI for template creation/editing

## Security Considerations

1. **Version Integrity**: Sign template versions to prevent tampering
2. **Access Control**: Role-based permissions for version management
3. **Audit Logging**: Track all version changes and deployments
4. **Sandboxed Testing**: Isolated testing environment for new versions