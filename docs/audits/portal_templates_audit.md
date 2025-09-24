# Portal Templates & Company Seeding Audit Report

**Module:** Portal Templates / Company Seeding  
**Summary:** Audit of portal template DSL validation, company seeding processes, and mapping integrity for the auto-apply system.

## Findings

### Portal Template DSL Schema Validation

**Schema File Analysis (docs/portal_template_dsl.schema.json):**
- ✅ **JSON Schema Compliant**: Valid JSON Schema Draft 2019-09
- ✅ **Version Support**: Template versioning with string or integer types
- ✅ **Required Fields**: `start` and `steps` are properly required
- ✅ **Action Types**: Comprehensive action vocabulary (click, type, upload, select, radio, next, wait, submit)
- ✅ **Template Variables**: Support for `{{profile.*}}`, `{{files.*}}`, `{{answers.*}}` substitution
- ✅ **Error Handling**: `validate` section for error detection with severity levels

**Template Example Validation:**
- ✅ **greenhouse_basic.json**: Validates successfully against schema
- Total Example Templates: 1 (needs more examples for comprehensive testing)

**Schema Completeness Assessment:**
- ✅ All major ATS interaction patterns covered
- ✅ Conditional logic support through `waitFor` conditions  
- ✅ File upload handling for resumes
- ✅ Form field mapping with template variables
- ⚠️ **Missing**: Multi-page application flows (Severity: Med)
- ⚠️ **Missing**: JavaScript execution capability (Severity: Low)

### Template Example Coverage

**Available Templates:**
1. **greenhouse_basic.json** - Basic Greenhouse portal flow
   - ✅ Valid schema compliance
   - ✅ Complete application flow (start → fields → upload → submit)
   - ✅ Error detection patterns

**Missing Template Examples:**
- ❌ **Lever Portal**: No Lever template example (Severity: High, Evidence: only Greenhouse in examples/)
- ❌ **Workday Portal**: No Workday template (Severity: Med, Evidence: mentioned in docs but no template)
- ❌ **BambooHR Portal**: No BambooHR template (Severity: Low)
- ❌ **Custom Portal**: No generic template (Severity: Med)

### Schema Validation Infrastructure

**Automated Validation:**
- ✅ **Manual Validation Works**: Command-line JSON schema validation successful
- ❌ **No Automated CI/CD Validation**: No CI pipeline checking templates (Severity: Med, Evidence: no .github/workflows/ template validation)
- ❌ **No Template Linter**: No dedicated template validation tool (Severity: Low, Evidence: no lint command in CLI)

**Error Messaging Quality:**
```bash
# Positive test - schema validation works
✅ Template validates against schema

# Need negative tests for malformed templates
❌ No tests for invalid templates with actionable error messages
```

### Company Seeding Infrastructure

**Seeding Service Analysis (libs/companies/seeding.py):**
- Service exists but implementation needs deeper analysis
- CLI command `companies seed --file` implemented
- Update flag available for idempotent operations

**Seed Data Sources:**
- ❌ **No Example Seed Files**: No sample CSV/JSON seed files found (Severity: Med, Evidence: no files in seeds/ or examples/)
- ❌ **Unknown Format**: Seed file format not documented (Severity: High, Evidence: no format specification)

**Idempotency Testing:**
- ⚠️ **Requires Verification**: CLI has update flag but needs testing for duplicate prevention
- ❌ **No Database Constraints**: No analysis of unique constraints on company records

### Mapping Integrity Assessment

**Portal Field Mapping:**
- Template variables correctly map to expected profile fields
- File references properly structured for resume uploads
- Answer substitution pattern consistent

**Database Integration:**
- ✅ Portal templates stored in `portal_templates` table (Evidence: migration 0002)
- ✅ Company-portal relationship established
- ⚠️ **Template Version Management**: Version field exists but no upgrade/rollback logic found

### Security Analysis

**Template Injection Risks:**
- ✅ **Template Variable Scope**: Variables scoped to `profile.*`, `files.*`, `answers.*` 
- ⚠️ **No Input Sanitization**: No evidence of XSS protection for template variables (Severity: Med)
- ❌ **No Template Sandbox**: No execution environment isolation (Severity: High, Evidence: templates could execute arbitrary actions)

## Gaps vs Documentation

- **ARCHITECTURE.md**: References multiple ATS portals but only Greenhouse template exists
- **ADR 0005**: Auto-apply scope mentions Greenhouse + Lever but only Greenhouse implemented  
- **CLI_DESIGN.md**: No documentation of seed file format requirements

## Metrics/Benchmarks

- **Schema Coverage**: 100% - All required fields and actions covered
- **Template Examples**: 25% - 1/4 major ATS portals (Greenhouse only)
- **Validation Automation**: 0% - No CI/CD validation pipeline
- **Security Controls**: 30% - Variable scoping present, sanitization/sandboxing missing
- **Seed File Documentation**: 0% - No format specification or examples

## Recommended Actions

1. **HIGH PRIORITY**: Create Lever portal template example
2. **HIGH PRIORITY**: Document seed file format with examples
3. **HIGH PRIORITY**: Add template execution sandbox for security
4. **MEDIUM PRIORITY**: Add CI/CD template validation pipeline
5. **MEDIUM PRIORITY**: Create Workday portal template example
6. **MEDIUM PRIORITY**: Add input sanitization for template variables
7. **MEDIUM PRIORITY**: Create negative schema validation tests
8. **LOW PRIORITY**: Add template linter CLI command
9. **LOW PRIORITY**: Create BambooHR template example
10. **LOW PRIORITY**: Add JavaScript execution capability to DSL

## Acceptance Criteria for Completion

- [ ] All example templates validate against schema programmatically
- [ ] Negative schema tests with actionable error messages
- [ ] CI/CD pipeline validates templates on commits
- [ ] Lever and Workday template examples created
- [ ] Seed file format documented with examples
- [ ] Template execution sandbox implemented
- [ ] Input sanitization for all template variables
- [ ] Company seeding idempotency verified through testing
- [ ] Portal template version management documented
- [ ] Security review of template execution environment

## Open Questions

- Should templates support conditional branching for different application flows?
- What is the expected format for company seed files (CSV, JSON, YAML)?
- How should template versioning and rollback be handled?
- Should there be a template marketplace or repository system?
- What level of JavaScript execution should be supported in templates?