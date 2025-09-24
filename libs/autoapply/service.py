"""Auto-apply service - orchestrates job application submission using DSL templates.

This service implements the auto-apply workflow:
1. Portal template resolution and validation
2. Application profile data mapping
3. Browser automation for form submission  
4. Receipt capture and artifact storage
5. Application tracking and status updates

Based on ADR 0005 and auto-apply requirements from the gap analysis.
"""
from __future__ import annotations
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid
import json

from libs.observability import get_logger, timer, counter
from libs.db.models import Application, ApplicationProfile, Job, Portal, PortalTemplate
from libs.autoapply.template_dsl import ApplicationTemplate, DSLValidator

logger = get_logger(__name__)

@dataclass
class ApplicationRequest:
    """Request to submit a job application"""
    job_id: str
    application_profile_id: str
    portal_id: Optional[str] = None
    dry_run: bool = False

@dataclass
class ApplicationResult:
    """Result of application submission"""
    application_id: str
    status: str  # submitted, failed, pending
    external_id: Optional[str] = None
    receipt_url: Optional[str] = None
    error_message: Optional[str] = None
    artifacts: List[Dict[str, Any]] = None
    submission_time_ms: float = 0.0

@dataclass
class FieldMapping:
    """Mapping of profile data to form field"""
    field_name: str
    source_value: str
    transformed_value: Optional[str] = None
    field_type: str = "text"


class AutoApplyService:
    """Service for automated job application submission"""
    
    def __init__(self, db_session, browser_service=None, artifact_storage=None):
        self.db_session = db_session
        self.browser_service = browser_service
        self.artifact_storage = artifact_storage
        self.dsl_validator = DSLValidator()
        
    async def submit_application(self, request: ApplicationRequest) -> ApplicationResult:
        """
        Submit a job application using portal templates and browser automation.
        
        Args:
            request: Application submission request
            
        Returns:
            ApplicationResult with submission status and artifacts
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info("Starting application submission", 
                       job_id=request.job_id,
                       profile_id=request.application_profile_id)
            
            with timer("auto_apply.total_submission"):
                # Stage 1: Load and validate application data
                job, profile, portal_template = await self._load_application_data(request)
                
                if not portal_template:
                    raise ValueError("No portal template found for job")
                
                # Stage 2: Generate field mappings using DSL
                field_mappings = await self._generate_field_mappings(profile, portal_template)
                
                # Stage 3: Submit application via browser automation
                if request.dry_run:
                    logger.info("Dry run mode - skipping actual submission")
                    result = ApplicationResult(
                        application_id=str(uuid.uuid4()),
                        status="dry_run_success",
                        submission_time_ms=0.0
                    )
                else:
                    result = await self._submit_via_browser(job, field_mappings, portal_template)
                
                # Stage 4: Persist application record
                await self._persist_application(request, result)
                
                end_time = asyncio.get_event_loop().time()
                result.submission_time_ms = (end_time - start_time) * 1000
                
                counter("auto_apply.applications_submitted")
                logger.info("Application submission completed",
                           application_id=result.application_id,
                           status=result.status,
                           time_ms=result.submission_time_ms)
                
                return result
                
        except Exception as e:
            counter("auto_apply.submission_failure")
            logger.error("Application submission failed", 
                        job_id=request.job_id,
                        error=str(e))
            
            # Create failed result
            result = ApplicationResult(
                application_id=str(uuid.uuid4()),
                status="failed",
                error_message=str(e)
            )
            
            # Try to persist failed application record
            try:
                await self._persist_application(request, result)
            except Exception as persist_error:
                logger.error("Failed to persist failed application", error=str(persist_error))
            
            return result
    
    async def get_application_status(self, application_id: str) -> Dict[str, Any]:
        """Get current status of submitted application"""
        try:
            query = await self.db_session.execute("""
                SELECT a.*, j.title, j.url as job_url, ap.name as profile_name
                FROM applications a
                JOIN jobs j ON a.job_id = j.id
                JOIN application_profiles ap ON a.application_profile_id = ap.id
                WHERE a.id = %s
            """, (application_id,))
            
            row = query.fetchone()
            if not row:
                return {"error": "Application not found"}
            
            return {
                "application_id": str(row[0]),
                "status": row[4],  # status column
                "job_title": row[-3],
                "job_url": row[-2], 
                "profile_name": row[-1],
                "applied_at": row[7].isoformat() if row[7] else None,  # applied_at
                "external_id": row[5],  # external_id
            }
            
        except Exception as e:
            logger.error("Failed to get application status", 
                        application_id=application_id,
                        error=str(e))
            return {"error": str(e)}
    
    async def _load_application_data(self, request: ApplicationRequest) -> tuple:
        """Load job, profile, and portal template data"""
        try:
            # Load job
            job_query = await self.db_session.execute(
                "SELECT * FROM jobs WHERE id = %s", (request.job_id,)
            )
            job_row = job_query.fetchone()
            if not job_row:
                raise ValueError(f"Job {request.job_id} not found")
            
            # Load application profile
            profile_query = await self.db_session.execute(
                "SELECT * FROM application_profiles WHERE id = %s", 
                (request.application_profile_id,)
            )
            profile_row = profile_query.fetchone()
            if not profile_row:
                raise ValueError(f"Application profile {request.application_profile_id} not found")
            
            # Load portal template (simplified - use job's company portal config)
            template_query = await self.db_session.execute("""
                SELECT pt.* FROM portal_templates pt
                JOIN company_portal_configs cpc ON pt.portal_id = cpc.portal_id
                JOIN jobs j ON j.company_id = cpc.company_id
                WHERE j.id = %s AND pt.is_active = true
                LIMIT 1
            """, (request.job_id,))
            template_row = template_query.fetchone()
            
            return job_row, profile_row, template_row
            
        except Exception as e:
            logger.error("Failed to load application data", error=str(e))
            raise
    
    async def _generate_field_mappings(self, profile_row, template_row) -> List[FieldMapping]:
        """Generate form field mappings using DSL template"""
        try:
            if not template_row:
                return []
            
            # Parse DSL template JSON
            template_json = json.loads(template_row[3])  # dsl_json column
            application_template = ApplicationTemplate.from_dict(template_json)
            
            # Parse profile data
            profile_data = json.loads(profile_row[3]) if profile_row[3] else {}  # answers_json
            
            # Generate mappings
            field_mappings = []
            for form_field in application_template.fields:
                if form_field.mapping:
                    source_value = profile_data.get(form_field.mapping.source_field, "")
                    
                    # Apply transformation if specified
                    transformed_value = source_value
                    if form_field.mapping.transform:
                        transformed_value = self._apply_transformation(
                            source_value, 
                            form_field.mapping.transform
                        )
                    
                    mapping = FieldMapping(
                        field_name=form_field.name,
                        source_value=str(source_value),
                        transformed_value=str(transformed_value),
                        field_type=form_field.type.value
                    )
                    field_mappings.append(mapping)
            
            logger.debug("Generated field mappings", count=len(field_mappings))
            return field_mappings
            
        except Exception as e:
            logger.error("Failed to generate field mappings", error=str(e))
            return []
    
    async def _submit_via_browser(self, job_row, field_mappings: List[FieldMapping], 
                                 template_row) -> ApplicationResult:
        """Submit application using browser automation"""
        try:
            if not self.browser_service:
                raise ValueError("Browser service not available")
            
            # Navigate to job application URL
            job_url = job_row[2]  # url column
            await self.browser_service.navigate(job_url)
            
            # Fill form fields using mappings
            for mapping in field_mappings:
                try:
                    await self.browser_service.fill_field(
                        field_name=mapping.field_name,
                        value=mapping.transformed_value or mapping.source_value,
                        field_type=mapping.field_type
                    )
                except Exception as field_error:
                    logger.warning("Failed to fill field",
                                 field_name=mapping.field_name,
                                 error=str(field_error))
                    continue
            
            # Submit form
            submit_result = await self.browser_service.submit_form()
            
            # Capture receipt/confirmation
            receipt_url = await self.browser_service.get_current_url()
            
            # TODO: Extract external application ID from confirmation page
            external_id = None
            
            return ApplicationResult(
                application_id=str(uuid.uuid4()),
                status="submitted",
                external_id=external_id,
                receipt_url=receipt_url
            )
            
        except Exception as e:
            logger.error("Browser submission failed", error=str(e))
            return ApplicationResult(
                application_id=str(uuid.uuid4()),
                status="failed",
                error_message=str(e)
            )
    
    async def _persist_application(self, request: ApplicationRequest, result: ApplicationResult):
        """Persist application record to database"""
        try:
            application = Application(
                id=result.application_id,
                job_id=request.job_id,
                application_profile_id=request.application_profile_id,
                status=result.status,
                external_id=result.external_id,
                applied_at=datetime.utcnow(),
                metadata_json=json.dumps({
                    "receipt_url": result.receipt_url,
                    "error_message": result.error_message,
                    "submission_time_ms": result.submission_time_ms
                })
            )
            
            self.db_session.add(application)
            await self.db_session.commit()
            
            logger.debug("Application persisted", application_id=result.application_id)
            
        except Exception as e:
            logger.error("Application persistence failed", error=str(e))
            await self.db_session.rollback()
            raise
    
    def _apply_transformation(self, value: str, transform: str) -> str:
        """Apply transformation to field value"""
        if transform == "upper":
            return value.upper()
        elif transform == "lower":  
            return value.lower()
        elif transform == "strip":
            return value.strip()
        elif transform.startswith("format:"):
            # Simple format transformation
            format_str = transform[7:]  # Remove "format:" prefix
            try:
                return format_str.format(value=value)
            except:
                return value
        else:
            # Default: return original value
            return value


def create_auto_apply_service(db_session, browser_service=None, artifact_storage=None) -> AutoApplyService:
    """Factory function to create auto-apply service"""
    return AutoApplyService(
        db_session=db_session,
        browser_service=browser_service,
        artifact_storage=artifact_storage
    )