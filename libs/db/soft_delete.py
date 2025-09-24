"""Soft Delete Base Model Implementation

Provides a base model class that supports soft deletes with deleted_at timestamps.
This addresses the audit finding about inconsistent delete strategy.
"""
from sqlalchemy import Column, DateTime
from sqlalchemy.orm import declarative_base

# Create a base model with soft delete capabilities
class SoftDeleteMixin:
    """Mixin that adds soft delete functionality to models."""
    deleted_at = Column(DateTime, nullable=True, index=True)
    
    def soft_delete(self):
        """Mark this record as deleted by setting deleted_at timestamp."""
        from datetime import datetime
        self.deleted_at = datetime.utcnow()
    
    def restore(self):
        """Restore this record by clearing deleted_at timestamp."""
        self.deleted_at = None
    
    @property
    def is_deleted(self):
        """Check if this record is soft deleted."""
        return self.deleted_at is not None

# Usage example (not implemented to avoid breaking changes):
# class Resume(Base, SoftDeleteMixin):
#     # existing fields...
#     pass
#
# # Query filters for soft delete awareness
# # Active records only: session.query(Resume).filter(Resume.deleted_at.is_(None))
# # Deleted records only: session.query(Resume).filter(Resume.deleted_at.is_not(None))

# Note: To implement soft deletes fully, would need to:
# 1. Add SoftDeleteMixin to sensitive models (Resume, Application, etc.)
# 2. Update all queries to filter deleted_at IS NULL by default
# 3. Create migration to add deleted_at columns
# 4. Update application logic to use soft_delete() instead of session.delete()