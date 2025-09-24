"""Jobs package for LazyJobSearch"""
from .ingestion import JobIngestionService, create_job_ingestion_service, CrawledJob, ProcessedJob, IngestionStats

__all__ = [
    'JobIngestionService',
    'create_job_ingestion_service', 
    'CrawledJob',
    'ProcessedJob', 
    'IngestionStats'
]