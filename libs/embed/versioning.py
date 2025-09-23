"""Embedding Version Management (ADR 0006)

Provides a lightweight manager to fetch the active version, stamp new embeddings,
mark legacy rows for re-embedding, and iterate progressive migration batches.

NOTE: Real DB session wiring & vector handling will be added later.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

@dataclass
class EmbeddingVersionInfo:
    version_id: str
    model_name: str
    dimensions: int
    created_at: datetime
    deprecated_at: Optional[datetime] = None

class EmbeddingVersionManager:
    def __init__(self, session):
        self.session = session

    def get_active_version(self) -> EmbeddingVersionInfo:
        # Placeholder: query embedding_versions where deprecated_at is null order by created_at desc limit 1
        return EmbeddingVersionInfo("v1.0", "text-embedding-3-large", 3072, datetime.utcnow())

    def stamp_embedding_metadata(self, row, version: EmbeddingVersionInfo):
        row.embedding_version = version.version_id
        row.embedding_model = version.model_name

    def mark_legacy_for_reembedding(self, legacy_versions: Iterable[str]) -> int:
        # Execute update setting needs_reembedding=true where embedding_version in legacy_versions
        # Return count (placeholder)
        return 0

    def next_reembedding_batch(self, table, batch_size: int = 500):
        # Query rows with needs_reembedding true limit batch_size
        return []

    def mark_reembedded(self, row, new_version: EmbeddingVersionInfo):
        row.embedding_version = new_version.version_id
        row.embedding_model = new_version.model_name
        row.needs_reembedding = False
