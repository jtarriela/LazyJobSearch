"""Embedding Version Management (ADR 0006)

Advanced implementation with progressive migration, performance optimization,
and vector similarity preservation strategies.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional, List, Dict, Tuple, Any
import hashlib
import json
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingVersionInfo:
    version_id: str
    model_name: str
    dimensions: int
    created_at: datetime
    deprecated_at: Optional[datetime] = None
    compatibility_score: Optional[float] = None  # Similarity to previous version
    migration_progress: Optional[Dict[str, Any]] = None

@dataclass
class MigrationBatch:
    """Batch of items to re-embed during migration"""
    table_name: str
    batch_id: str
    items: List[Dict[str, Any]]
    total_tokens: int
    estimated_cost: float

@dataclass
class MigrationProgress:
    """Track progress of embedding migration"""
    source_version: str
    target_version: str
    total_items: int
    completed_items: int
    failed_items: int
    started_at: datetime
    estimated_completion: Optional[datetime] = None
    error_summary: Optional[Dict[str, int]] = None

class EmbeddingCompatibilityAnalyzer:
    """Analyze compatibility between embedding versions"""
    
    def __init__(self, sample_size: int = 1000):
        self.sample_size = sample_size
        
    async def analyze_compatibility(self, old_version: str, new_version: str, 
                                  session) -> float:
        """
        Compute compatibility score between embedding versions by comparing
        similarity preservation on a sample of text chunks.
        
        Returns: Compatibility score (0-1, higher = more compatible)
        """
        try:
            # Get sample of existing embeddings from old version
            sample_chunks = await self._get_sample_chunks(old_version, session)
            
            if len(sample_chunks) < 10:
                logger.warning("Insufficient sample for compatibility analysis")
                return 0.5  # Default neutral score
            
            # Re-embed sample with new version
            old_embeddings = []
            new_embeddings = []
            
            for chunk in sample_chunks:
                old_emb = chunk['embedding']
                new_emb = await self._embed_with_version(chunk['text'], new_version)
                
                old_embeddings.append(old_emb)
                new_embeddings.append(new_emb)
            
            # Compare similarity preservation
            compatibility = self._compute_similarity_preservation(
                old_embeddings, new_embeddings
            )
            
            logger.info(f"Compatibility {old_version} -> {new_version}: {compatibility:.3f}")
            return compatibility
            
        except Exception as e:
            logger.error(f"Compatibility analysis failed: {e}")
            return 0.5
            
    async def _get_sample_chunks(self, version: str, session) -> List[Dict]:
        """Get random sample of chunks with embeddings from given version"""
        # Sample from both resume and job chunks for diversity
        resume_sample = await session.execute("""
            SELECT chunk_text, embedding 
            FROM resume_chunks 
            WHERE embedding_version = %s 
            ORDER BY RANDOM() 
            LIMIT %s
        """, (version, self.sample_size // 2))
        
        job_sample = await session.execute("""
            SELECT chunk_text, embedding
            FROM job_chunks
            WHERE embedding_version = %s
            ORDER BY RANDOM()
            LIMIT %s  
        """, (version, self.sample_size // 2))
        
        all_samples = []
        for row in resume_sample.fetchall() + job_sample.fetchall():
            all_samples.append({
                'text': row[0],
                'embedding': row[1]
            })
            
        return all_samples
        
    async def _embed_with_version(self, text: str, version: str) -> List[float]:
        """Embed text with specific model version (placeholder)"""
        # This would call the actual embedding service
        # For now, return dummy embedding
        return [0.0] * 1536  # Placeholder
        
    def _compute_similarity_preservation(self, old_embeddings: List[List[float]], 
                                       new_embeddings: List[List[float]]) -> float:
        """
        Compute how well the new embeddings preserve similarity relationships
        from the old embeddings using rank correlation.
        """
        try:
            import numpy as np
            from scipy.stats import spearmanr
            
            old_emb = np.array(old_embeddings)
            new_emb = np.array(new_embeddings)
            
            # Compute pairwise similarities for both versions
            old_similarities = np.dot(old_emb, old_emb.T)
            new_similarities = np.dot(new_emb, new_emb.T)
            
            # Flatten upper triangular matrices (excluding diagonal)
            old_sim_flat = old_similarities[np.triu_indices_from(old_similarities, k=1)]
            new_sim_flat = new_similarities[np.triu_indices_from(new_similarities, k=1)]
            
            # Compute Spearman correlation
            correlation, _ = spearmanr(old_sim_flat, new_sim_flat)
            
            # Convert correlation to 0-1 compatibility score
            compatibility = (correlation + 1) / 2
            return max(0.0, min(1.0, compatibility))
            
        except ImportError:
            logger.warning("scipy not available, using cosine similarity fallback")
            return self._fallback_compatibility_score(old_embeddings, new_embeddings)
        except Exception as e:
            logger.error(f"Similarity preservation calculation failed: {e}")
            return 0.5
            
    def _fallback_compatibility_score(self, old_embeddings: List[List[float]], 
                                    new_embeddings: List[List[float]]) -> float:
        """Fallback compatibility calculation without scipy"""
        try:
            # Simple average cosine similarity between corresponding embeddings
            similarities = []
            
            for old_emb, new_emb in zip(old_embeddings, new_embeddings):
                # Cosine similarity
                dot_product = sum(a * b for a, b in zip(old_emb, new_emb))
                old_norm = sum(a * a for a in old_emb) ** 0.5
                new_norm = sum(b * b for b in new_emb) ** 0.5
                
                if old_norm > 0 and new_norm > 0:
                    similarity = dot_product / (old_norm * new_norm)
                    similarities.append(abs(similarity))  # Absolute value
                    
            return sum(similarities) / len(similarities) if similarities else 0.5
            
        except Exception as e:
            logger.error(f"Fallback compatibility calculation failed: {e}")
            return 0.5

class ProgressiveMigrationManager:
    """Manage progressive re-embedding with performance optimization"""
    
    def __init__(self, session, embedding_service, batch_size: int = 500,
                 max_concurrent_batches: int = 3, cost_limit_usd: float = 100.0):
        self.session = session
        self.embedding_service = embedding_service
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.cost_limit_usd = cost_limit_usd
        
        # Progress tracking
        self.migration_stats = defaultdict(int)
        self.error_log = []
        
    async def start_migration(self, source_version: str, target_version: str) -> str:
        """Start progressive migration with cost and performance controls"""
        migration_id = self._generate_migration_id(source_version, target_version)
        
        try:
            # Estimate migration scope and cost
            scope = await self._estimate_migration_scope(source_version)
            estimated_cost = self._estimate_migration_cost(scope)
            
            if estimated_cost > self.cost_limit_usd:
                raise ValueError(f"Migration cost ${estimated_cost:.2f} exceeds limit ${self.cost_limit_usd:.2f}")
                
            # Initialize migration progress
            progress = MigrationProgress(
                source_version=source_version,
                target_version=target_version,
                total_items=scope['total_items'],
                completed_items=0,
                failed_items=0,
                started_at=datetime.now(),
                estimated_completion=datetime.now() + timedelta(
                    hours=scope['total_items'] / (self.batch_size * self.max_concurrent_batches)
                )
            )
            
            # Store migration metadata
            await self._store_migration_progress(migration_id, progress)
            
            # Start async migration process
            asyncio.create_task(self._execute_migration(migration_id, source_version, target_version))
            
            logger.info(f"Started migration {migration_id}: {scope['total_items']} items, ~${estimated_cost:.2f}")
            return migration_id
            
        except Exception as e:
            logger.error(f"Failed to start migration: {e}")
            raise
            
    async def _execute_migration(self, migration_id: str, source_version: str, 
                                target_version: str):
        """Execute the actual migration with batching and error handling"""
        try:
            # Process each table separately  
            tables = ['resume_chunks', 'job_chunks']
            
            for table in tables:
                await self._migrate_table(migration_id, table, source_version, target_version)
                
            # Mark migration as complete
            await self._complete_migration(migration_id)
            
        except Exception as e:
            logger.error(f"Migration {migration_id} failed: {e}")
            await self._fail_migration(migration_id, str(e))
            
    async def _migrate_table(self, migration_id: str, table: str, 
                           source_version: str, target_version: str):
        """Migrate a single table with concurrent batching"""
        
        # Get batches of items needing migration
        batches = await self._get_migration_batches(table, source_version)
        
        # Process batches concurrently with semaphore for rate limiting
        semaphore = asyncio.Semaphore(self.max_concurrent_batches)
        
        async def process_batch(batch: MigrationBatch):
            async with semaphore:
                return await self._process_migration_batch(
                    migration_id, batch, target_version
                )
                
        # Execute batches
        batch_tasks = [process_batch(batch) for batch in batches]
        results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Log results
        succeeded = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - succeeded
        
        logger.info(f"Table {table} migration: {succeeded} batches succeeded, {failed} failed")
        
    async def _get_migration_batches(self, table: str, source_version: str) -> List[MigrationBatch]:
        """Get batches of items that need re-embedding"""
        batches = []
        offset = 0
        
        while True:
            # Get next batch of items
            query = f"""
                SELECT id, chunk_text, token_count
                FROM {table}
                WHERE embedding_version = %s
                AND needs_reembedding = TRUE
                ORDER BY id
                LIMIT %s OFFSET %s
            """
            
            rows = await self.session.execute(query, (source_version, self.batch_size, offset))
            items = [dict(row) for row in rows.fetchall()]
            
            if not items:
                break
                
            # Calculate batch metrics
            total_tokens = sum(item.get('token_count', 100) for item in items)
            estimated_cost = self._estimate_batch_cost(total_tokens)
            
            batch = MigrationBatch(
                table_name=table,
                batch_id=f"{table}_{offset}",
                items=items,
                total_tokens=total_tokens,
                estimated_cost=estimated_cost
            )
            
            batches.append(batch)
            offset += self.batch_size
            
        logger.info(f"Created {len(batches)} batches for table {table}")
        return batches
        
    async def _process_migration_batch(self, migration_id: str, batch: MigrationBatch,
                                     target_version: str) -> Dict[str, Any]:
        """Process a single batch of re-embedding"""
        start_time = datetime.now()
        
        try:
            # Extract texts for embedding
            texts = [item['chunk_text'] for item in batch.items]
            
            # Batch embedding call
            embeddings = await self.embedding_service.embed_batch(texts)
            
            # Update database with new embeddings
            updates = []
            for item, embedding in zip(batch.items, embeddings):
                updates.append({
                    'id': item['id'],
                    'embedding': embedding,
                    'embedding_version': target_version,
                    'needs_reembedding': False,
                    'updated_at': datetime.now()
                })
                
            # Bulk update
            await self._bulk_update_embeddings(batch.table_name, updates)
            
            # Update progress
            await self._update_migration_progress(migration_id, len(batch.items), 0)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return {
                'batch_id': batch.batch_id,
                'items_processed': len(batch.items),
                'duration_seconds': duration,
                'cost_usd': batch.estimated_cost,
                'status': 'success'
            }
            
        except Exception as e:
            # Update progress with failures
            await self._update_migration_progress(migration_id, 0, len(batch.items))
            
            logger.error(f"Batch {batch.batch_id} failed: {e}")
            return {
                'batch_id': batch.batch_id,
                'status': 'failed',
                'error': str(e)
            }
            
    async def _bulk_update_embeddings(self, table: str, updates: List[Dict]):
        """Efficiently update embeddings in bulk"""
        try:
            # Use UNNEST for efficient batch updates in PostgreSQL
            ids = [u['id'] for u in updates]
            embeddings = [u['embedding'] for u in updates]
            versions = [u['embedding_version'] for u in updates]
            
            query = f"""
                UPDATE {table} SET
                    embedding = data.embedding::vector,
                    embedding_version = data.version,
                    needs_reembedding = FALSE,
                    updated_at = NOW()
                FROM (
                    SELECT UNNEST(%s::uuid[]) as id,
                           UNNEST(%s::vector[]) as embedding,
                           UNNEST(%s::text[]) as version
                ) data
                WHERE {table}.id = data.id
            """
            
            await self.session.execute(query, (ids, embeddings, versions))
            await self.session.commit()
            
        except Exception as e:
            logger.error(f"Bulk update failed: {e}")
            await self.session.rollback()
            raise
            
    def _estimate_migration_scope(self, source_version: str) -> Dict[str, Any]:
        """Estimate scope of migration"""
        # This would query actual database
        return {
            'total_items': 10000,  # Placeholder
            'resume_chunks': 6000,
            'job_chunks': 4000,
            'total_tokens': 1000000,
        }
        
    def _estimate_migration_cost(self, scope: Dict) -> float:
        """Estimate cost of migration based on token count"""
        total_tokens = scope['total_tokens']
        cost_per_1k_tokens = 0.0001  # OpenAI text-embedding-3-large pricing
        return (total_tokens / 1000) * cost_per_1k_tokens
        
    def _estimate_batch_cost(self, tokens: int) -> float:
        """Estimate cost for a single batch"""
        cost_per_1k_tokens = 0.0001
        return (tokens / 1000) * cost_per_1k_tokens
        
    def _generate_migration_id(self, source: str, target: str) -> str:
        """Generate unique migration ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        content = f"{source}_{target}_{timestamp}"
        hash_suffix = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"migration_{hash_suffix}"
        
    async def _store_migration_progress(self, migration_id: str, progress: MigrationProgress):
        """Store migration progress to database"""
        # Implementation would store to migration_progress table
        pass
        
    async def _update_migration_progress(self, migration_id: str, 
                                       completed: int, failed: int):
        """Update migration progress counters"""
        # Implementation would update progress counters
        pass
        
    async def _complete_migration(self, migration_id: str):
        """Mark migration as complete"""
        logger.info(f"Migration {migration_id} completed successfully")
        
    async def _fail_migration(self, migration_id: str, error: str):
        """Mark migration as failed"""
        logger.error(f"Migration {migration_id} failed: {error}")

class EmbeddingVersionManager:
    """Enhanced version manager with migration and compatibility support"""
    
    def __init__(self, session, embedding_service=None):
        self.session = session
        self.embedding_service = embedding_service
        self.compatibility_analyzer = EmbeddingCompatibilityAnalyzer()
        self.migration_manager = ProgressiveMigrationManager(session, embedding_service)
        self._cache = {}  # In-memory cache for active version

    async def get_active_version(self) -> EmbeddingVersionInfo:
        """Get currently active embedding version with caching"""
        if 'active_version' in self._cache:
            cached_time, version = self._cache['active_version']
            if datetime.now() - cached_time < timedelta(minutes=5):
                return version
        
        try:
            # Query active version from database
            query = """
                SELECT version_id, model_name, dimensions, created_at, deprecated_at
                FROM embedding_versions 
                WHERE deprecated_at IS NULL 
                ORDER BY created_at DESC 
                LIMIT 1
            """
            
            row = await self.session.execute(query)
            result = row.fetchone()
            
            if result:
                version = EmbeddingVersionInfo(
                    version_id=result[0],
                    model_name=result[1],
                    dimensions=result[2], 
                    created_at=result[3],
                    deprecated_at=result[4]
                )
            else:
                # Default version if none found
                version = EmbeddingVersionInfo(
                    "v1.0", "text-embedding-3-large", 3072, datetime.utcnow()
                )
                
            # Cache result
            self._cache['active_version'] = (datetime.now(), version)
            return version
            
        except Exception as e:
            logger.error(f"Failed to get active version: {e}")
            # Return default version
            return EmbeddingVersionInfo(
                "v1.0", "text-embedding-3-large", 3072, datetime.utcnow()
            )

    async def create_new_version(self, model_name: str, dimensions: int) -> EmbeddingVersionInfo:
        """Create new embedding version with compatibility analysis"""
        try:
            # Get current active version for compatibility check
            current_version = await self.get_active_version()
            
            # Generate new version ID
            version_num = len(await self.get_all_versions()) + 1
            new_version_id = f"v{version_num}.0"
            
            # Analyze compatibility with current version
            compatibility_score = await self.compatibility_analyzer.analyze_compatibility(
                current_version.version_id, new_version_id, self.session
            )
            
            # Create new version record
            new_version = EmbeddingVersionInfo(
                version_id=new_version_id,
                model_name=model_name,
                dimensions=dimensions,
                created_at=datetime.now(),
                compatibility_score=compatibility_score
            )
            
            # Store in database
            await self._store_version(new_version)
            
            # Clear cache
            self._cache.clear()
            
            logger.info(f"Created new embedding version: {new_version_id}")
            return new_version
            
        except Exception as e:
            logger.error(f"Failed to create new version: {e}")
            raise

    def stamp_embedding_metadata(self, row, version: EmbeddingVersionInfo):
        """Stamp embedding with version metadata"""
        row.embedding_version = version.version_id
        row.embedding_model = version.model_name
        row.updated_at = datetime.now()

    async def mark_legacy_for_reembedding(self, legacy_versions: Iterable[str]) -> int:
        """Mark embeddings from legacy versions for re-embedding"""
        try:
            version_list = list(legacy_versions)
            if not version_list:
                return 0
                
            # Update resume chunks
            resume_update = """
                UPDATE resume_chunks 
                SET needs_reembedding = TRUE, updated_at = NOW()
                WHERE embedding_version = ANY(%s)
                AND needs_reembedding = FALSE
            """
            
            resume_result = await self.session.execute(resume_update, (version_list,))
            
            # Update job chunks
            job_update = """
                UPDATE job_chunks
                SET needs_reembedding = TRUE, updated_at = NOW() 
                WHERE embedding_version = ANY(%s)
                AND needs_reembedding = FALSE
            """
            
            job_result = await self.session.execute(job_update, (version_list,))
            await self.session.commit()
            
            total_marked = resume_result.rowcount + job_result.rowcount
            
            logger.info(f"Marked {total_marked} embeddings for re-embedding from versions: {version_list}")
            return total_marked
            
        except Exception as e:
            logger.error(f"Failed to mark legacy embeddings: {e}")
            await self.session.rollback()
            return 0

    async def next_reembedding_batch(self, table: str, batch_size: int = 500) -> List[Dict]:
        """Get next batch of items needing re-embedding"""
        try:
            query = f"""
                SELECT id, chunk_text, token_count, embedding_version
                FROM {table}
                WHERE needs_reembedding = TRUE
                ORDER BY updated_at ASC  -- Oldest first
                LIMIT %s
            """
            
            result = await self.session.execute(query, (batch_size,))
            rows = result.fetchall()
            
            return [
                {
                    'id': row[0],
                    'chunk_text': row[1],
                    'token_count': row[2],
                    'old_version': row[3]
                }
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"Failed to get reembedding batch: {e}")
            return []

    async def mark_reembedded(self, items: List[Dict], new_version: EmbeddingVersionInfo):
        """Mark items as successfully re-embedded"""
        try:
            item_ids = [item['id'] for item in items]
            
            # This assumes the embeddings were already updated by the migration process
            query = """
                UPDATE {table} SET
                    needs_reembedding = FALSE,
                    embedding_version = %s,
                    updated_at = NOW()
                WHERE id = ANY(%s)
            """
            
            # Update both tables
            for table in ['resume_chunks', 'job_chunks']:
                await self.session.execute(
                    query.format(table=table), 
                    (new_version.version_id, item_ids)
                )
                
            await self.session.commit()
            
            logger.info(f"Marked {len(items)} items as re-embedded with version {new_version.version_id}")
            
        except Exception as e:
            logger.error(f"Failed to mark items as re-embedded: {e}")
            await self.session.rollback()
            
    async def get_migration_status(self, migration_id: str) -> Optional[Dict]:
        """Get status of ongoing migration"""
        # Implementation would query migration_progress table
        return None
        
    async def get_all_versions(self) -> List[EmbeddingVersionInfo]:
        """Get all embedding versions"""
        try:
            query = """
                SELECT version_id, model_name, dimensions, created_at, deprecated_at
                FROM embedding_versions
                ORDER BY created_at DESC
            """
            
            result = await self.session.execute(query)
            rows = result.fetchall()
            
            return [
                EmbeddingVersionInfo(
                    version_id=row[0],
                    model_name=row[1], 
                    dimensions=row[2],
                    created_at=row[3],
                    deprecated_at=row[4]
                )
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"Failed to get all versions: {e}")
            return []
            
    async def _store_version(self, version: EmbeddingVersionInfo):
        """Store version in database"""
        query = """
            INSERT INTO embedding_versions 
            (version_id, model_name, dimensions, created_at)
            VALUES (%s, %s, %s, %s)
        """
        
        await self.session.execute(query, (
            version.version_id,
            version.model_name, 
            version.dimensions,
            version.created_at
        ))
        await self.session.commit()
