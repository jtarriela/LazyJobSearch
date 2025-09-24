"""Embedding service for resume and job content

Handles text embedding with caching, batching, and cost tracking.
Designed to work with various embedding models while maintaining performance.
"""
from __future__ import annotations
import logging
import hashlib
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class EmbeddingProvider(Enum):
    """Supported embedding providers"""
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    COHERE = "cohere"
    MOCK = "mock"  # For testing

@dataclass
class EmbeddingRequest:
    """Request for embedding generation"""
    text: str
    model: str
    text_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.text_id is None:
            self.text_id = self._generate_text_id()
    
    def _generate_text_id(self) -> str:
        """Generate unique ID for text content"""
        content_hash = hashlib.sha256(
            f"{self.text}:{self.model}".encode('utf-8')
        ).hexdigest()
        return f"emb_{content_hash[:16]}"

@dataclass
class EmbeddingResponse:
    """Response from embedding generation"""
    text_id: str
    embedding: List[float]
    model: str
    dimensions: int
    tokens_used: int
    cost_cents: float
    created_at: datetime
    cached: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class EmbeddingBatch:
    """Batch of embedding requests for efficient processing"""
    requests: List[EmbeddingRequest]
    batch_id: str
    created_at: datetime
    priority: int = 0  # Higher numbers = higher priority

class EmbeddingCache:
    """Simple in-memory cache for embeddings with TTL"""
    
    def __init__(self, max_size: int = 10000, ttl_hours: int = 24):
        self.cache: Dict[str, Tuple[EmbeddingResponse, datetime]] = {}
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
    
    def get(self, text_id: str) -> Optional[EmbeddingResponse]:
        """Get cached embedding if available and not expired"""
        if text_id not in self.cache:
            return None
        
        embedding, cached_at = self.cache[text_id]
        
        # Check if expired
        if datetime.now() - cached_at > self.ttl:
            del self.cache[text_id]
            return None
        
        # Mark as cached and return
        embedding.cached = True
        return embedding
    
    def put(self, embedding: EmbeddingResponse) -> None:
        """Cache an embedding response"""
        # Evict oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[embedding.text_id] = (embedding, datetime.now())
    
    def _evict_oldest(self) -> None:
        """Evict oldest 10% of entries"""
        if not self.cache:
            return
        
        # Sort by cache time and remove oldest 10%
        sorted_items = sorted(
            self.cache.items(), 
            key=lambda x: x[1][1]  # Sort by cached_at time
        )
        
        evict_count = max(1, len(sorted_items) // 10)
        for text_id, _ in sorted_items[:evict_count]:
            del self.cache[text_id]
    
    def clear(self) -> None:
        """Clear all cached embeddings"""
        self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)

class MockEmbeddingProvider:
    """Mock embedding provider for testing"""
    
    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions
        self.request_count = 0
    
    async def embed_texts(self, texts: List[str], model: str = "mock-model") -> List[List[float]]:
        """Generate mock embeddings"""
        embeddings = []
        
        for text in texts:
            # Generate deterministic embeddings based on text hash
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            
            # Convert hash to float values between -1 and 1
            embedding = []
            for i in range(0, min(len(text_hash), self.dimensions * 8), 8):
                hex_val = text_hash[i:i+8]
                int_val = int(hex_val, 16) if hex_val else 0
                float_val = (int_val / (16**8)) * 2 - 1  # Scale to [-1, 1]
                embedding.append(float_val)
            
            # Pad or truncate to desired dimensions
            while len(embedding) < self.dimensions:
                embedding.append(0.0)
            embedding = embedding[:self.dimensions]
            
            embeddings.append(embedding)
        
        self.request_count += 1
        return embeddings
    
    def count_tokens(self, text: str) -> int:
        """Mock token counting"""
        return len(text.split())
    
    def estimate_cost(self, token_count: int) -> float:
        """Mock cost estimation (in cents)"""
        return token_count * 0.001  # $0.00001 per token

class EmbeddingService:
    """Main embedding service with caching, batching, and cost tracking"""
    
    def __init__(
        self, 
        provider: EmbeddingProvider = EmbeddingProvider.MOCK,
        model: str = "text-embedding-ada-002",
        cache_enabled: bool = True,
        batch_size: int = 100,
        max_retries: int = 3,
        rate_limit_delay: float = 0.1
    ):
        self.provider = provider
        self.model = model
        self.cache_enabled = cache_enabled
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        
        # Initialize cache
        self.cache = EmbeddingCache() if cache_enabled else None
        
        # Initialize provider
        self._init_provider()
        
        # Cost tracking
        self.total_tokens_used = 0
        self.total_cost_cents = 0.0
        self.requests_made = 0
        self.cache_hits = 0
    
    def _init_provider(self):
        """Initialize the embedding provider"""
        if self.provider == EmbeddingProvider.MOCK:
            self._provider = MockEmbeddingProvider()
        elif self.provider == EmbeddingProvider.OPENAI:
            # In production, initialize OpenAI client
            logger.warning("OpenAI provider not implemented - using mock")
            self._provider = MockEmbeddingProvider()
        else:
            # Other providers
            logger.warning(f"Provider {self.provider} not implemented - using mock")
            self._provider = MockEmbeddingProvider()
    
    async def embed_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> EmbeddingResponse:
        """Embed a single text"""
        request = EmbeddingRequest(
            text=text,
            model=self.model,
            metadata=metadata or {}
        )
        
        responses = await self.embed_batch([request])
        return responses[0]
    
    async def embed_batch(self, requests: List[EmbeddingRequest]) -> List[EmbeddingResponse]:
        """Embed a batch of texts with advanced optimization: deduplication, caching, and batching
        
        Implements the batch optimization algorithm from PERFORMANCE_OPTIMIZATION.md:
        1. Deduplicate requests (common in job descriptions)
        2. Check cache for existing embeddings  
        3. Batch API calls for remaining texts
        4. Reconstruct full response list maintaining order
        """
        if not requests:
            return []
            
        start_time = time.time()
        
        # Step 1: Deduplicate requests while tracking original positions
        unique_texts = {}
        text_to_indices = {}
        
        for i, request in enumerate(requests):
            text_key = f"{request.text}:{request.model}"
            
            if text_key not in unique_texts:
                unique_texts[text_key] = request
                text_to_indices[text_key] = []
            text_to_indices[text_key].append(i)
        
        logger.debug(f"Deduplication: {len(requests)} requests -> {len(unique_texts)} unique texts")
        
        # Step 2: Check cache for unique texts
        cached_responses = {}
        uncached_requests = []
        
        for text_key, request in unique_texts.items():
            if self.cache_enabled and self.cache:
                cached_response = self.cache.get(request.text_id)
                if cached_response:
                    self.cache_hits += 1
                    cached_responses[text_key] = cached_response
                    continue
            
            uncached_requests.append((text_key, request))
        
        logger.debug(f"Cache check: {len(cached_responses)} hits, {len(uncached_requests)} misses")
        
        # Step 3: Process uncached requests in batches
        new_responses = {}
        if uncached_requests:
            batch_responses = await self._process_embedding_requests_optimized(uncached_requests)
            
            # Cache new responses
            if self.cache_enabled and self.cache:
                for text_key, response in batch_responses.items():
                    self.cache.put(response)
                    
            new_responses = batch_responses
        
        # Step 4: Reconstruct full response list maintaining original order
        all_responses = {**cached_responses, **new_responses}
        final_responses = [None] * len(requests)
        
        for text_key, indices in text_to_indices.items():
            if text_key in all_responses:
                response = all_responses[text_key]
                for idx in indices:
                    final_responses[idx] = response
            else:
                # Should not happen, but handle gracefully
                logger.error(f"Missing response for text_key: {text_key}")
                # Create error response
                error_response = EmbeddingResponse(
                    text_id=requests[indices[0]].text_id,
                    embedding=[0.0] * 1536,  # Default embedding size
                    model=requests[indices[0]].model,
                    dimensions=1536,
                    tokens_used=0,
                    cost_cents=0.0,
                    created_at=datetime.now()
                )
                for idx in indices:
                    final_responses[idx] = error_response
        
        # Performance metrics
        processing_time = time.time() - start_time
        deduplication_ratio = len(requests) / len(unique_texts) if unique_texts else 1.0
        cache_hit_ratio = len(cached_responses) / len(unique_texts) if unique_texts else 0.0
        
        logger.info(f"Batch embedding completed: {processing_time:.3f}s, "
                   f"dedup_ratio={deduplication_ratio:.2f}, cache_hit_ratio={cache_hit_ratio:.2f}")
        
        return final_responses
    
    async def _process_embedding_requests_optimized(self, text_key_requests: List[Tuple[str, EmbeddingRequest]]) -> Dict[str, EmbeddingResponse]:
        """Process embedding requests with optimal batching"""
        responses = {}
        
        # Split into batches respecting batch_size limit
        for i in range(0, len(text_key_requests), self.batch_size):
            batch = text_key_requests[i:i + self.batch_size]
            
            try:
                batch_texts = [req.text for _, req in batch]
                batch_models = [req.model for _, req in batch]
                
                # Ensure all requests in batch use same model (API requirement)
                if len(set(batch_models)) > 1:
                    logger.warning("Mixed models in batch - processing individually")
                    for text_key, request in batch:
                        individual_responses = await self._process_single_embedding(request)
                        responses[text_key] = individual_responses
                else:
                    # Process batch with single model
                    model = batch_models[0]
                    batch_embeddings = await self._provider.embed_texts(batch_texts, model)
                    
                    # Create responses
                    for (text_key, request), embedding in zip(batch, batch_embeddings):
                        # Estimate tokens (rough approximation: 1 token ~= 0.75 words)
                        estimated_tokens = max(1, len(request.text.split()) * 0.75)
                        
                        # Estimate cost (OpenAI pricing: ~$0.0001 per 1k tokens for ada-002)
                        cost_per_token = 0.0001 / 1000
                        estimated_cost_cents = estimated_tokens * cost_per_token * 100
                        
                        response = EmbeddingResponse(
                            text_id=request.text_id,
                            embedding=embedding,
                            model=model,
                            dimensions=len(embedding),
                            tokens_used=int(estimated_tokens),
                            cost_cents=estimated_cost_cents,
                            created_at=datetime.now()
                        )
                        
                        responses[text_key] = response
                        
                        # Update service stats
                        self.total_tokens_used += int(estimated_tokens)
                        self.total_cost_cents += estimated_cost_cents
                
                self.requests_made += 1
                
                # Rate limiting between batches
                if i + self.batch_size < len(text_key_requests):
                    await asyncio.sleep(self.rate_limit_delay)
                    
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                
                # Create error responses for failed batch
                for text_key, request in batch:
                    error_response = EmbeddingResponse(
                        text_id=request.text_id,
                        embedding=[0.0] * 1536,  # Default embedding size
                        model=request.model,
                        dimensions=1536,
                        tokens_used=0,
                        cost_cents=0.0,
                        created_at=datetime.now()
                    )
                    responses[text_key] = error_response
        
        return responses
    
    async def _process_single_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Process a single embedding request"""
        try:
            embeddings = await self._provider.embed_texts([request.text], request.model)
            embedding = embeddings[0]
            
            # Estimate tokens and cost
            estimated_tokens = max(1, len(request.text.split()) * 0.75)
            cost_per_token = 0.0001 / 1000
            estimated_cost_cents = estimated_tokens * cost_per_token * 100
            
            response = EmbeddingResponse(
                text_id=request.text_id,
                embedding=embedding,
                model=request.model,
                dimensions=len(embedding),
                tokens_used=int(estimated_tokens),
                cost_cents=estimated_cost_cents,
                created_at=datetime.now()
            )
            
            # Update service stats
            self.total_tokens_used += int(estimated_tokens)
            self.total_cost_cents += estimated_cost_cents
            
            return response
            
        except Exception as e:
            logger.error(f"Single embedding failed: {e}")
            return EmbeddingResponse(
                text_id=request.text_id,
                embedding=[0.0] * 1536,
                model=request.model,
                dimensions=1536,
                tokens_used=0,
                cost_cents=0.0,
                created_at=datetime.now()
            )
        responses.sort(key=lambda r: next(
            i for i, req in enumerate(requests) 
            if req.text_id == r.text_id
        ))
        
        return responses
    
    async def _process_embedding_requests(self, requests: List[EmbeddingRequest]) -> List[EmbeddingResponse]:
        """Process embedding requests in batches"""
        all_responses = []
        
        # Process in batches
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            batch_responses = await self._embed_batch_with_retry(batch)
            all_responses.extend(batch_responses)
            
            # Rate limiting
            if i + self.batch_size < len(requests):
                await asyncio.sleep(self.rate_limit_delay)
        
        return all_responses
    
    async def _embed_batch_with_retry(self, batch: List[EmbeddingRequest]) -> List[EmbeddingResponse]:
        """Embed a batch with retry logic"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return await self._embed_batch_direct(batch)
            except Exception as e:
                last_error = e
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = (2 ** attempt) * self.rate_limit_delay
                    await asyncio.sleep(wait_time)
        
        # All retries failed
        logger.error(f"All embedding attempts failed: {last_error}")
        raise Exception(f"Embedding failed after {self.max_retries} attempts: {last_error}")
    
    async def _embed_batch_direct(self, batch: List[EmbeddingRequest]) -> List[EmbeddingResponse]:
        """Direct embedding call to provider"""
        texts = [req.text for req in batch]
        
        # Get embeddings from provider
        embeddings = await self._provider.embed_texts(texts, self.model)
        
        responses = []
        for i, request in enumerate(batch):
            # Count tokens and estimate cost
            token_count = self._provider.count_tokens(request.text)
            cost_cents = self._provider.estimate_cost(token_count)
            
            # Update tracking
            self.total_tokens_used += token_count
            self.total_cost_cents += cost_cents
            self.requests_made += 1
            
            response = EmbeddingResponse(
                text_id=request.text_id,
                embedding=embeddings[i],
                model=self.model,
                dimensions=len(embeddings[i]),
                tokens_used=token_count,
                cost_cents=cost_cents,
                created_at=datetime.now(),
                cached=False,
                metadata=request.metadata
            )
            responses.append(response)
        
        return responses
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding service statistics"""
        cache_size = self.cache.size() if self.cache else 0
        cache_hit_rate = (
            self.cache_hits / (self.requests_made + self.cache_hits) 
            if (self.requests_made + self.cache_hits) > 0 
            else 0.0
        )
        
        return {
            'provider': self.provider.value,
            'model': self.model,
            'total_requests': self.requests_made,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': cache_size,
            'total_tokens_used': self.total_tokens_used,
            'total_cost_cents': self.total_cost_cents,
            'total_cost_dollars': self.total_cost_cents / 100.0
        }
    
    def clear_cache(self) -> None:
        """Clear the embedding cache"""
        if self.cache:
            self.cache.clear()
    
    def reset_stats(self) -> None:
        """Reset usage statistics"""
        self.total_tokens_used = 0
        self.total_cost_cents = 0.0
        self.requests_made = 0
        self.cache_hits = 0

def create_embedding_service(
    provider: EmbeddingProvider = EmbeddingProvider.MOCK,
    model: str = "text-embedding-ada-002",
    **kwargs
) -> EmbeddingService:
    """Factory function to create configured embedding service"""
    return EmbeddingService(provider=provider, model=model, **kwargs)