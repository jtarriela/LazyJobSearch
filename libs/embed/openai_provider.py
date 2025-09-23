"""OpenAI embedding provider implementation."""

import os
from typing import List, Optional
import numpy as np
from openai import OpenAI

from .provider_base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using text-embedding-3-large model.
    
    This is the primary provider for the MVP as per ADR 0004.
    Provides high-quality embeddings with 1536 dimensions.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-large"):
        """Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: OpenAI embedding model to use.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter.")
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        
        # Model dimensions mapping
        self._dimensions = {
            "text-embedding-3-large": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536,
        }
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.get_dimension())
        
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float"
        )
        
        embedding_data = response.data[0].embedding
        return np.array(embedding_data, dtype=np.float32)
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts efficiently.
        
        OpenAI's API supports batching up to ~8000 tokens per request.
        For larger batches, this will chunk appropriately.
        """
        if not texts:
            return []
        
        # Filter out empty texts but keep track of indices
        non_empty = [(i, text) for i, text in enumerate(texts) if text.strip()]
        if not non_empty:
            return [np.zeros(self.get_dimension()) for _ in texts]
        
        # Extract just the texts for API call
        batch_texts = [text for _, text in non_empty]
        
        response = self.client.embeddings.create(
            model=self.model,
            input=batch_texts,
            encoding_format="float"
        )
        
        # Build result array with zero vectors for empty texts
        results = [np.zeros(self.get_dimension()) for _ in texts]
        for (original_idx, _), embedding_data in zip(non_empty, response.data):
            results[original_idx] = np.array(embedding_data.embedding, dtype=np.float32)
        
        return results
    
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimensions.get(self.model, 1536)
    
    def get_model_name(self) -> str:
        """Return the model identifier."""
        return self.model
    
    def estimate_cost(self, token_count: int) -> float:
        """Estimate the cost for embedding a given number of tokens.
        
        Based on OpenAI pricing as of model implementation.
        text-embedding-3-large: $0.00013 per 1K tokens
        """
        cost_per_1k = {
            "text-embedding-3-large": 0.00013,
            "text-embedding-3-small": 0.00002,
            "text-embedding-ada-002": 0.00010,
        }
        
        rate = cost_per_1k.get(self.model, 0.00013)
        return (token_count / 1000) * rate