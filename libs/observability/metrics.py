"""Metrics and observability infrastructure for LazyJobSearch.

This module provides a simple abstraction for metrics collection that can be extended
with actual implementations (Prometheus, OpenTelemetry, etc.) later.

Based on requirements from PERFORMANCE_OPTIMIZATION.md and the gap analysis.
"""
from __future__ import annotations
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import threading
from collections import defaultdict
import json

@dataclass
class MetricData:
    """Container for metric data point"""
    name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    type: str = "counter"  # counter, histogram, gauge


class MetricsCollector:
    """Thread-safe metrics collector with in-memory storage.
    
    This is a basic implementation that can be swapped for Prometheus,
    OpenTelemetry, or other observability systems later.
    """
    
    def __init__(self):
        self._metrics: Dict[str, List[MetricData]] = defaultdict(list)
        self._lock = threading.Lock()
        
    def counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric (cumulative value)"""
        self._record(name, value, tags or {}, "counter")
        
    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric (distribution of values)"""  
        self._record(name, value, tags or {}, "histogram")
        
    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric (current state)"""
        self._record(name, value, tags or {}, "gauge")
        
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations"""
        return TimerContext(self, name, tags or {})
        
    def _record(self, name: str, value: float, tags: Dict[str, str], metric_type: str) -> None:
        """Internal method to record metric data"""
        with self._lock:
            self._metrics[name].append(MetricData(
                name=name,
                value=value, 
                tags=tags,
                type=metric_type
            ))
            
    def get_metrics(self, name: Optional[str] = None) -> Dict[str, List[MetricData]]:
        """Get recorded metrics, optionally filtered by name"""
        with self._lock:
            if name:
                return {name: self._metrics.get(name, [])}
            return dict(self._metrics)
            
    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics of recorded metrics"""
        with self._lock:
            stats = {}
            for name, metrics in self._metrics.items():
                if metrics:
                    values = [m.value for m in metrics]
                    stats[name] = {
                        'count': len(values),
                        'latest': values[-1] if values else 0,
                        'sum': sum(values),
                        'avg': sum(values) / len(values) if values else 0
                    }
            return stats
            
    def clear(self) -> None:
        """Clear all recorded metrics"""
        with self._lock:
            self._metrics.clear()


class TimerContext:
    """Context manager for measuring operation duration"""
    
    def __init__(self, collector: MetricsCollector, name: str, tags: Dict[str, str]):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time: Optional[float] = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.collector.histogram(f"{self.name}.duration_ms", duration * 1000, self.tags)


# Global metrics instance - can be replaced with proper DI later
_global_metrics = MetricsCollector()

def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance"""
    return _global_metrics

def counter(name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
    """Record a counter metric using global collector"""
    _global_metrics.counter(name, value, tags)

def histogram(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """Record a histogram metric using global collector"""
    _global_metrics.histogram(name, value, tags)

def gauge(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """Record a gauge metric using global collector"""
    _global_metrics.gauge(name, value, tags)

def timer(name: str, tags: Optional[Dict[str, str]] = None):
    """Context manager for timing operations using global collector"""
    return _global_metrics.timer(name, tags)


# Predefined metrics from PERFORMANCE_OPTIMIZATION.md
class PerformanceMetrics:
    """Centralized definitions of key performance metrics"""
    
    # Throughput metrics
    JOBS_PROCESSED_PER_HOUR = "matching.jobs_processed_per_hour"
    PAGES_PER_MINUTE = "scraping.pages_per_minute"
    CHUNKS_PER_MINUTE = "embedding.chunks_per_minute"
    
    # Latency metrics  
    MATCHING_P95_LATENCY = "matching.p95_latency_ms"
    VECTOR_SEARCH_P50_LATENCY = "vector_search.p50_latency_ms"
    LLM_SCORING_P95_LATENCY = "llm_scoring.p95_latency_ms"
    
    # Quality metrics
    MATCHING_PRECISION_AT_10 = "matching.precision_at_10"
    USER_SATISFACTION_SCORE = "matching.user_satisfaction_score" 
    SCRAPING_SUCCESS_RATE = "scraping.success_rate"
    
    # Cost metrics
    LLM_DAILY_SPEND_USD = "llm.daily_spend_usd"
    EMBEDDING_DAILY_API_CALLS = "embedding.daily_api_calls"
    INFRASTRUCTURE_MONTHLY_COST = "infrastructure.monthly_cost_usd"
    
    # Resource utilization
    POSTGRES_CONNECTION_POOL_UTIL = "postgres.connection_pool_utilization"
    POSTGRES_QUERY_DURATION_P95 = "postgres.query_duration_p95"
    REDIS_MEMORY_USAGE_MB = "redis.memory_usage_mb"


def log_performance_metric(metric_name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """Helper to log predefined performance metrics"""
    histogram(metric_name, value, tags)


# Simple structured logging support
class StructuredLogger:
    """Basic structured logger that will be enhanced with proper logging later"""
    
    def __init__(self, name: str = "lazyjobsearch"):
        self.name = name
        
    def info(self, msg: str, **kwargs) -> None:
        self._log("INFO", msg, kwargs)
        
    def error(self, msg: str, **kwargs) -> None:
        self._log("ERROR", msg, kwargs)
        
    def warning(self, msg: str, **kwargs) -> None:
        self._log("WARNING", msg, kwargs)
        
    def debug(self, msg: str, **kwargs) -> None:
        self._log("DEBUG", msg, kwargs)
        
    def _log(self, level: str, msg: str, context: Dict[str, Any]) -> None:
        # Basic logging - replace with proper structured logging later
        if context:
            context_str = json.dumps(context, default=str)
            print(f"[{level}] {self.name}: {msg} | {context_str}")
        else:
            print(f"[{level}] {self.name}: {msg}")


def get_logger(name: str = "lazyjobsearch") -> StructuredLogger:
    """Get a structured logger instance"""
    return StructuredLogger(name)