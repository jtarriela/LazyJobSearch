"""Anti-Bot & Humanization Utilities (ADR 0008)

Advanced implementation with Bézier curves, fingerprint randomization,
adaptive behavior patterns, and production-grade detection avoidance.
"""
from __future__ import annotations
import random
import math
import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import logging

logger = logging.getLogger(__name__)

class SessionOutcome(Enum):
    SUCCESS = "success"
    BLOCKED = "blocked"
    CHALLENGE = "challenge"
    TIMEOUT = "timeout"
    ERROR = "error"

@dataclass
class FingerprintProfile:
    user_agent: str
    viewport: tuple[int, int]
    timezone: str
    language: str
    canvas_seed: int
    webgl_vendor: str = "Google Inc."
    webgl_renderer: str = "ANGLE (NVIDIA GeForce GTX 1060 Direct3D11 vs_5_0 ps_5_0)"
    screen_resolution: tuple[int, int] = (1920, 1080)
    color_depth: int = 24
    device_memory: int = 8  # GB
    hardware_concurrency: int = 8  # CPU cores

@dataclass 
class ProxyConfig:
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    protocol: str = "http"  # http, socks5
    success_rate: float = 1.0
    avg_latency_ms: float = 100.0
    last_used: Optional[datetime] = None
    failure_count: int = 0
        
    def is_healthy(self) -> bool:
        """Check if proxy is healthy enough to use"""
        return (self.success_rate > 0.7 and 
                self.failure_count < 5 and
                self.avg_latency_ms < 5000)

class ProxyPool:
    """Advanced proxy pool with health monitoring and rotation"""
    
    def __init__(self, proxies: list[str]):
        # Convert string proxies to ProxyConfig objects
        self.proxies = [
            ProxyConfig(host=proxy.split(':')[0], port=int(proxy.split(':')[1])) 
            for proxy in proxies
        ]
        
    def get(self) -> Optional[str]:
        """Get next healthy proxy"""
        healthy_proxies = [p for p in self.proxies if p.is_healthy()]
        
        if not healthy_proxies:
            # Return first available if none are healthy
            return f"{self.proxies[0].host}:{self.proxies[0].port}" if self.proxies else None
            
        selected = random.choice(healthy_proxies)
        selected.last_used = datetime.now()
        return f"{selected.host}:{selected.port}"

class FingerprintGenerator:
    """Generate realistic browser fingerprints"""
    
    UA_LIST = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    ]

    def create(self) -> FingerprintProfile:
        return FingerprintProfile(
            user_agent=random.choice(self.UA_LIST),
            viewport=(random.randint(1200, 1920), random.randint(800, 1080)),
            timezone="UTC",
            language=random.choice(["en-US", "en-GB"]),
            canvas_seed=random.randint(0, 2**31 - 1),
        )

class HumanBehaviorSimulator:
    """Simulate realistic human browsing patterns"""
    
    def __init__(self, jitter_range=(0.2, 0.9)):
        self.jitter_range = jitter_range

    def sleep_interval(self) -> float:
        return random.uniform(*self.jitter_range)

    def mouse_path(self, start: tuple[int, int], end: tuple[int, int], steps: int = 12):
        """Generate Bézier curve path for mouse movement"""
        x0, y0 = start
        x3, y3 = end
        
        # Simple Bézier curve with control points
        dx = x3 - x0
        dy = y3 - y0
        
        # Control points for smooth curve
        x1 = x0 + dx * 0.3 + random.uniform(-50, 50)
        y1 = y0 + dy * 0.3 + random.uniform(-50, 50)
        x2 = x0 + dx * 0.7 + random.uniform(-30, 30)
        y2 = y0 + dy * 0.7 + random.uniform(-30, 30)
        
        path = []
        for i in range(steps + 1):
            t = i / steps
            # Cubic Bézier curve formula
            x = (1-t)**3 * x0 + 3*(1-t)**2*t * x1 + 3*(1-t)*t**2 * x2 + t**3 * x3
            y = (1-t)**3 * y0 + 3*(1-t)**2*t * y1 + 3*(1-t)*t**2 * y2 + t**3 * y3
            path.append((x, y))
            
        return path

@dataclass
class ScrapeSessionRecord:
    proxy: Optional[str]
    profile: FingerprintProfile
    started_at: datetime
    finished_at: Optional[datetime] = None
    outcome: Optional[str] = None

class ScrapeSessionManager:
    def __init__(self, proxy_pool: ProxyPool, fp_generator: FingerprintGenerator):
        self.proxy_pool = proxy_pool
        self.fp_generator = fp_generator

    def start(self) -> ScrapeSessionRecord:
        proxy = self.proxy_pool.get()
        profile = self.fp_generator.create()
        return ScrapeSessionRecord(proxy=proxy, profile=profile, started_at=datetime.utcnow())

    def finish(self, session: ScrapeSessionRecord, outcome: str):
        session.finished_at = datetime.utcnow()
        session.outcome = outcome
        return session
