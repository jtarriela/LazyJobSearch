"""Anti-Bot & Humanization Utilities (ADR 0008)

Provides stubs for proxy rotation, fingerprint profile generation, and human-like
interaction simulation. Real implementations will integrate with Selenium or
Playwright contexts.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime

@dataclass
class FingerprintProfile:
    user_agent: str
    viewport: tuple[int, int]
    timezone: str
    language: str
    canvas_seed: int

class ProxyPool:
    def __init__(self, proxies: list[str]):
        self.proxies = proxies

    def get(self) -> Optional[str]:
        return random.choice(self.proxies) if self.proxies else None

class FingerprintGenerator:
    UA_LIST = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
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
    def __init__(self, jitter_range=(0.2, 0.9)):
        self.jitter_range = jitter_range

    def sleep_interval(self) -> float:
        import random
        return random.uniform(*self.jitter_range)

    def mouse_path(self, start: tuple[int, int], end: tuple[int, int], steps: int = 12):
        # Simple linear interpolation placeholder; real impl would use Bezier curves
        (x0, y0), (x1, y1) = start, end
        dx = (x1 - x0) / steps
        dy = (y1 - y0) / steps
        return [(x0 + i * dx, y0 + i * dy) for i in range(steps + 1)]

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
