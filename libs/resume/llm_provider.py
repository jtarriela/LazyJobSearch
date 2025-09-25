# llm_provider.py
from __future__ import annotations
from typing import Dict, Any, Optional
import json
import re
import asyncio
from dataclasses import dataclass

# --- Optional import for richer interop with your llm_service -----------------
try:
    # If your project provides these, we'll use them so accounting/typing matches.
    from .llm_service import LLMRequest, LLMResponse  # type: ignore
except Exception:
    # Minimal local shims so tests can run without llm_service.
    @dataclass
    class LLMRequest:  # type: ignore
        prompt: str
        model: str
        max_tokens: int = 2000
        temperature: float = 0.1
        request_id: str = "mock_req"
        metadata: Dict[str, Any] = None

    @dataclass
    class LLMResponse:  # type: ignore
        content: str
        model: str
        tokens_used: int
        cost_cents: float
        request_id: str
        processing_time_ms: float
        metadata: Dict[str, Any] | None = None

# -----------------------------------------------------------------------------


class MockLLMProvider:
    """
    Simple mock provider that supports BOTH:
      - extract_resume(system_prompt, user_prompt) -> str  (sync, legacy)
      - generate_completion(request: LLMRequest) -> LLMResponse (async, modern)

    Behavior:
      - Deterministic JSON output suitable for tests.
      - Makes a light attempt to extract name/email/phone/skills from the prompt body
        so downstream parsers see realistic values.
    """

    def __init__(self, *, mode: str = "mock"):
        self.mode = mode
        self._req_count = 0

    # ---------------- Legacy sync method (kept for compatibility) -------------
    def extract_resume(self, system_prompt: str, user_prompt: str) -> str:
        """
        Return a deterministic JSON string (minimal parsed resume).
        This preserves your original signature so existing unit tests won't break.
        """
        # Try to pull something that looks like resume text out of the user prompt
        text = self._slice_resume_text(user_prompt)
        payload = self._build_payload_from_text(text)
        return json.dumps(payload)

    # ---------------- Modern async method (preferred) -------------------------
    async def generate_completion(self, request: LLMRequest) -> LLMResponse:
        """
        Async API compatible with service layers that expect a completion-style call.
        """
        await asyncio.sleep(0.01)  # tiny, deterministic "latency" for tests

        # Heuristic: the request.prompt holds both system+user in most callers.
        text = self._slice_resume_text(request.prompt)
        payload = self._build_payload_from_text(text)

        content = json.dumps(payload, indent=2)
        self._req_count += 1

        # Minimal, deterministic accounting
        tokens_used = len(request.prompt.split()) + len(content.split())
        meta = {"provider": "mock", "request_index": self._req_count}

        return LLMResponse(
            content=content,
            model=getattr(request, "model", "mock-model"),
            tokens_used=tokens_used,
            cost_cents=0.0,
            request_id=getattr(request, "request_id", f"mock_{self._req_count}"),
            processing_time_ms=10.0,
            metadata=meta,
        )

    # ---------------- Internals ----------------------------------------------
    def _slice_resume_text(self, prompt: str) -> str:
        """
        Extract likely resume text from a combined prompt. We look for common
        delimiters used by your service. Falls back to the whole prompt.
        """
        sentinel_keys = ["RESUME TEXT:", "Resume:", "TEXT:"]
        for key in sentinel_keys:
            if key in prompt:
                seg = prompt.split(key, 1)[1].strip()
                # Stop at common tail markers
                for stop in ["Extract the following", "Return JSON", "MISSING FIELDS", "Current data:", "IMPORTANT:"]:
                    if stop in seg:
                        seg = seg.split(stop, 1)[0].strip()
                return seg
        return prompt.strip()

    def _build_payload_from_text(self, text: str) -> Dict[str, Any]:
        """
        Deterministic, minimal JSON payload; lightly parses for realism.
        """
        # Email
        email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
        email = email_match.group(0) if email_match else "test@example.com"

        # Phone (basic)
        phone_match = re.search(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", text)
        phone = phone_match.group(0) if phone_match else "(555) 555-5555"

        # Name heuristic: first non-empty, non-email, no digits, <=5 words near the top
        full_name = "Test Candidate"
        for line in [ln.strip() for ln in text.splitlines() if ln.strip()][:5]:
            if "@" in line or any(ch.isdigit() for ch in line):
                continue
            if 2 <= len(line.split()) <= 5 and not any(
                h in line.lower() for h in ["summary", "experience", "education", "skills", "objective"]
            ):
                full_name = line
                break

        # Skills: search a small dictionary
        low = text.lower()
        skill_bank = [
            "python", "java", "react", "sql", "aws", "docker",
            "linux", "pytorch", "tensorflow", "kubernetes",
        ]
        skills = sorted({s for s in skill_bank if s in low}) or ["python"]

        # Deterministic scaffold for tests
        payload = {
            "full_name": full_name,
            "email": email,
            "phone": phone,
            "skills": skills,
            "experience": [],  # keep empty to exercise downstream merges if you like
            "education": [],
            "full_text": text or "N/A",
            "summary": None,
            "certifications": [],
            "years_of_experience": None,
            "education_level": None,
            "links": {"linkedin": None, "github": None, "portfolio": None},
            "parsing_method": "llm",
            "version": 1,
        }
        return payload


# ---------------- Public factory ----------------

def get_provider(name: str = "mock"):
    """
    Simple factory. Extend with other providers as needed (openai/anthropic/gemini).
    """
    name = (name or "mock").lower()
    if name == "mock":
        return MockLLMProvider()
    raise RuntimeError(f"Unknown provider: {name!r}")
