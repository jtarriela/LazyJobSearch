"""Matching pipeline module for LazyJobSearch

This module provides the core matching pipeline that ranks jobs against resumes
using a hybrid approach: FTS prefilter → vector similarity → LLM scoring.

Current implementation includes basic skill and experience matching as a foundation.
"""

from .basic_pipeline import (
    BasicMatchingPipeline,
    MatchingStrategy, 
    MatchResult,
    MatchingConfig,
    create_matching_pipeline
)

__all__ = [
    'BasicMatchingPipeline',
    'MatchingStrategy',
    'MatchResult', 
    'MatchingConfig',
    'create_matching_pipeline'
]