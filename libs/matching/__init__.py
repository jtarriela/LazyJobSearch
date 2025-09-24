"""Matching pipeline module for LazyJobSearch

This module provides the core matching pipeline that ranks jobs against resumes
using a hybrid approach: FTS prefilter → vector similarity → LLM scoring.
"""