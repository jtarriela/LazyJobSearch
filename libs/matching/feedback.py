"""Adaptive Matching Feedback Loop (ADR 0007)

Stubs for capturing outcomes and computing learned weights over feature vectors.
Real implementation will plug in SQLAlchemy sessions and scikit-learn logistic regression.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any
from datetime import datetime

@dataclass
class MatchOutcomeRecord:
    match_id: str
    got_response: bool | None
    response_time_hours: int | None
    got_interview: bool | None
    got_offer: bool | None
    user_satisfaction: int | None
    captured_at: datetime

class FeedbackCapture:
    def __init__(self, session):
        self.session = session

    def capture(self, outcome: MatchOutcomeRecord):
        # Persist outcome row (placeholder)
        return True

class FeatureWeightModel:
    def __init__(self, weights: Dict[str, float], model_version: str, trained_at: datetime):
        self.weights = weights
        self.model_version = model_version
        self.trained_at = trained_at

    def score(self, feature_vector: Dict[str, float]) -> float:
        return sum(self.weights.get(k, 0.0) * v for k, v in feature_vector.items())

class FeedbackTrainer:
    def __init__(self, session):
        self.session = session

    def fetch_training_data(self, lookback_days: int = 30) -> List[Dict[str, Any]]:
        # Placeholder: return list of {vector_score:..., llm_score:..., fts_rank:..., yoe_gap:..., skill_overlap:..., got_interview: bool}
        return []

    def train(self) -> FeatureWeightModel:
        data = self.fetch_training_data()
        if not data:
            return FeatureWeightModel({}, "baseline", datetime.utcnow())
        # Placeholder heuristic: weight vector_score and llm_score equally
        weights = {"vector_score": 0.5, "llm_score": 0.5}
        return FeatureWeightModel(weights, "v1", datetime.utcnow())
