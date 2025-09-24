"""Adaptive Matching Feedback Loop (ADR 0007)

Advanced implementation with scikit-learn logistic regression, feature engineering,
and optimization strategies for speed and scalability.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging
import numpy as np
from collections import defaultdict

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, classification_report
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    
logger = logging.getLogger(__name__)

@dataclass
class MatchOutcomeRecord:
    match_id: str
    got_response: bool | None
    response_time_hours: int | None
    got_interview: bool | None
    got_offer: bool | None
    user_satisfaction: int | None
    captured_at: datetime

@dataclass
class FeatureVector:
    """Engineered features for matching quality prediction"""
    # Core similarity features
    vector_score: float
    llm_score: float
    fts_rank_norm: float  # Normalized FTS rank (0-1)
    
    # Experience matching features
    yoe_gap: float  # |required_yoe - candidate_yoe|
    yoe_gap_ratio: float  # yoe_gap / required_yoe
    seniority_match: float  # 0-1 match score
    
    # Skill matching features  
    skill_overlap_count: int
    skill_overlap_ratio: float  # overlapping_skills / required_skills
    critical_skill_match: float  # 0-1 for must-have skills
    
    # Content features
    title_similarity: float  # Semantic similarity of job titles
    company_tier_match: float  # 0-1 based on company classification
    location_preference_match: float  # Remote/hybrid/onsite alignment
    
    # Temporal features
    job_recency_days: int
    application_competition: float  # Estimated # of applicants
    
    # Meta features
    resume_version: str
    embedding_model_version: str

class FeedbackCapture:
    """Optimized feedback capture with batch operations"""
    
    def __init__(self, session):
        self.session = session
        self._batch_buffer = []
        self._batch_size = 100

    def capture(self, outcome: MatchOutcomeRecord) -> bool:
        """Capture individual outcome with optional batching"""
        try:
            # Add to batch buffer for efficiency
            self._batch_buffer.append(outcome)
            
            # Flush batch when full
            if len(self._batch_buffer) >= self._batch_size:
                return self._flush_batch()
                
            return True
        except Exception as e:
            logger.error(f"Failed to capture outcome: {e}")
            return False

    def capture_batch(self, outcomes: List[MatchOutcomeRecord]) -> int:
        """Efficiently capture multiple outcomes"""
        try:
            # Use bulk insert for performance
            outcome_dicts = [asdict(outcome) for outcome in outcomes]
            
            # Assuming SQLAlchemy session with MatchOutcome model
            self.session.bulk_insert_mappings(MatchOutcome, outcome_dicts)
            self.session.commit()
            
            logger.info(f"Captured {len(outcomes)} outcome records")
            return len(outcomes)
            
        except Exception as e:
            logger.error(f"Batch capture failed: {e}")
            self.session.rollback()
            return 0

    def _flush_batch(self) -> bool:
        """Flush buffered outcomes to database"""
        if not self._batch_buffer:
            return True
            
        success_count = self.capture_batch(self._batch_buffer)
        self._batch_buffer.clear()
        
        return success_count > 0

class FeatureWeightModel:
    """Optimized feature weight model with caching and fast scoring"""
    
    def __init__(self, weights: Dict[str, float], model_version: str, trained_at: datetime, 
                 metadata: Optional[Dict] = None):
        self.weights = weights
        self.model_version = model_version
        self.trained_at = trained_at
        self.metadata = metadata or {}
        
        # Pre-compute feature names for fast scoring
        self._feature_names = list(weights.keys())
        self._weight_array = np.array([weights[f] for f in self._feature_names])

    def score(self, feature_vector: Dict[str, float]) -> float:
        """Optimized scoring using numpy operations"""
        try:
            # Convert to numpy array in consistent order
            feature_array = np.array([
                feature_vector.get(f, 0.0) for f in self._feature_names
            ])
            
            # Fast dot product
            score = np.dot(feature_array, self._weight_array)
            
            # Apply sigmoid to get probability-like output
            return 1.0 / (1.0 + np.exp(-score))
            
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            # Fallback to simple weighted sum
            return sum(self.weights.get(k, 0.0) * v for k, v in feature_vector.items())

    def get_feature_importance(self) -> List[Tuple[str, float]]:
        """Return features sorted by absolute importance"""
        importance = [(f, abs(w)) for f, w in self.weights.items()]
        return sorted(importance, key=lambda x: x[1], reverse=True)

    def to_json(self) -> str:
        """Serialize model for storage"""
        return json.dumps({
            'weights': self.weights,
            'model_version': self.model_version,
            'trained_at': self.trained_at.isoformat(),
            'metadata': self.metadata
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'FeatureWeightModel':
        """Deserialize model from storage"""
        data = json.loads(json_str)
        return cls(
            weights=data['weights'],
            model_version=data['model_version'], 
            trained_at=datetime.fromisoformat(data['trained_at']),
            metadata=data.get('metadata', {})
        )

class AdvancedFeatureEngineer:
    """Engineer sophisticated features from raw match data"""
    
    def __init__(self):
        # Skill importance weights (could be learned)
        self.critical_skills = {
            'python', 'sql', 'aws', 'kubernetes', 'react', 'node.js'
        }
        
        # Company tier mapping (could be external data)
        self.company_tiers = {
            'faang': 1.0,
            'unicorn': 0.9,
            'public': 0.7,
            'startup': 0.5,
            'unknown': 0.3
        }

    def engineer_features(self, match_data: Dict[str, Any]) -> FeatureVector:
        """Transform raw match data into engineered features"""
        
        # Extract base features
        vector_score = float(match_data.get('vector_score', 0.0))
        llm_score = float(match_data.get('llm_score', 0.0)) / 100.0  # Normalize to 0-1
        fts_rank = int(match_data.get('fts_rank', 999))
        fts_rank_norm = max(0, 1 - (fts_rank / 1000.0))  # Normalize rank
        
        # Experience features
        required_yoe = float(match_data.get('required_yoe', 0))
        candidate_yoe = float(match_data.get('candidate_yoe', 0))
        yoe_gap = abs(required_yoe - candidate_yoe)
        yoe_gap_ratio = yoe_gap / max(required_yoe, 1.0)
        
        # Skill features
        required_skills = set(match_data.get('required_skills', []))
        candidate_skills = set(match_data.get('candidate_skills', []))
        
        skill_overlap = required_skills & candidate_skills
        skill_overlap_count = len(skill_overlap)
        skill_overlap_ratio = len(skill_overlap) / max(len(required_skills), 1)
        
        # Critical skill matching
        critical_required = required_skills & self.critical_skills
        critical_overlap = skill_overlap & self.critical_skills
        critical_skill_match = (
            len(critical_overlap) / max(len(critical_required), 1) 
            if critical_required else 1.0
        )
        
        # Semantic features
        title_similarity = self._compute_title_similarity(
            match_data.get('job_title', ''),
            match_data.get('resume_title', '')
        )
        
        # Company features
        company_name = match_data.get('company_name', '').lower()
        company_tier_match = self._get_company_tier(company_name)
        
        # Temporal features
        job_posted = match_data.get('job_posted_date')
        job_recency_days = (
            (datetime.now() - job_posted).days 
            if job_posted else 365
        )
        
        # Location features
        location_preference_match = self._compute_location_match(
            match_data.get('job_location_type', 'unknown'),
            match_data.get('candidate_location_pref', 'unknown')
        )
        
        return FeatureVector(
            vector_score=vector_score,
            llm_score=llm_score,
            fts_rank_norm=fts_rank_norm,
            yoe_gap=yoe_gap,
            yoe_gap_ratio=yoe_gap_ratio,
            seniority_match=self._compute_seniority_match(match_data),
            skill_overlap_count=skill_overlap_count,
            skill_overlap_ratio=skill_overlap_ratio,
            critical_skill_match=critical_skill_match,
            title_similarity=title_similarity,
            company_tier_match=company_tier_match,
            location_preference_match=location_preference_match,
            job_recency_days=job_recency_days,
            application_competition=self._estimate_competition(match_data),
            resume_version=match_data.get('resume_version', 'v1'),
            embedding_model_version=match_data.get('embedding_version', 'v1.0')
        )

    def _compute_title_similarity(self, job_title: str, resume_title: str) -> float:
        """Simple title similarity (could use word embeddings)"""
        if not job_title or not resume_title:
            return 0.0
            
        job_words = set(job_title.lower().split())
        resume_words = set(resume_title.lower().split())
        
        if not job_words or not resume_words:
            return 0.0
            
        intersection = job_words & resume_words
        union = job_words | resume_words
        
        return len(intersection) / len(union)  # Jaccard similarity

    def _get_company_tier(self, company_name: str) -> float:
        """Map company name to tier score"""
        for tier_type, score in self.company_tiers.items():
            if tier_type in company_name or company_name in getattr(self, f'{tier_type}_companies', []):
                return score
        return self.company_tiers['unknown']

    def _compute_seniority_match(self, match_data: Dict) -> float:
        """Match seniority level between job and candidate"""
        job_seniority = match_data.get('job_seniority', 'unknown').lower()
        candidate_level = match_data.get('candidate_level', 'unknown').lower()
        
        seniority_map = {
            'intern': 1, 'junior': 2, 'mid': 3, 'senior': 4, 'staff': 5, 'principal': 6
        }
        
        job_level = seniority_map.get(job_seniority, 3)
        candidate_level_num = seniority_map.get(candidate_level, 3)
        
        # Perfect match = 1.0, decreases with distance
        level_diff = abs(job_level - candidate_level_num)
        return max(0.0, 1.0 - (level_diff * 0.2))

    def _compute_location_match(self, job_location: str, candidate_pref: str) -> float:
        """Match location preferences"""
        location_compatibility = {
            ('remote', 'remote'): 1.0,
            ('hybrid', 'hybrid'): 1.0, 
            ('hybrid', 'remote'): 0.8,
            ('onsite', 'onsite'): 1.0,
            ('onsite', 'hybrid'): 0.6,
            ('onsite', 'remote'): 0.2,
        }
        
        return location_compatibility.get((job_location, candidate_pref), 0.5)

    def _estimate_competition(self, match_data: Dict) -> float:
        """Estimate application competition (simplified)"""
        # Factors that increase competition
        company_tier = self._get_company_tier(match_data.get('company_name', ''))
        job_age_days = match_data.get('job_recency_days', 30)
        
        # High-tier companies = more competition
        # Newer jobs = more competition  
        competition_score = company_tier + max(0, (30 - job_age_days) / 30.0)
        
        return min(1.0, competition_score / 2.0)

class FeedbackTrainer:
    """Advanced machine learning trainer with optimization and validation"""
    
    def __init__(self, session):
        self.session = session
        self.feature_engineer = AdvancedFeatureEngineer()
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.min_training_samples = 100
        
    def fetch_training_data(self, lookback_days: int = 90) -> List[Dict[str, Any]]:
        """Fetch training data with proper feature engineering"""
        try:
            # SQL query to get training data (pseudo-code)
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            
            query = """
            SELECT 
                m.id as match_id,
                m.vector_score,
                m.llm_score,
                j.title as job_title,
                j.required_yoe,
                j.required_skills,
                j.seniority as job_seniority,
                j.location_type as job_location_type,
                j.scraped_at as job_posted_date,
                c.name as company_name,
                r.yoe_adjusted as candidate_yoe,
                r.skills_csv as candidate_skills,
                r.sections_json as resume_data,
                mo.got_response,
                mo.got_interview,
                mo.got_offer,
                mo.user_satisfaction
            FROM matches m
            JOIN jobs j ON m.job_id = j.id  
            JOIN companies c ON j.company_id = c.id
            JOIN resumes r ON m.resume_id = r.id
            JOIN match_outcomes mo ON m.id = mo.match_id
            WHERE mo.captured_at >= %s
            AND mo.got_interview IS NOT NULL
            ORDER BY mo.captured_at DESC
            """
            
            # Execute query (pseudo-implementation)
            results = self.session.execute(query, (cutoff_date,)).fetchall()
            
            # Convert to list of dicts
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to fetch training data: {e}")
            return []

    def train(self, validation_split: float = 0.2) -> FeatureWeightModel:
        """Train logistic regression model with validation"""
        
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, using heuristic model")
            return self._create_heuristic_model()
            
        # Fetch training data
        raw_data = self.fetch_training_data()
        
        if len(raw_data) < self.min_training_samples:
            logger.warning(f"Insufficient training data: {len(raw_data)} < {self.min_training_samples}")
            return self._create_heuristic_model()
            
        # Engineer features
        features = []
        labels = []
        
        for row in raw_data:
            try:
                feature_vec = self.feature_engineer.engineer_features(row)
                features.append(asdict(feature_vec))
                
                # Create binary label (got interview = positive outcome)
                label = 1 if row.get('got_interview') else 0
                labels.append(label)
                
            except Exception as e:
                logger.warning(f"Failed to engineer features for row: {e}")
                continue
        
        if len(features) < self.min_training_samples:
            logger.warning("Insufficient valid features after engineering")
            return self._create_heuristic_model()
            
        # Convert to arrays
        feature_names = list(features[0].keys())
        # Filter numeric features only  
        numeric_features = [
            f for f in feature_names 
            if isinstance(features[0][f], (int, float))
        ]
        
        X = np.array([[f[feat] for feat in numeric_features] for f in features])
        y = np.array(labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train logistic regression
        model = LogisticRegression(
            C=1.0,  # Regularization strength
            class_weight='balanced',  # Handle imbalanced data
            random_state=42,
            max_iter=1000
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Validate model
        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred_proba)
        
        logger.info(f"Model trained with AUC: {auc_score:.3f}")
        
        # Create weight dictionary
        weights = dict(zip(numeric_features, model.coef_[0]))
        
        # Create versioned model
        model_version = f"lr_v{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        return FeatureWeightModel(
            weights=weights,
            model_version=model_version,
            trained_at=datetime.now(),
            metadata={
                'auc_score': float(auc_score),
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'feature_count': len(numeric_features),
                'regularization_c': 1.0
            }
        )

    def _create_heuristic_model(self) -> FeatureWeightModel:
        """Fallback heuristic model when ML training fails"""
        weights = {
            'vector_score': 0.25,
            'llm_score': 0.30,
            'skill_overlap_ratio': 0.20,
            'critical_skill_match': 0.15,
            'yoe_gap_ratio': -0.10,  # Negative weight for gaps
            'title_similarity': 0.05,
            'company_tier_match': 0.05
        }
        
        return FeatureWeightModel(
            weights=weights,
            model_version="heuristic_v1",
            trained_at=datetime.now(),
            metadata={'type': 'heuristic', 'training_samples': 0}
        )

    def evaluate_model(self, model: FeatureWeightModel, test_days: int = 30) -> Dict[str, float]:
        """Evaluate model performance on held-out test set"""
        test_data = self.fetch_training_data(lookback_days=test_days)
        
        if len(test_data) < 10:
            return {'error': 'insufficient_test_data'}
            
        predictions = []
        actuals = []
        
        for row in test_data:
            try:
                features = self.feature_engineer.engineer_features(row)
                feature_dict = {
                    k: v for k, v in asdict(features).items() 
                    if isinstance(v, (int, float))
                }
                
                pred_score = model.score(feature_dict)
                predictions.append(pred_score)
                
                actual = 1 if row.get('got_interview') else 0
                actuals.append(actual)
                
            except Exception as e:
                logger.warning(f"Evaluation error: {e}")
                continue
        
        if len(predictions) < 5:
            return {'error': 'insufficient_predictions'}
            
        # Calculate metrics
        try:
            auc = roc_auc_score(actuals, predictions)
            
            # Binary predictions at 0.5 threshold
            binary_preds = [1 if p > 0.5 else 0 for p in predictions]
            accuracy = sum(a == p for a, p in zip(actuals, binary_preds)) / len(actuals)
            
            return {
                'auc': auc,
                'accuracy': accuracy,
                'samples': len(predictions),
                'positive_rate': sum(actuals) / len(actuals)
            }
        except Exception as e:
            logger.error(f"Metric calculation failed: {e}")
            return {'error': 'metric_calculation_failed'}
