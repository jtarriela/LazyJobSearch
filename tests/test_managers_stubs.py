from libs.embed.versioning import EmbeddingVersionManager
from libs.matching.feedback import FeedbackTrainer, FeatureWeightModel
from libs.scraper.anti_bot import ProxyPool, FingerprintGenerator, HumanBehaviorSimulator, ScrapeSessionManager

class DummySession:
    pass

def test_embedding_version_manager_active_version():
    mgr = EmbeddingVersionManager(DummySession())
    info = mgr.get_active_version()
    assert info.version_id.startswith("v")
    assert info.dimensions > 0

def test_feedback_trainer_returns_model():
    trainer = FeedbackTrainer(DummySession())
    model = trainer.train()
    assert isinstance(model, FeatureWeightModel)


def test_scrape_session_manager_start_finish():
    ppm = ProxyPool(["proxy1", "proxy2"])  # simple list
    fpg = FingerprintGenerator()
    sm = ScrapeSessionManager(ppm, fpg)
    session = sm.start()
    assert session.profile.user_agent
    finished = sm.finish(session, "ok")
    assert finished.outcome == "ok"
