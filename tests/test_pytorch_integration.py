"""Test PyTorch integration and modular components."""

import pytest
import torch
import numpy as np
from libs.embed import EmbeddingProvider, OpenAIEmbeddingProvider
from libs.nlp.chunkers import SemanticChunker, FixedSizeChunker
from libs.nlp.extractors import SkillExtractor, YearsOfExperienceExtractor


def test_pytorch_installation():
    """Test that PyTorch is properly installed and functional."""
    # Test basic tensor operations
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    z = torch.dot(x, y)
    
    assert z.item() == 32.0  # 1*4 + 2*5 + 3*6 = 32
    
    # Test that we can use PyTorch for vector operations
    embedding_dim = 1536  # OpenAI embedding dimension
    fake_embedding1 = torch.randn(embedding_dim)
    fake_embedding2 = torch.randn(embedding_dim)
    
    # Cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(
        fake_embedding1.unsqueeze(0), 
        fake_embedding2.unsqueeze(0)
    )
    
    assert -1.0 <= cosine_sim.item() <= 1.0


def test_semantic_chunker():
    """Test the semantic chunker component."""
    chunker = SemanticChunker(target_chunk_size=200, max_chunk_size=400)
    
    sample_resume = """
    PROFESSIONAL SUMMARY
    Experienced software engineer with 5+ years developing scalable web applications.
    
    EXPERIENCE
    Senior Software Engineer - TechCorp (2020-2023)
    • Developed microservices using Python and Django
    • Led team of 3 engineers on critical projects
    • Improved system performance by 40%
    
    Software Engineer - StartupInc (2018-2020)  
    • Built REST APIs and database systems
    • Worked with React and Node.js frontends
    
    EDUCATION
    Bachelor of Science in Computer Science - StateU (2018)
    
    SKILLS
    Python, JavaScript, Django, React, PostgreSQL, Docker
    """
    
    chunks = chunker.chunk_text(sample_resume, {'doc_type': 'resume'})
    
    assert len(chunks) > 1
    assert all('text' in chunk for chunk in chunks)
    assert all('index' in chunk for chunk in chunks)
    
    # Should preserve section structure
    chunk_texts = [chunk['text'] for chunk in chunks]
    resume_sections = ['SUMMARY', 'EXPERIENCE', 'EDUCATION', 'SKILLS']
    
    found_sections = sum(1 for section in resume_sections 
                        if any(section in text for text in chunk_texts))
    assert found_sections >= 2  # Should find at least 2 sections


def test_skill_extractor():
    """Test the skill extraction component."""
    extractor = SkillExtractor()
    
    sample_text = """
    I have 5 years of experience with Python and Django.
    Proficient in PostgreSQL, Docker, and AWS.
    Built React applications and REST APIs.
    """
    
    skills = extractor.extract_skills(sample_text, min_confidence=0.6)
    
    assert len(skills) > 0
    
    # Check that we found some expected skills
    skill_names = [skill.skill for skill in skills]
    expected_skills = ['python', 'django', 'postgresql', 'docker', 'aws', 'react']
    
    found_skills = [skill for skill in expected_skills if skill in skill_names]
    assert len(found_skills) >= 3  # Should find at least 3 skills
    
    # Test skill categorization
    categories = set(skill.category for skill in skills)
    assert len(categories) > 1  # Should have multiple categories


def test_experience_extractor():
    """Test the years of experience extraction."""
    extractor = YearsOfExperienceExtractor()
    
    sample_text = """
    Senior Software Engineer with 8 years of experience.
    Master's degree in Computer Science from MIT.
    Previously worked at Google from 2018-2023.
    """
    
    experience = extractor.extract_experience(sample_text)
    
    assert 'raw_experience' in experience
    assert 'education_bonus' in experience
    assert 'total_experience' in experience
    
    # Should find around 8 years raw experience
    assert experience['raw_experience'] >= 5
    
    # Should have education bonus for Master's degree
    assert experience['education_bonus'] >= 2
    
    # Total should be sum of raw + bonus
    assert experience['total_experience'] >= experience['raw_experience']


def test_pytorch_vector_similarity():
    """Test using PyTorch for vector similarity computations."""
    # Simulate embeddings from our system
    dim = 1536
    
    # Create some test embeddings
    resume_embedding = torch.randn(dim)
    job_embedding1 = torch.randn(dim)  
    job_embedding2 = resume_embedding + torch.randn(dim) * 0.1  # Similar to resume
    
    # Normalize embeddings (typical for similarity computation)
    resume_norm = torch.nn.functional.normalize(resume_embedding, dim=0)
    job1_norm = torch.nn.functional.normalize(job_embedding1, dim=0)
    job2_norm = torch.nn.functional.normalize(job_embedding2, dim=0) 
    
    # Compute similarities
    sim1 = torch.dot(resume_norm, job1_norm).item()
    sim2 = torch.dot(resume_norm, job2_norm).item()
    
    # job2 should be more similar to resume (higher similarity)
    assert sim2 > sim1
    
    # Test batch similarity computation
    job_embeddings = torch.stack([job1_norm, job2_norm])
    similarities = torch.mm(job_embeddings, resume_norm.unsqueeze(1)).squeeze()
    
    assert len(similarities) == 2
    assert similarities[1] > similarities[0]  # job2 more similar


@pytest.mark.skip(reason="Requires OpenAI API key")
def test_openai_embedding_provider():
    """Test OpenAI embedding provider (requires API key)."""
    # This test is skipped by default since it requires API key
    # Uncomment and set OPENAI_API_KEY to run
    
    provider = OpenAIEmbeddingProvider()
    
    test_texts = [
        "Python software engineer with Django experience",
        "Data scientist with machine learning background", 
        "Frontend developer using React and TypeScript"
    ]
    
    # Test single embedding
    embedding1 = provider.embed_text(test_texts[0])
    assert embedding1.shape == (1536,)  # text-embedding-3-large dimension
    
    # Test batch embedding
    embeddings = provider.embed_batch(test_texts)
    assert len(embeddings) == 3
    assert all(emb.shape == (1536,) for emb in embeddings)
    
    # Test similarity computation
    sim = provider.cosine_similarity(embeddings[0], embeddings[1])
    assert -1.0 <= sim <= 1.0


def test_pytorch_model_mock():
    """Test a mock ML model using PyTorch for demonstration."""
    # This shows how we could use PyTorch for custom models
    # For example, a job-resume compatibility scorer
    
    class SimpleCompatibilityModel(torch.nn.Module):
        def __init__(self, embedding_dim=1536):
            super().__init__()
            self.embedding_dim = embedding_dim
            
            # Simple network to score job-resume compatibility
            self.compatibility_net = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim * 2, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(512, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1),
                torch.nn.Sigmoid()  # Output between 0-1
            )
        
        def forward(self, resume_embedding, job_embedding):
            # Concatenate embeddings
            combined = torch.cat([resume_embedding, job_embedding], dim=-1)
            compatibility_score = self.compatibility_net(combined)
            return compatibility_score
    
    # Test the model
    model = SimpleCompatibilityModel()
    
    # Create mock embeddings
    batch_size = 4
    embedding_dim = 1536
    
    resume_embeddings = torch.randn(batch_size, embedding_dim)
    job_embeddings = torch.randn(batch_size, embedding_dim)
    
    # Forward pass
    scores = model(resume_embeddings, job_embeddings)
    
    assert scores.shape == (batch_size, 1)
    assert torch.all(scores >= 0) and torch.all(scores <= 1)
    
    # Test that the model parameters can be updated
    initial_params = list(model.parameters())[0].clone()
    
    # Dummy training step
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    
    # Mock target scores
    target_scores = torch.rand(batch_size, 1)
    
    # Forward + backward pass
    optimizer.zero_grad()
    predicted_scores = model(resume_embeddings, job_embeddings)
    loss = loss_fn(predicted_scores, target_scores)
    loss.backward()
    optimizer.step()
    
    # Parameters should have changed
    updated_params = list(model.parameters())[0]
    assert not torch.equal(initial_params, updated_params)


if __name__ == "__main__":
    # Run basic tests manually if needed
    test_pytorch_installation()
    test_semantic_chunker()
    test_skill_extractor()
    test_experience_extractor()
    test_pytorch_vector_similarity()
    test_pytorch_model_mock()
    print("All tests passed!")