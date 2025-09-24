#!/usr/bin/env python3
"""
Script to test jdtarriela resume against Anduril-style jobs.
This implements the core functionality requested in the problem statement.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import PyPDF2
from rich.console import Console
from rich.table import Table

console = Console()

class ResumeParser:
    """Basic resume parser for PDF files"""
    
    def __init__(self, resume_path: str):
        self.resume_path = Path(resume_path)
        self.text = ""
        self.skills = set()
        self.experience_years = 0
        
    def parse(self):
        """Extract text and basic information from PDF resume"""
        try:
            with open(self.resume_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text()
                self.text = text_content.lower()
                
            self._extract_skills()
            self._extract_experience()
            
        except Exception as e:
            console.print(f"[red]Error parsing resume: {e}[/red]")
            return False
        return True
    
    def _extract_skills(self):
        """Extract technical skills from resume text"""
        # Common tech skills that might be relevant to Anduril
        skill_keywords = [
            # Programming Languages
            'python', 'java', 'c++', 'c#', 'javascript', 'typescript', 'go', 'rust',
            'matlab', 'sql', 'r', 'scala', 'kotlin', 'swift',
            
            # Frameworks & Tools
            'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring',
            'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
            'docker', 'kubernetes', 'aws', 'gcp', 'azure', 'terraform',
            'jenkins', 'gitlab', 'github', 'jira',
            
            # Defense/Aerospace relevant
            'ros', 'opencv', 'control systems', 'embedded systems', 'real-time',
            'simulation', 'modeling', 'autonomy', 'machine learning', 'computer vision',
            'robotics', 'sensor fusion', 'slam', 'path planning',
            
            # General
            'agile', 'scrum', 'devops', 'microservices', 'api', 'rest',
            'graphql', 'mongodb', 'postgresql', 'redis', 'elasticsearch'
        ]
        
        for skill in skill_keywords:
            if skill in self.text:
                self.skills.add(skill)
    
    def _extract_experience(self):
        """Rough estimate of years of experience"""
        # Look for years mentioned in context of experience
        import re
        year_patterns = [
            r'(\d+)\+?\s*years?\s*(of\s*)?experience',
            r'experience.*?(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*in'
        ]
        
        max_years = 0
        for pattern in year_patterns:
            matches = re.findall(pattern, self.text, re.IGNORECASE)
            for match in matches:
                try:
                    years = int(match[0] if isinstance(match, tuple) else match)
                    max_years = max(max_years, years)
                except ValueError:
                    continue
        
        self.experience_years = max_years

class AndurilJobMatcher:
    """Matches resume against typical Anduril job requirements"""
    
    def __init__(self):
        # Mock Anduril job postings based on their typical requirements
        self.jobs = [
            {
                "id": "senior-software-engineer-1",
                "title": "Senior Software Engineer - Autonomous Systems",
                "department": "Engineering",
                "location": "Costa Mesa, CA",
                "description": """
                We're looking for a Senior Software Engineer to work on autonomous systems
                that will help keep our service members and allies safe. You'll be working
                on real-time control systems, computer vision, and machine learning algorithms.
                
                Requirements:
                - 5+ years of software engineering experience
                - Strong proficiency in Python, C++, or similar languages
                - Experience with robotics, computer vision, or autonomous systems
                - Knowledge of ROS, OpenCV, or similar frameworks
                - Experience with machine learning frameworks (TensorFlow, PyTorch)
                - Security clearance eligible
                """,
                "required_skills": ["python", "c++", "computer vision", "machine learning", 
                                  "robotics", "ros", "opencv", "tensorflow", "pytorch"],
                "min_experience": 5,
                "seniority": "senior"
            },
            {
                "id": "ml-engineer-1", 
                "title": "Machine Learning Engineer - Perception",
                "department": "AI/ML",
                "location": "Seattle, WA",
                "description": """
                Join our Perception team to develop ML models for autonomous defense systems.
                You'll work on computer vision, sensor fusion, and real-time inference.
                
                Requirements:
                - 3+ years in machine learning or computer vision
                - PhD or MS in related field preferred
                - Experience with deep learning frameworks
                - Python, MATLAB, or C++ proficiency
                - Experience with sensor data processing
                """,
                "required_skills": ["machine learning", "python", "computer vision", 
                                  "deep learning", "tensorflow", "pytorch", "matlab", "c++"],
                "min_experience": 3,
                "seniority": "mid"
            },
            {
                "id": "embedded-engineer-1",
                "title": "Embedded Software Engineer",
                "department": "Hardware",
                "location": "Orange County, CA", 
                "description": """
                Develop embedded software for our cutting-edge defense platforms.
                Work on real-time systems, device drivers, and low-level optimization.
                
                Requirements:
                - 4+ years embedded systems experience
                - C/C++ proficiency required
                - Real-time systems experience
                - Knowledge of embedded Linux, device drivers
                - Hardware debugging skills
                """,
                "required_skills": ["c++", "c", "embedded systems", "real-time", "linux"],
                "min_experience": 4,
                "seniority": "senior"
            },
            {
                "id": "devops-engineer-1",
                "title": "DevOps Engineer - Platform",
                "department": "Infrastructure", 
                "location": "Remote",
                "description": """
                Build and maintain the infrastructure that powers our autonomous systems.
                Kubernetes, AWS, and CI/CD pipelines are your specialty.
                
                Requirements:
                - 3+ years DevOps/Infrastructure experience  
                - Kubernetes and Docker experience
                - AWS/GCP cloud platforms
                - Infrastructure as Code (Terraform)
                - CI/CD pipeline experience
                """,
                "required_skills": ["kubernetes", "docker", "aws", "gcp", "terraform", 
                                  "devops", "python", "bash"],
                "min_experience": 3,
                "seniority": "mid"
            }
        ]
    
    def calculate_match_score(self, resume: ResumeParser, job: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate how well the resume matches a job posting"""
        
        # Skill matching
        job_skills = set(job["required_skills"])
        resume_skills = resume.skills
        matching_skills = job_skills.intersection(resume_skills)
        skill_match_ratio = len(matching_skills) / len(job_skills) if job_skills else 0
        
        # Experience matching
        experience_match = min(resume.experience_years / job["min_experience"], 1.0) if job["min_experience"] > 0 else 0
        
        # Overall score (weighted)
        overall_score = (skill_match_ratio * 0.7) + (experience_match * 0.3)
        
        return {
            "job_id": job["id"],
            "title": job["title"],
            "overall_score": round(overall_score * 100, 1),
            "skill_match_ratio": round(skill_match_ratio * 100, 1),
            "experience_match": round(experience_match * 100, 1),
            "matching_skills": list(matching_skills),
            "missing_skills": list(job_skills - resume_skills),
            "location": job["location"],
            "department": job["department"]
        }
    
    def match_all_jobs(self, resume: ResumeParser) -> List[Dict[str, Any]]:
        """Match resume against all Anduril jobs"""
        matches = []
        for job in self.jobs:
            match_result = self.calculate_match_score(resume, job)
            matches.append(match_result)
        
        # Sort by overall score descending
        return sorted(matches, key=lambda x: x["overall_score"], reverse=True)

def main():
    """Main function to run the resume matching test"""
    
    # Path to the jdtarriela resume
    resume_path = "/home/runner/work/LazyJobSearch/LazyJobSearch/tests/jtarriela_resume[sp].pdf"
    
    console.print("[bold blue]LazyJobSearch: JD Tarriela Resume vs Anduril Careers Analysis[/bold blue]")
    console.print()
    
    # Parse the resume
    console.print("📄 Parsing resume...")
    resume = ResumeParser(resume_path)
    if not resume.parse():
        console.print("[red]Failed to parse resume[/red]")
        return 1
    
    # Display resume summary
    console.print(f"✅ Resume parsed successfully")
    console.print(f"📊 Extracted {len(resume.skills)} technical skills")
    console.print(f"⏱️  Estimated experience: {resume.experience_years} years")
    console.print()
    
    # Display found skills
    if resume.skills:
        skills_table = Table(title="🔧 Technical Skills Found")
        skills_table.add_column("Skills", style="cyan")
        skills_list = sorted(list(resume.skills))
        for i in range(0, len(skills_list), 5):
            row_skills = skills_list[i:i+5]
            skills_table.add_row(", ".join(row_skills))
        console.print(skills_table)
        console.print()
    
    # Match against Anduril jobs
    console.print("🎯 Matching against Anduril career opportunities...")
    matcher = AndurilJobMatcher()
    matches = matcher.match_all_jobs(resume)
    
    # Display results
    results_table = Table(title="🏆 Job Match Results (Ranked by Compatibility)")
    results_table.add_column("Rank", style="bold")
    results_table.add_column("Job Title", style="bold cyan")
    results_table.add_column("Department", style="yellow")
    results_table.add_column("Location", style="green")  
    results_table.add_column("Overall Score", style="bold red")
    results_table.add_column("Skill Match", style="blue")
    results_table.add_column("Experience Match", style="magenta")
    
    for i, match in enumerate(matches, 1):
        results_table.add_row(
            str(i),
            match["title"],
            match["department"],
            match["location"],
            f"{match['overall_score']}%",
            f"{match['skill_match_ratio']}%",
            f"{match['experience_match']}%"
        )
    
    console.print(results_table)
    console.print()
    
    # Show detailed analysis for top match
    if matches:
        top_match = matches[0]
        console.print(f"[bold green]🥇 Top Match Analysis: {top_match['title']}[/bold green]")
        console.print(f"📍 Location: {top_match['location']}")
        console.print(f"🏢 Department: {top_match['department']}")
        console.print(f"📈 Overall Compatibility: {top_match['overall_score']}%")
        console.print()
        
        if top_match["matching_skills"]:
            console.print("[green]✅ Matching Skills:[/green]")
            for skill in sorted(top_match["matching_skills"]):
                console.print(f"  • {skill}")
            console.print()
        
        if top_match["missing_skills"]:
            console.print("[yellow]⚠️  Skills to Develop:[/yellow]")
            for skill in sorted(top_match["missing_skills"]):
                console.print(f"  • {skill}")
            console.print()
    
    # Recommendations
    console.print("[bold yellow]💡 Recommendations:[/bold yellow]")
    if matches:
        avg_score = sum(m["overall_score"] for m in matches) / len(matches)
        if avg_score >= 70:
            console.print("• Strong candidate profile for Anduril positions")
        elif avg_score >= 50:
            console.print("• Good potential - focus on developing missing technical skills")
        else:
            console.print("• Consider gaining experience in autonomous systems, defense, or related domains")
            
        # Find most common missing skills across all jobs
        all_missing = []
        for match in matches:
            all_missing.extend(match["missing_skills"])
        
        from collections import Counter
        common_missing = Counter(all_missing).most_common(3)
        if common_missing:
            console.print("• Priority skill development areas:")
            for skill, count in common_missing:
                console.print(f"  - {skill} (required in {count} of {len(matches)} positions)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())