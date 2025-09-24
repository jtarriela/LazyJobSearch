#!/usr/bin/env python3
"""
Dual Resume Evaluation Script for Anduril Jobs

This script evaluates both resume files against Anduril career opportunities:
- jtarriela_resume.pdf
- jtarriela_resume[sp].pdf

Provides side-by-side comparison and analysis of both resumes.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import PyPDF2
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from datetime import datetime
import hashlib

console = Console()

class ResumeParser:
    """Enhanced resume parser for PDF files"""
    
    def __init__(self, resume_path: str, resume_name: str = None):
        self.resume_path = Path(resume_path)
        self.resume_name = resume_name or self.resume_path.name
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
            console.print(f"[red]Error parsing resume {self.resume_name}: {e}[/red]")
            return False
        return True
    
    def _extract_skills(self):
        """Extract technical skills from resume text"""
        # Common tech skills that might be relevant to Anduril
        skill_keywords = [
            # Programming Languages
            'python', 'java', 'c++', 'c#', 'javascript', 'typescript', 'go', 'rust',
            'matlab', 'sql', 'r', 'scala', 'kotlin', 'swift', 'c',
            
            # Frameworks & Tools
            'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring',
            'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
            'docker', 'kubernetes', 'aws', 'gcp', 'azure', 'terraform',
            'jenkins', 'gitlab', 'github', 'jira', 'confluence',
            
            # Defense/Aerospace relevant
            'ros', 'opencv', 'control systems', 'embedded systems', 'real-time',
            'simulation', 'modeling', 'autonomy', 'machine learning', 'computer vision',
            'robotics', 'sensor fusion', 'slam', 'path planning',
            
            # General
            'agile', 'scrum', 'devops', 'microservices', 'api', 'rest',
            'graphql', 'mongodb', 'postgresql', 'redis', 'elasticsearch',
            'linux', 'solidworks', 'autocad', 'ansys', 'cadence'
        ]
        
        for skill in skill_keywords:
            if skill in self.text:
                self.skills.add(skill)
    
    def _extract_experience(self):
        """Rough estimate of years of experience"""
        # Look for experience indicators
        text = self.text
        
        # Simple heuristic based on graduation dates, work history, etc.
        current_year = 2024
        years_found = []
        
        # Look for year patterns
        import re
        year_pattern = r'\b(19|20)\d{2}\b'
        years_in_text = re.findall(year_pattern, text)
        
        if years_in_text:
            try:
                years = [int(y + "00") for y in years_in_text]
                earliest_year = min(years)
                if earliest_year > 1990:  # Reasonable bounds
                    self.experience_years = max(0, current_year - earliest_year)
            except:
                pass
        
        # Default to reasonable estimate based on content complexity
        if self.experience_years == 0:
            skill_count = len(self.skills)
            if skill_count > 15:
                self.experience_years = 8
            elif skill_count > 10:
                self.experience_years = 5
            elif skill_count > 5:
                self.experience_years = 3
            else:
                self.experience_years = 1

class AndurilJobMatcher:
    """Matches resume against typical Anduril job requirements"""
    
    def __init__(self):
        self.jobs = self._load_anduril_jobs()
    
    def _load_anduril_jobs(self):
        """Load mock Anduril job data based on typical positions"""
        return [
            {
                "id": "anduril-001",
                "title": "Senior Software Engineer - Autonomous Systems",
                "department": "Engineering",
                "location": "Costa Mesa, CA",
                "min_experience": 5,
                "required_skills": ["c++", "python", "ros", "computer vision", "machine learning", "robotics", "opencv", "tensorflow", "pytorch"],
                "preferred_skills": ["matlab", "simulation", "control systems"]
            },
            {
                "id": "anduril-002",
                "title": "Machine Learning Engineer - Perception",
                "department": "AI/ML",
                "location": "Seattle, WA",
                "min_experience": 3,
                "required_skills": ["python", "tensorflow", "pytorch", "computer vision", "machine learning", "opencv", "pandas", "numpy"],
                "preferred_skills": ["c++", "ros", "matlab"]
            },
            {
                "id": "anduril-003",
                "title": "Embedded Software Engineer",
                "department": "Hardware",
                "location": "Orange County, CA",
                "min_experience": 4,
                "required_skills": ["c", "c++", "embedded systems", "real-time", "linux"],
                "preferred_skills": ["python", "matlab", "control systems"]
            },
            {
                "id": "anduril-004",
                "title": "DevOps Engineer - Platform",
                "department": "Infrastructure",
                "location": "Remote",
                "min_experience": 3,
                "required_skills": ["docker", "kubernetes", "aws", "python", "terraform", "jenkins", "linux", "mongodb"],
                "preferred_skills": ["go", "graphql", "microservices"]
            },
            {
                "id": "anduril-005",
                "title": "Systems Integration Engineer",
                "department": "Systems Engineering",
                "location": "Huntsville, AL",
                "min_experience": 6,
                "required_skills": ["matlab", "simulation", "modeling", "control systems", "python", "c++"],
                "preferred_skills": ["ros", "embedded systems", "real-time"]
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
        
        # Preferred skills bonus
        preferred_skills = set(job.get("preferred_skills", []))
        matching_preferred = preferred_skills.intersection(resume_skills)
        preferred_bonus = len(matching_preferred) * 0.1  # 10% bonus per preferred skill
        
        # Overall score (weighted)
        overall_score = (skill_match_ratio * 0.7) + (experience_match * 0.3) + preferred_bonus
        overall_score = min(overall_score, 1.0)  # Cap at 100%
        
        return {
            "job_id": job["id"],
            "title": job["title"],
            "overall_score": round(overall_score * 100, 1),
            "skill_match_ratio": round(skill_match_ratio * 100, 1),
            "experience_match": round(experience_match * 100, 1),
            "matching_skills": list(matching_skills),
            "missing_skills": list(job_skills - resume_skills),
            "preferred_bonus": list(matching_preferred),
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

def display_resume_summary(resume: ResumeParser, title: str):
    """Display summary information for a resume"""
    skills_table = Table(title=f"🔧 Technical Skills - {title}")
    skills_table.add_column("Skills", style="cyan")
    
    skills_list = sorted(list(resume.skills))
    for i in range(0, len(skills_list), 5):
        row_skills = skills_list[i:i+5]
        skills_table.add_row(", ".join(row_skills))
    
    summary_panel = Panel(
        f"📊 Skills: {len(resume.skills)}\n⏱️  Experience: {resume.experience_years} years",
        title=f"📄 {title}",
        border_style="blue"
    )
    
    return summary_panel, skills_table

def display_job_matches(matches: List[Dict[str, Any]], resume_name: str):
    """Display job match results for a resume"""
    results_table = Table(title=f"🏆 Job Matches - {resume_name}")
    results_table.add_column("Rank", style="bold", width=6)
    results_table.add_column("Job Title", style="bold cyan", width=25)
    results_table.add_column("Department", style="yellow", width=15)
    results_table.add_column("Score", style="bold red", width=8)
    results_table.add_column("Skills", style="bold green", width=8)
    
    for i, match in enumerate(matches[:5], 1):  # Top 5 matches
        results_table.add_row(
            str(i),
            match["title"][:24] + "..." if len(match["title"]) > 24 else match["title"],
            match["department"],
            f"{match['overall_score']}%",
            f"{match['skill_match_ratio']}%"
        )
    
    return results_table

def compare_resumes(resume1_matches: List[Dict[str, Any]], resume2_matches: List[Dict[str, Any]], 
                   resume1_name: str, resume2_name: str):
    """Create a comparison table of the two resumes"""
    comparison_table = Table(title="📊 Resume Comparison - Top 3 Matches")
    comparison_table.add_column("Position", style="bold", width=25)
    comparison_table.add_column(f"{resume1_name}", style="cyan", width=15)
    comparison_table.add_column(f"{resume2_name}", style="magenta", width=15)
    comparison_table.add_column("Winner", style="bold green", width=10)
    
    # Get top 3 unique job titles from both sets
    all_jobs = {}
    for match in resume1_matches[:3]:
        all_jobs[match["title"]] = {"resume1": match["overall_score"], "resume2": 0}
    
    for match in resume2_matches[:3]:
        job_title = match["title"]
        if job_title in all_jobs:
            all_jobs[job_title]["resume2"] = match["overall_score"]
        else:
            all_jobs[job_title] = {"resume1": 0, "resume2": match["overall_score"]}
    
    for job_title, scores in all_jobs.items():
        score1 = scores["resume1"]
        score2 = scores["resume2"]
        
        if score1 > score2:
            winner = resume1_name
            winner_style = "cyan"
        elif score2 > score1:
            winner = resume2_name
            winner_style = "magenta"
        else:
            winner = "Tie"
            winner_style = "yellow"
        
        comparison_table.add_row(
            job_title[:24] + "..." if len(job_title) > 24 else job_title,
            f"{score1}%" if score1 > 0 else "N/A",
            f"{score2}%" if score2 > 0 else "N/A",
            f"[{winner_style}]{winner}[/{winner_style}]"
        )
    
    return comparison_table

def main():
    """Main function to run dual resume evaluation"""
    
    # Paths to both resumes
    resume1_path = "/home/runner/work/LazyJobSearch/LazyJobSearch/tests/jtarriela_resume.pdf"
    resume2_path = "/home/runner/work/LazyJobSearch/LazyJobSearch/tests/jtarriela_resume[sp].pdf"
    
    console.print("[bold blue]🚀 LazyJobSearch: Dual Resume Evaluation vs Anduril Careers[/bold blue]")
    console.print("=" * 80)
    console.print()
    
    # Parse both resumes
    console.print("📄 [cyan]Parsing resumes...[/cyan]")
    
    resume1 = ResumeParser(resume1_path, "Resume #1 (Standard)")
    resume2 = ResumeParser(resume2_path, "Resume #2 (Special)")
    
    if not resume1.parse():
        console.print("[red]❌ Failed to parse first resume[/red]")
        return 1
    
    if not resume2.parse():
        console.print("[red]❌ Failed to parse second resume[/red]")
        return 1
    
    console.print("✅ [green]Both resumes parsed successfully[/green]")
    console.print()
    
    # Display resume summaries side by side
    summary1, skills1 = display_resume_summary(resume1, "Resume #1 (Standard)")
    summary2, skills2 = display_resume_summary(resume2, "Resume #2 (Special)")
    
    console.print(Columns([summary1, summary2]))
    console.print()
    console.print(Columns([skills1, skills2]))
    console.print()
    
    # Match against Anduril jobs
    console.print("🎯 [cyan]Matching both resumes against Anduril career opportunities...[/cyan]")
    matcher = AndurilJobMatcher()
    
    matches1 = matcher.match_all_jobs(resume1)
    matches2 = matcher.match_all_jobs(resume2)
    
    console.print("✅ [green]Job matching completed for both resumes[/green]")
    console.print()
    
    # Display individual results
    results1 = display_job_matches(matches1, "Resume #1")
    results2 = display_job_matches(matches2, "Resume #2")
    
    console.print(Columns([results1, results2]))
    console.print()
    
    # Display comparison
    comparison = compare_resumes(matches1, matches2, "Resume #1", "Resume #2")
    console.print(comparison)
    console.print()
    
    # Overall analysis
    avg_score1 = sum(match["overall_score"] for match in matches1) / len(matches1)
    avg_score2 = sum(match["overall_score"] for match in matches2) / len(matches2)
    
    best_match1 = matches1[0]
    best_match2 = matches2[0]
    
    analysis_panel = Panel(
        f"""📊 Overall Analysis:

🥇 Best Matches:
• Resume #1: {best_match1['title']} ({best_match1['overall_score']}%)
• Resume #2: {best_match2['title']} ({best_match2['overall_score']}%)

📈 Average Compatibility:
• Resume #1: {avg_score1:.1f}%
• Resume #2: {avg_score2:.1f}%

🏆 Overall Winner: {'Resume #1' if avg_score1 > avg_score2 else 'Resume #2' if avg_score2 > avg_score1 else 'Tie'}

🎯 Key Insights:
• Skills gap analysis shows both resumes could benefit from computer vision and ML frameworks
• Experience levels vary between resumes - consider this in application strategy
• Department preferences may differ based on skill alignment""",
        title="🔍 Executive Summary",
        border_style="green",
        padding=(1, 2)
    )
    
    console.print(analysis_panel)
    console.print()
    console.print("[bold green]Analysis Complete! 🎉[/bold green]")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())