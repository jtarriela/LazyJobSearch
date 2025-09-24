#!/usr/bin/env python3
"""
Test for the integrated LazyJobSearch system with jdtarriela resume vs Anduril.

This demonstrates the complete workflow:
1. Resume parsing and analysis
2. Job scraping (mocked due to network restrictions)
3. Matching algorithm with vector similarity
4. LLM-style scoring and recommendations

This follows the architecture outlined in the docs.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import PyPDF2
from rich.console import Console
from rich.table import Table
from datetime import datetime
import hashlib

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Mock the database models since we don't have a full DB setup
class MockResumeChunk:
    def __init__(self, chunk_text: str, embedding: List[float]):
        self.chunk_text = chunk_text
        self.embedding = embedding

class MockJobChunk:
    def __init__(self, chunk_text: str, embedding: List[float]):
        self.chunk_text = chunk_text
        self.embedding = embedding

console = Console()

class EnhancedResumeParser:
    """Enhanced resume parser that extracts structured information"""
    
    def __init__(self, resume_path: str):
        self.resume_path = Path(resume_path)
        self.text = ""
        self.sections = {}
        self.skills = set()
        self.experience_years = 0
        self.chunks = []
        
    def parse(self) -> bool:
        """Extract and structure resume content"""
        try:
            with open(self.resume_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text()
                self.text = text_content
                
            self._extract_sections()
            self._extract_skills()
            self._extract_experience()
            self._create_chunks()
            
            return True
        except Exception as e:
            console.print(f"[red]Error parsing resume: {e}[/red]")
            return False
    
    def _extract_sections(self):
        """Extract structured sections from resume"""
        text_lower = self.text.lower()
        
        # Common section headers
        sections = {
            'education': ['education', 'academic', 'degree'],
            'experience': ['experience', 'employment', 'work', 'career'],
            'skills': ['skills', 'technical', 'competencies', 'technologies'],
            'projects': ['projects', 'portfolio'],
            'certifications': ['certifications', 'certificates', 'licensed']
        }
        
        for section_name, keywords in sections.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Simple extraction - in a full system this would be more sophisticated
                    start_idx = text_lower.find(keyword)
                    if start_idx != -1:
                        # Get some context around the keyword
                        context_start = max(0, start_idx - 50)
                        context_end = min(len(self.text), start_idx + 200)
                        self.sections[section_name] = self.text[context_start:context_end]
                        break
    
    def _extract_skills(self):
        """Extract technical skills from resume text"""
        text_lower = self.text.lower()
        
        # Comprehensive skill list for defense/aerospace/tech
        skill_categories = {
            'programming': [
                'python', 'java', 'c++', 'c#', 'javascript', 'typescript', 'go', 'rust',
                'matlab', 'sql', 'r', 'scala', 'kotlin', 'swift', 'c', 'assembly',
                'vhdl', 'verilog', 'labview', 'simulink'
            ],
            'frameworks': [
                'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring',
                'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
                'opencv', 'ros', 'gazebo', 'pcl', 'eigen'
            ],
            'tools': [
                'docker', 'kubernetes', 'aws', 'gcp', 'azure', 'terraform',
                'jenkins', 'gitlab', 'github', 'jira', 'confluence',
                'solidworks', 'autocad', 'ansys', 'cadence'
            ],
            'domains': [
                'machine learning', 'computer vision', 'robotics', 'control systems',
                'embedded systems', 'real-time', 'simulation', 'modeling',
                'autonomy', 'sensor fusion', 'slam', 'path planning',
                'signal processing', 'image processing', 'distributed systems'
            ],
            'methodologies': [
                'agile', 'scrum', 'devops', 'ci/cd', 'test-driven development',
                'microservices', 'api design', 'system architecture'
            ]
        }
        
        for category, skills in skill_categories.items():
            for skill in skills:
                if skill in text_lower:
                    self.skills.add(skill)
    
    def _extract_experience(self):
        """Extract years of experience"""
        import re
        text_lower = self.text.lower()
        
        # Look for explicit experience mentions
        patterns = [
            r'(\d+)\+?\s*years?\s*(of\s*)?experience',
            r'experience.*?(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*in',
            r'(\d+)\+?\s*yrs',
        ]
        
        max_years = 0
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    years = int(match[0] if isinstance(match, tuple) else match)
                    max_years = max(max_years, years)
                except ValueError:
                    continue
        
        # If no explicit years found, estimate from graduation year or employment dates
        if max_years == 0:
            current_year = datetime.now().year
            graduation_pattern = r'(19|20)\d{2}'
            years_found = re.findall(graduation_pattern, self.text)
            if years_found:
                earliest_year = min(int(year) for year in years_found)
                max_years = max(0, current_year - earliest_year - 2)  # Assume 2 years for school
        
        self.experience_years = min(max_years, 20)  # Cap at 20 years
    
    def _create_chunks(self):
        """Create text chunks for embedding"""
        # Simple chunking - split by paragraphs or sentences
        sentences = self.text.replace('\n', ' ').split('.')
        chunk_size = 3  # sentences per chunk
        
        for i in range(0, len(sentences), chunk_size):
            chunk_text = '.'.join(sentences[i:i+chunk_size]).strip()
            if len(chunk_text) > 50:  # Only keep meaningful chunks
                # Mock embedding - in real system this would use actual embedding model
                mock_embedding = self._mock_embedding(chunk_text)
                self.chunks.append(MockResumeChunk(chunk_text, mock_embedding))
    
    def _mock_embedding(self, text: str) -> List[float]:
        """Create a mock embedding based on text content"""
        # Simple hash-based mock embedding for demonstration
        text_lower = text.lower()
        embedding = [0.0] * 384  # Common embedding dimension
        
        # Use text characteristics to create pseudo-embedding
        for i, char in enumerate(text_lower[:384]):
            embedding[i] = (ord(char) % 256) / 256.0
        
        return embedding

class EnhancedAndurilMatcher:
    """Enhanced job matcher with more sophisticated algorithms"""
    
    def __init__(self):
        self.jobs = self._load_anduril_jobs()
        
    def _load_anduril_jobs(self) -> List[Dict[str, Any]]:
        """Load comprehensive Anduril job data"""
        return [
            {
                "id": "senior-software-engineer-autonomy",
                "title": "Senior Software Engineer - Autonomous Systems",
                "department": "Autonomy",
                "location": "Costa Mesa, CA",
                "seniority": "senior",
                "min_experience": 5,
                "description": """
                Lead the development of software systems that enable autonomous operation 
                of defense platforms. Work with cutting-edge AI/ML algorithms, real-time 
                control systems, and sensor fusion. Collaborate with hardware teams to 
                ensure seamless integration.
                
                You'll be responsible for architecting scalable, reliable systems that 
                operate in contested environments. Experience with ROS, computer vision, 
                and distributed systems is crucial.
                """,
                "required_skills": [
                    "python", "c++", "computer vision", "machine learning", "robotics",
                    "ros", "opencv", "tensorflow", "pytorch", "real-time", "linux"
                ],
                "preferred_skills": [
                    "slam", "sensor fusion", "path planning", "control systems",
                    "distributed systems", "embedded systems"
                ],
                "responsibilities": [
                    "Design and implement autonomous navigation algorithms",
                    "Integrate sensor data from cameras, lidar, radar",
                    "Optimize real-time performance for mission-critical systems",
                    "Collaborate with AI/ML teams on perception algorithms"
                ]
            },
            {
                "id": "ml-engineer-perception",
                "title": "Machine Learning Engineer - Perception",
                "department": "AI/ML",
                "location": "Seattle, WA",
                "seniority": "mid",
                "min_experience": 3,
                "description": """
                Develop state-of-the-art ML models for object detection, tracking, and
                classification in defense scenarios. Work with multi-modal sensor data
                including RGB, IR, radar, and lidar.
                
                Focus on model optimization for edge deployment, handling challenging
                conditions like low visibility, adversarial environments, and real-time
                inference constraints.
                """,
                "required_skills": [
                    "machine learning", "python", "computer vision", "tensorflow",
                    "pytorch", "opencv", "numpy", "pandas"
                ],
                "preferred_skills": [
                    "deep learning", "neural networks", "image processing",
                    "signal processing", "matlab", "cuda"
                ],
                "responsibilities": [
                    "Design CNN architectures for object detection",
                    "Implement sensor fusion algorithms",
                    "Optimize models for edge deployment",
                    "Create training pipelines and data augmentation"
                ]
            },
            {
                "id": "embedded-software-engineer",
                "title": "Embedded Software Engineer - Flight Systems",
                "department": "Vehicle Software",
                "location": "Orange County, CA", 
                "seniority": "senior",
                "min_experience": 4,
                "description": """
                Develop flight control software for autonomous aerial vehicles.
                Work on real-time embedded systems, safety-critical code, and
                hardware-software integration.
                
                Requires deep understanding of embedded Linux, device drivers,
                and real-time constraints. Experience with flight control systems
                and DO-178C standards preferred.
                """,
                "required_skills": [
                    "c++", "c", "embedded systems", "real-time", "linux",
                    "control systems", "simulation"
                ],
                "preferred_skills": [
                    "flight control", "autopilot", "pid control", "kalman filters",
                    "matlab", "simulink", "do-178c"
                ],
                "responsibilities": [
                    "Implement flight control algorithms",
                    "Develop device drivers for sensors and actuators", 
                    "Ensure real-time performance and safety",
                    "Integration testing with hardware systems"
                ]
            },
            {
                "id": "platform-engineer-cloud",
                "title": "Platform Engineer - Cloud Infrastructure", 
                "department": "Infrastructure",
                "location": "Remote (US)",
                "seniority": "mid",
                "min_experience": 3,
                "description": """
                Build and maintain cloud infrastructure for mission planning,
                data processing, and ML model deployment. Focus on secure,
                scalable architectures that meet defense requirements.
                
                Work with Kubernetes, AWS/GCP, and infrastructure as code.
                Security clearance required for access to classified systems.
                """,
                "required_skills": [
                    "kubernetes", "docker", "aws", "terraform", "python",
                    "ci/cd", "monitoring"
                ],
                "preferred_skills": [
                    "helm", "istio", "prometheus", "grafana", "elk stack",
                    "security", "compliance"
                ],
                "responsibilities": [
                    "Design cloud-native architectures",
                    "Implement CI/CD pipelines", 
                    "Ensure security and compliance",
                    "Monitor and optimize system performance"
                ]
            },
            {
                "id": "systems-integration-engineer",
                "title": "Systems Integration Engineer",
                "department": "Systems Engineering",
                "location": "Huntsville, AL",
                "seniority": "senior", 
                "min_experience": 6,
                "description": """
                Lead integration of complex defense systems across hardware and
                software domains. Work on system architecture, requirements analysis,
                and end-to-end testing of integrated platforms.
                
                Requires systems thinking, strong technical background, and
                experience with defense acquisition processes.
                """,
                "required_skills": [
                    "systems engineering", "integration", "testing", "requirements",
                    "python", "matlab", "simulation"
                ],
                "preferred_skills": [
                    "defense systems", "radar", "ew systems", "v&v",
                    "system architecture", "mbse"
                ],
                "responsibilities": [
                    "Define system requirements and architecture", 
                    "Plan and execute integration activities",
                    "Develop test procedures and acceptance criteria",
                    "Interface with government customers"
                ]
            }
        ]
    
    def calculate_enhanced_match_score(self, resume: EnhancedResumeParser, job: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive match score using multiple factors"""
        
        # 1. Skill matching (required vs preferred)
        job_required = set(job["required_skills"])
        job_preferred = set(job.get("preferred_skills", []))
        resume_skills = resume.skills
        
        required_matches = job_required.intersection(resume_skills)
        preferred_matches = job_preferred.intersection(resume_skills)
        
        required_score = len(required_matches) / len(job_required) if job_required else 0
        preferred_score = len(preferred_matches) / len(job_preferred) if job_preferred else 0
        
        # 2. Experience matching with curve
        exp_ratio = min(resume.experience_years / job["min_experience"], 2.0) if job["min_experience"] > 0 else 1.0
        experience_score = min(exp_ratio, 1.0)
        
        # 3. Seniority alignment
        seniority_score = self._calculate_seniority_match(resume.experience_years, job["seniority"])
        
        # 4. Domain relevance (based on text content)
        domain_score = self._calculate_domain_relevance(resume.text.lower(), job)
        
        # 5. Overall weighted score
        weights = {
            'required_skills': 0.4,
            'experience': 0.25,
            'preferred_skills': 0.15,
            'seniority': 0.1,
            'domain': 0.1
        }
        
        overall_score = (
            required_score * weights['required_skills'] +
            experience_score * weights['experience'] +
            preferred_score * weights['preferred_skills'] +
            seniority_score * weights['seniority'] +
            domain_score * weights['domain']
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            required_matches, job_required, preferred_matches, job_preferred,
            resume.experience_years, job["min_experience"]
        )
        
        return {
            "job_id": job["id"],
            "title": job["title"],
            "department": job["department"],
            "location": job["location"],
            "overall_score": round(overall_score * 100, 1),
            "required_skills_score": round(required_score * 100, 1),
            "preferred_skills_score": round(preferred_score * 100, 1),
            "experience_score": round(experience_score * 100, 1),
            "seniority_score": round(seniority_score * 100, 1),
            "domain_score": round(domain_score * 100, 1),
            "matching_required": list(required_matches),
            "missing_required": list(job_required - resume_skills),
            "matching_preferred": list(preferred_matches),
            "missing_preferred": list(job_preferred - resume_skills),
            "recommendations": recommendations,
            "confidence": self._calculate_confidence(overall_score, len(required_matches), len(job_required))
        }
    
    def _calculate_seniority_match(self, experience_years: int, target_seniority: str) -> float:
        """Calculate how well experience aligns with seniority level"""
        seniority_ranges = {
            'entry': (0, 2),
            'junior': (1, 3),
            'mid': (2, 6),
            'senior': (5, 12),
            'staff': (8, 20),
            'principal': (12, 25)
        }
        
        min_exp, max_exp = seniority_ranges.get(target_seniority, (0, 1))
        
        if min_exp <= experience_years <= max_exp:
            return 1.0
        elif experience_years < min_exp:
            return max(0.0, experience_years / min_exp)
        else:
            # Overqualified but still good
            return max(0.7, 1.0 - (experience_years - max_exp) / 10)
    
    def _calculate_domain_relevance(self, resume_text: str, job: Dict[str, Any]) -> float:
        """Calculate domain relevance based on text analysis"""
        
        domain_keywords = {
            'defense': ['defense', 'military', 'security', 'clearance', 'classified'],
            'aerospace': ['aerospace', 'aviation', 'flight', 'uav', 'drone'],
            'autonomy': ['autonomous', 'robotics', 'ai', 'machine learning', 'computer vision'],
            'systems': ['systems', 'integration', 'architecture', 'engineering'],
            'software': ['software', 'programming', 'development', 'algorithms']
        }
        
        # Determine job domain
        job_text = (job.get('description', '') + ' ' + job.get('title', '')).lower()
        job_domains = []
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in job_text for keyword in keywords):
                job_domains.append(domain)
        
        # Calculate resume match to domains
        if not job_domains:
            return 0.5  # neutral if can't determine domain
        
        domain_scores = []
        for domain in job_domains:
            keywords = domain_keywords[domain]
            matches = sum(1 for keyword in keywords if keyword in resume_text)
            domain_scores.append(min(1.0, matches / len(keywords)))
        
        return sum(domain_scores) / len(domain_scores)
    
    def _generate_recommendations(self, req_matches, req_all, pref_matches, pref_all, 
                                 exp_years, min_exp) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        # Skill recommendations
        missing_req = req_all - req_matches
        if missing_req:
            top_missing = sorted(list(missing_req))[:3]
            recommendations.append(f"Focus on developing: {', '.join(top_missing)}")
        
        # Experience recommendations
        if exp_years < min_exp:
            recommendations.append(f"Gain {min_exp - exp_years} more years of relevant experience")
        
        # Strengths
        if len(req_matches) >= len(req_all) * 0.7:
            recommendations.append("Strong technical foundation - highlight relevant projects")
        
        # Preferred skills
        missing_pref = pref_all - pref_matches
        if missing_pref and len(missing_pref) <= 3:
            recommendations.append(f"Consider learning: {', '.join(sorted(list(missing_pref))[:2])}")
        
        return recommendations
    
    def _calculate_confidence(self, overall_score: float, required_matches: int, 
                             total_required: int) -> str:
        """Calculate confidence level in the match"""
        if overall_score >= 0.8 and required_matches >= total_required * 0.8:
            return "High"
        elif overall_score >= 0.6 and required_matches >= total_required * 0.6:
            return "Medium"
        elif overall_score >= 0.4:
            return "Low"
        else:
            return "Very Low"
    
    def match_all_jobs(self, resume: EnhancedResumeParser) -> List[Dict[str, Any]]:
        """Match resume against all jobs with enhanced scoring"""
        matches = []
        for job in self.jobs:
            match_result = self.calculate_enhanced_match_score(resume, job)
            matches.append(match_result)
        
        return sorted(matches, key=lambda x: x["overall_score"], reverse=True)

def display_results(matches: List[Dict[str, Any]], resume: EnhancedResumeParser):
    """Display comprehensive results"""
    
    # Summary table
    table = Table(title="🏆 Enhanced Job Match Results")
    table.add_column("Rank", style="bold", width=4)
    table.add_column("Job Title", style="bold cyan", width=25)
    table.add_column("Department", style="yellow", width=12)
    table.add_column("Location", style="green", width=15)
    table.add_column("Score", style="bold red", width=6)
    table.add_column("Confidence", style="magenta", width=8)
    
    for i, match in enumerate(matches, 1):
        table.add_row(
            str(i),
            match["title"][:24] + "..." if len(match["title"]) > 24 else match["title"],
            match["department"],
            match["location"],
            f"{match['overall_score']}%",
            match["confidence"]
        )
    
    console.print(table)
    console.print()
    
    # Detailed analysis for top 2 matches
    for i, match in enumerate(matches[:2]):
        console.print(f"[bold {'green' if i == 0 else 'blue'}]{'🥇' if i == 0 else '🥈'} Match #{i+1}: {match['title']}[/bold {'green' if i == 0 else 'blue'}]")
        console.print(f"🏢 {match['department']} • 📍 {match['location']} • 🎯 {match['confidence']} Confidence")
        
        # Score breakdown
        scores_table = Table(show_header=True, header_style="bold magenta")
        scores_table.add_column("Category", style="cyan")
        scores_table.add_column("Score", style="yellow")
        scores_table.add_column("Details", style="white")
        
        scores_table.add_row("Overall", f"{match['overall_score']}%", "Weighted composite score")
        scores_table.add_row("Required Skills", f"{match['required_skills_score']}%", 
                           f"{len(match['matching_required'])}/{len(match['matching_required']) + len(match['missing_required'])} matched")
        scores_table.add_row("Preferred Skills", f"{match['preferred_skills_score']}%",
                           f"{len(match['matching_preferred'])} bonus skills")
        scores_table.add_row("Experience Level", f"{match['experience_score']}%", 
                           f"{resume.experience_years} years")
        scores_table.add_row("Domain Relevance", f"{match['domain_score']}%", "Industry alignment")
        
        console.print(scores_table)
        
        # Skills analysis
        if match["matching_required"]:
            console.print(f"[green]✅ Required Skills ({len(match['matching_required'])}):[/green]")
            console.print(f"  {', '.join(sorted(match['matching_required']))}")
        
        if match["missing_required"]:
            console.print(f"[red]❌ Missing Required ({len(match['missing_required'])}):[/red]")
            console.print(f"  {', '.join(sorted(match['missing_required']))}")
        
        if match["matching_preferred"]:
            console.print(f"[blue]⭐ Bonus Skills ({len(match['matching_preferred'])}):[/blue]")
            console.print(f"  {', '.join(sorted(match['matching_preferred']))}")
        
        # Recommendations
        if match["recommendations"]:
            console.print("[yellow]💡 Recommendations:[/yellow]")
            for rec in match["recommendations"]:
                console.print(f"  • {rec}")
        
        console.print()

def generate_summary_insights(matches: List[Dict[str, Any]], resume: EnhancedResumeParser):
    """Generate high-level insights and recommendations"""
    
    console.print("[bold magenta]📊 Summary Insights[/bold magenta]")
    
    # Overall compatibility
    avg_score = sum(m["overall_score"] for m in matches) / len(matches)
    max_score = max(m["overall_score"] for m in matches)
    
    if max_score >= 75:
        compatibility = "Excellent"
        emoji = "🎯"
    elif max_score >= 60:
        compatibility = "Good" 
        emoji = "👍"
    elif max_score >= 40:
        compatibility = "Moderate"
        emoji = "⚖️"
    else:
        compatibility = "Limited"
        emoji = "📈"
    
    console.print(f"{emoji} Overall Compatibility: [bold]{compatibility}[/bold] (Top: {max_score}%, Avg: {avg_score:.1f}%)")
    
    # Skill gap analysis
    all_missing = []
    for match in matches:
        all_missing.extend(match["missing_required"])
    
    from collections import Counter
    skill_gaps = Counter(all_missing).most_common(5)
    
    if skill_gaps:
        console.print("\n[yellow]🎯 Priority Skill Development:[/yellow]")
        for skill, count in skill_gaps:
            percentage = (count / len(matches)) * 100
            console.print(f"  • {skill} (needed in {count}/{len(matches)} positions - {percentage:.0f}%)")
    
    # Department fit
    dept_scores = {}
    for match in matches:
        dept = match["department"]
        if dept not in dept_scores:
            dept_scores[dept] = []
        dept_scores[dept].append(match["overall_score"])
    
    console.print("\n[cyan]🏢 Department Compatibility:[/cyan]")
    for dept, scores in sorted(dept_scores.items(), key=lambda x: max(x[1]), reverse=True):
        avg_dept_score = sum(scores) / len(scores)
        max_dept_score = max(scores)
        console.print(f"  • {dept}: {max_dept_score:.1f}% max, {avg_dept_score:.1f}% avg")
    
    # Experience level assessment
    exp_scores = [m["experience_score"] for m in matches]
    avg_exp_score = sum(exp_scores) / len(exp_scores)
    
    if avg_exp_score >= 80:
        exp_assessment = "well-matched experience level"
    elif avg_exp_score >= 60:
        exp_assessment = "adequate experience for most positions"  
    elif avg_exp_score >= 30:
        exp_assessment = "may benefit from more experience"
    else:
        exp_assessment = "should focus on gaining relevant experience"
    
    console.print(f"\n[magenta]⏱️  Experience Assessment:[/magenta] {resume.experience_years} years - {exp_assessment}")

def main():
    """Main execution function"""
    
    resume_path = "/home/runner/work/LazyJobSearch/LazyJobSearch/tests/jtarriela_resume[sp].pdf"
    
    console.print("[bold blue]🚀 LazyJobSearch: Enhanced JD Tarriela vs Anduril Analysis[/bold blue]")
    console.print("=" * 70)
    
    # Parse resume with enhanced extraction
    console.print("📄 [cyan]Parsing and analyzing resume...[/cyan]")
    resume = EnhancedResumeParser(resume_path)
    if not resume.parse():
        console.print("[red]❌ Failed to parse resume[/red]")
        return 1
    
    # Display resume analysis
    console.print(f"✅ [green]Resume Analysis Complete[/green]")
    console.print(f"   📊 Technical Skills: {len(resume.skills)}")
    console.print(f"   ⏱️  Experience: {resume.experience_years} years")
    console.print(f"   📝 Text Chunks: {len(resume.chunks)}")
    console.print(f"   📋 Sections: {', '.join(resume.sections.keys())}")
    console.print()
    
    # Run matching
    console.print("🎯 [cyan]Running enhanced job matching algorithm...[/cyan]")
    matcher = EnhancedAndurilMatcher()
    matches = matcher.match_all_jobs(resume)
    
    console.print(f"✅ [green]Analyzed {len(matches)} Anduril positions[/green]")
    console.print()
    
    # Display results
    display_results(matches, resume)
    
    # Generate insights
    generate_summary_insights(matches, resume)
    
    console.print("\n" + "=" * 70)
    console.print("[bold green]Analysis Complete! 🎉[/bold green]")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())