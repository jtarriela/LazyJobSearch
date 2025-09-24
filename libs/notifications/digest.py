"""Daily digest generation and email delivery

Handles creation of daily digest emails with recent matches, new jobs,
and pending reviews. Supports templating and user preferences.
"""
from __future__ import annotations
import logging
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)

class DigestType(Enum):
    """Types of digest content"""
    DAILY = "daily"
    WEEKLY = "weekly"
    NEW_MATCHES = "new_matches"
    REVIEW_REMINDERS = "review_reminders"

@dataclass
class DigestMatch:
    """A match included in digest"""
    job_title: str
    company_name: str
    llm_score: int
    action: str
    reasoning: str
    job_url: Optional[str] = None
    match_date: Optional[datetime] = None

@dataclass
class DigestStats:
    """Statistics for digest period"""
    new_matches: int
    high_priority_matches: int
    jobs_crawled: int
    applications_submitted: int
    reviews_pending: int

@dataclass
class DigestContent:
    """Content for a digest email"""
    user_id: str
    digest_type: DigestType
    period_start: datetime
    period_end: datetime
    stats: DigestStats
    top_matches: List[DigestMatch]
    new_jobs: List[Dict[str, Any]]
    pending_reviews: List[Dict[str, Any]]
    generated_at: datetime = None
    
    def __post_init__(self):
        if self.generated_at is None:
            self.generated_at = datetime.now()

class DigestGenerator:
    """Generates digest content from database data"""
    
    def __init__(self, session):
        self.session = session
    
    def generate_daily_digest(self, user_id: str, date: Optional[datetime] = None) -> DigestContent:
        """Generate daily digest for a user
        
        Args:
            user_id: User to generate digest for
            date: Date to generate digest for (defaults to yesterday)
            
        Returns:
            DigestContent with all digest information
        """
        if date is None:
            date = datetime.now().date()
        
        period_start = datetime.combine(date - timedelta(days=1), datetime.min.time())
        period_end = datetime.combine(date, datetime.min.time())
        
        # Gather digest data
        stats = self._get_digest_stats(user_id, period_start, period_end)
        top_matches = self._get_top_matches(user_id, period_start, period_end)
        new_jobs = self._get_new_jobs(user_id, period_start, period_end)
        pending_reviews = self._get_pending_reviews(user_id)
        
        return DigestContent(
            user_id=user_id,
            digest_type=DigestType.DAILY,
            period_start=period_start,
            period_end=period_end,
            stats=stats,
            top_matches=top_matches,
            new_jobs=new_jobs,
            pending_reviews=pending_reviews
        )
    
    def _get_digest_stats(self, user_id: str, start: datetime, end: datetime) -> DigestStats:
        """Get statistics for digest period"""
        try:
            # Mock implementation - would query actual database
            return DigestStats(
                new_matches=12,
                high_priority_matches=3,
                jobs_crawled=45,
                applications_submitted=2,
                reviews_pending=1
            )
        except Exception as e:
            logger.error(f"Failed to get digest stats: {e}")
            return DigestStats(0, 0, 0, 0, 0)
    
    def _get_top_matches(self, user_id: str, start: datetime, end: datetime, limit: int = 5) -> List[DigestMatch]:
        """Get top matches for digest period"""
        try:
            # Mock implementation - would query matches table
            mock_matches = [
                DigestMatch(
                    job_title="Senior Python Developer",
                    company_name="TechCorp",
                    llm_score=92,
                    action="APPLY_HIGH",
                    reasoning="Excellent skills alignment with Python, Django, and AWS experience.",
                    job_url="https://techcorp.com/jobs/senior-python-dev",
                    match_date=datetime.now() - timedelta(hours=2)
                ),
                DigestMatch(
                    job_title="Full Stack Engineer",
                    company_name="StartupInc",
                    llm_score=87,
                    action="APPLY_HIGH",
                    reasoning="Strong match for web development skills and startup experience.",
                    job_url="https://startupinc.com/careers/fullstack",
                    match_date=datetime.now() - timedelta(hours=5)
                ),
                DigestMatch(
                    job_title="Data Engineer",
                    company_name="DataCorp",
                    llm_score=78,
                    action="APPLY_MEDIUM",
                    reasoning="Good technical fit, but slightly different domain experience.",
                    job_url="https://datacorp.com/jobs/data-eng",
                    match_date=datetime.now() - timedelta(hours=8)
                )
            ]
            
            return mock_matches[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get top matches: {e}")
            return []
    
    def _get_new_jobs(self, user_id: str, start: datetime, end: datetime, limit: int = 10) -> List[Dict[str, Any]]:
        """Get new jobs for digest period"""
        try:
            # Mock implementation - would query jobs table
            return [
                {
                    "title": "Python Developer",
                    "company": "NewTech",
                    "location": "Remote",
                    "posted_date": (datetime.now() - timedelta(hours=3)).isoformat()
                },
                {
                    "title": "Software Engineer",
                    "company": "InnovateCorp", 
                    "location": "San Francisco",
                    "posted_date": (datetime.now() - timedelta(hours=6)).isoformat()
                }
            ]
        except Exception as e:
            logger.error(f"Failed to get new jobs: {e}")
            return []
    
    def _get_pending_reviews(self, user_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get pending reviews for user"""
        try:
            # Mock implementation - would query reviews table
            return [
                {
                    "job_title": "Senior Backend Engineer",
                    "company": "TechGiant",
                    "review_requested": (datetime.now() - timedelta(days=1)).isoformat(),
                    "current_score": 75
                }
            ]
        except Exception as e:
            logger.error(f"Failed to get pending reviews: {e}")
            return []

class EmailTemplate:
    """HTML email template for digests"""
    
    def __init__(self):
        self.base_template = self._load_base_template()
    
    def render_digest(self, content: DigestContent) -> str:
        """Render digest content to HTML email"""
        try:
            # Generate sections
            stats_html = self._render_stats(content.stats)
            matches_html = self._render_matches(content.top_matches)
            jobs_html = self._render_new_jobs(content.new_jobs)
            reviews_html = self._render_pending_reviews(content.pending_reviews)
            
            # Substitute into template
            html = self.base_template.format(
                user_id=content.user_id,
                period_date=content.period_start.strftime("%B %d, %Y"),
                stats_section=stats_html,
                matches_section=matches_html,
                jobs_section=jobs_html,
                reviews_section=reviews_html,
                generated_time=content.generated_at.strftime("%Y-%m-%d %H:%M")
            )
            
            return html
            
        except Exception as e:
            logger.error(f"Failed to render digest: {e}")
            return self._render_error_template(str(e))
    
    def _load_base_template(self) -> str:
        """Load base HTML template"""
        return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>LazyJobSearch Daily Digest</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; text-align: center; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .stats {{ display: flex; justify-content: space-around; text-align: center; }}
        .stat {{ flex: 1; }}
        .stat-number {{ font-size: 24px; font-weight: bold; color: #3498db; }}
        .match {{ border-left: 4px solid #27ae60; padding: 10px; margin: 10px 0; }}
        .match-high {{ border-left-color: #e74c3c; }}
        .match-medium {{ border-left-color: #f39c12; }}
        .job-item {{ padding: 8px; border-bottom: 1px solid #eee; }}
        .footer {{ text-align: center; color: #777; font-size: 12px; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>LazyJobSearch Daily Digest</h1>
            <p>Your personalized job search update for {period_date}</p>
        </div>
        
        <div class="section">
            <h2>📊 Daily Statistics</h2>
            {stats_section}
        </div>
        
        <div class="section">
            <h2>🎯 Top Matches</h2>
            {matches_section}
        </div>
        
        <div class="section">
            <h2>🆕 New Jobs</h2>
            {jobs_section}
        </div>
        
        <div class="section">
            <h2>📝 Pending Reviews</h2>
            {reviews_section}
        </div>
        
        <div class="footer">
            <p>Generated on {generated_time} | LazyJobSearch</p>
            <p><a href="#">Unsubscribe</a> | <a href="#">Update Preferences</a></p>
        </div>
    </div>
</body>
</html>
"""
    
    def _render_stats(self, stats: DigestStats) -> str:
        """Render statistics section"""
        return f"""
        <div class="stats">
            <div class="stat">
                <div class="stat-number">{stats.new_matches}</div>
                <div>New Matches</div>
            </div>
            <div class="stat">
                <div class="stat-number">{stats.high_priority_matches}</div>
                <div>High Priority</div>
            </div>
            <div class="stat">
                <div class="stat-number">{stats.jobs_crawled}</div>
                <div>Jobs Crawled</div>
            </div>
            <div class="stat">
                <div class="stat-number">{stats.applications_submitted}</div>
                <div>Applications</div>
            </div>
        </div>
        """
    
    def _render_matches(self, matches: List[DigestMatch]) -> str:
        """Render matches section"""
        if not matches:
            return "<p>No new matches found.</p>"
        
        html = ""
        for match in matches:
            css_class = "match"
            if match.action == "APPLY_HIGH":
                css_class += " match-high"
            elif match.action == "APPLY_MEDIUM":
                css_class += " match-medium"
            
            html += f"""
            <div class="{css_class}">
                <h3>{match.job_title} at {match.company_name}</h3>
                <p><strong>Score:</strong> {match.llm_score}/100 | <strong>Action:</strong> {match.action}</p>
                <p>{match.reasoning}</p>
                {f'<p><a href="{match.job_url}">View Job</a></p>' if match.job_url else ''}
            </div>
            """
        
        return html
    
    def _render_new_jobs(self, jobs: List[Dict[str, Any]]) -> str:
        """Render new jobs section"""
        if not jobs:
            return "<p>No new jobs found.</p>"
        
        html = ""
        for job in jobs:
            html += f"""
            <div class="job-item">
                <strong>{job['title']}</strong> at {job['company']}
                <span style="float: right; color: #777;">{job['location']}</span>
            </div>
            """
        
        return html
    
    def _render_pending_reviews(self, reviews: List[Dict[str, Any]]) -> str:
        """Render pending reviews section"""
        if not reviews:
            return "<p>No pending reviews.</p>"
        
        html = ""
        for review in reviews:
            html += f"""
            <div class="job-item">
                <strong>{review['job_title']}</strong> at {review['company']}
                <span style="float: right;">Score: {review['current_score']}/100</span>
                <br><small>Review requested: {review['review_requested']}</small>
            </div>
            """
        
        return html
    
    def _render_error_template(self, error: str) -> str:
        """Render error template"""
        return f"""
        <html>
        <body>
            <h1>Digest Generation Error</h1>
            <p>Sorry, there was an error generating your digest: {error}</p>
        </body>
        </html>
        """

class EmailSender:
    """Handles email delivery for digests"""
    
    def __init__(self, smtp_host: str = "localhost", smtp_port: int = 587):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.sent_count = 0
    
    async def send_digest_email(
        self,
        recipient_email: str,
        subject: str,
        html_content: str,
        from_email: str = "noreply@lazyjobsearch.com"
    ) -> bool:
        """Send digest email to recipient
        
        Args:
            recipient_email: Recipient email address
            subject: Email subject line
            html_content: HTML email content
            from_email: Sender email address
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Mock implementation - would use actual SMTP or email service
            logger.info(f"Sending digest email to {recipient_email}")
            logger.info(f"Subject: {subject}")
            logger.info(f"Content length: {len(html_content)} characters")
            
            # Simulate email sending
            import asyncio
            await asyncio.sleep(0.1)  # Simulate network delay
            
            self.sent_count += 1
            logger.info(f"Digest email sent successfully to {recipient_email}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send digest email to {recipient_email}: {e}")
            return False

class DigestService:
    """Main service for generating and sending digests"""
    
    def __init__(self, session, smtp_config: Optional[Dict[str, Any]] = None):
        self.session = session
        self.generator = DigestGenerator(session)
        self.template = EmailTemplate()
        
        smtp_config = smtp_config or {}
        self.email_sender = EmailSender(
            smtp_host=smtp_config.get('host', 'localhost'),
            smtp_port=smtp_config.get('port', 587)
        )
    
    async def send_daily_digest(self, user_id: str, user_email: str) -> bool:
        """Generate and send daily digest for a user
        
        Args:
            user_id: User ID to generate digest for
            user_email: User's email address
            
        Returns:
            True if digest was sent successfully
        """
        try:
            # Generate digest content
            content = self.generator.generate_daily_digest(user_id)
            
            # Skip if no meaningful content
            if (content.stats.new_matches == 0 and 
                len(content.top_matches) == 0 and 
                len(content.new_jobs) == 0):
                logger.info(f"Skipping digest for {user_id} - no new content")
                return True
            
            # Render email
            html_content = self.template.render_digest(content)
            
            # Generate subject
            subject = self._generate_subject(content)
            
            # Send email
            success = await self.email_sender.send_digest_email(
                recipient_email=user_email,
                subject=subject,
                html_content=html_content
            )
            
            if success:
                # Log digest sent (would update database in real implementation)
                logger.info(f"Daily digest sent to {user_email}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send daily digest for {user_id}: {e}")
            return False
    
    async def send_digest_batch(self, user_list: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Send digests to multiple users
        
        Args:
            user_list: List of (user_id, email) tuples
            
        Returns:
            Dictionary with send statistics
        """
        results = {
            'total_users': len(user_list),
            'successful_sends': 0,
            'failed_sends': 0,
            'skipped_sends': 0,
            'errors': []
        }
        
        for user_id, user_email in user_list:
            try:
                success = await self.send_daily_digest(user_id, user_email)
                if success:
                    results['successful_sends'] += 1
                else:
                    results['failed_sends'] += 1
            except Exception as e:
                results['failed_sends'] += 1
                results['errors'].append(f"{user_id}: {str(e)}")
        
        logger.info(f"Digest batch complete: {results['successful_sends']}/{results['total_users']} sent")
        return results
    
    def _generate_subject(self, content: DigestContent) -> str:
        """Generate email subject line"""
        if content.stats.high_priority_matches > 0:
            return f"🎯 {content.stats.high_priority_matches} high-priority matches found!"
        elif content.stats.new_matches > 0:
            return f"📧 {content.stats.new_matches} new job matches for you"
        else:
            return "📋 Your daily job search update"

def create_digest_service(session, smtp_config: Optional[Dict[str, Any]] = None) -> DigestService:
    """Factory function to create digest service"""
    return DigestService(session, smtp_config)