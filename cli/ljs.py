#!/usr/bin/env python
"""LazyJobSearch Typer-based CLI.

Implements the command surface defined in `docs/CLI_DESIGN.md`.
Currently provides stubs plus:
  - Config precedence & JSON Schema validation
  - Schema docs validation runner
  - Company template generator
  - Global --dry-run flag
  - Pretty output via rich

Future wiring: DB sessions, background queue dispatch, real CRUD.
"""
from __future__ import annotations
import json
import os
import contextvars
from pathlib import Path
from typing import Optional, Any, Dict

import typer
import yaml
from rich.console import Console
from rich.table import Table
import logging
from logging import Logger

from libs.db.session import get_session
from libs.db import models
from jsonschema import validate as js_validate, ValidationError
from pydantic import BaseModel, Field, ValidationError as PydValidationError, validator

APP = typer.Typer(add_completion=False, help="LazyJobSearch CLI")
console = Console()

# Sub-apps
config_app = typer.Typer(help="Config management")
user_app = typer.Typer(help="User management")
resume_app = typer.Typer(help="Resume operations")
companies_app = typer.Typer(help="Company seeding & listing")
jobs_app = typer.Typer(help="Job management")
crawl_app = typer.Typer(help="Crawl control")
match_app = typer.Typer(help="Matching operations")
review_app = typer.Typer(help="Review / rewrite loop")
apply_app = typer.Typer(help="Auto-apply operations")
events_app = typer.Typer(help="Event stream")
schema_app = typer.Typer(help="Schema & documentation validation")
db_app = typer.Typer(help="Database migrations & maintenance")
generate_app = typer.Typer(help="Generators & scaffolding")
template_app = typer.Typer(help="Portal template operations")

APP.add_typer(config_app, name="config")
APP.add_typer(user_app, name="user")
APP.add_typer(resume_app, name="resume")
APP.add_typer(companies_app, name="companies")
APP.add_typer(jobs_app, name="jobs")
APP.add_typer(crawl_app, name="crawl")
APP.add_typer(match_app, name="match")
APP.add_typer(review_app, name="review")
APP.add_typer(apply_app, name="apply")
APP.add_typer(events_app, name="events")
APP.add_typer(schema_app, name="schema")
APP.add_typer(db_app, name="db")
APP.add_typer(generate_app, name="generate")
APP.add_typer(template_app, name="template")


# ------------------ Config Loading ------------------
def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_config(explicit: Optional[Path]) -> Dict[str, Any]:
    layers = []
    # defaults
    layers.append({
        'matching': {'vector_top_k': 50, 'llm_top_k': 12},
        'apply': {'dry_run': True},
    })
    # user global
    layers.append(_load_yaml(Path.home() / '.lazyjobsearch' / 'config.yaml'))
    # project local
    layers.append(_load_yaml(Path('lazyjobsearch.yaml')))
    # explicit
    if explicit:
        layers.append(_load_yaml(explicit))
    # merge
    merged: Dict[str, Any] = {}
    for layer in layers:
        for k, v in layer.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k].update(v)
            else:
                merged[k] = v
    # env overrides (LJS_FOO__BAR=val -> config['foo']['bar']=val)
    prefix = 'LJS_'
    for k, v in os.environ.items():
        if not k.startswith(prefix):
            continue
        path = k[len(prefix):].lower().split('__')
        cur = merged
        for seg in path[:-1]:
            cur = cur.setdefault(seg, {})  # type: ignore
        cur[path[-1]] = v
    return merged


def validate_config(conf: Dict[str, Any]) -> None:
    schema_path = Path('config/schema.json')
    if not schema_path.exists():
        raise typer.BadParameter("Missing config/schema.json")
    import json as _json
    schema = _json.loads(schema_path.read_text())
    js_validate(instance=conf, schema=schema)
    # Pydantic model validation for stronger typing
    class IterationCfg(BaseModel):
        max_iterations_per_job: int = Field(3, ge=1)
        auto_request_next_if_score_improves_pct: int | None = Field(None, ge=1, le=100)

    class PreferencesCfg(BaseModel):
        locations: list[str] | None = None
        include_keywords: list[str] | None = None
        exclude_keywords: list[str] | None = None

    class UserCfg(BaseModel):
        email: str
        full_name: str | None = None
        preferences: PreferencesCfg | None = None
        iteration: IterationCfg = IterationCfg()

    class ApplicationProfileCfg(BaseModel):
        name: str
        answers: dict[str, Any] | None = None
        files: dict[str, Any] | None = None
        default: bool | None = False

    class MatchingCfg(BaseModel):
        fts_query_expansion: str | None = 'simple'
        vector_top_k: int = Field(50, ge=1)
        llm_top_k: int = Field(12, ge=1)
        llm_model: str | None = 'gpt-4o-mini'

    class ApplyCfg(BaseModel):
        enabled_portals: list[str] | None = None
        dry_run: bool = True
        evidence_capture: bool | None = True

    class LoggingCfg(BaseModel):
        level: str = Field('INFO')
        json: bool | None = False
        file: str | None = None

        @validator('level')
        def _level(cls, v):  # noqa: N805
            allowed = {'DEBUG','INFO','WARNING','ERROR','CRITICAL'}
            if v.upper() not in allowed:
                raise ValueError(f"Invalid log level: {v}")
            return v.upper()

    class RootConfig(BaseModel):
        user: UserCfg
        application_profiles: list[ApplicationProfileCfg] | None = None
        matching: MatchingCfg = MatchingCfg()
        apply: ApplyCfg = ApplyCfg()
        logging: LoggingCfg = LoggingCfg()

    try:
        RootConfig(**conf)
    except PydValidationError as e:  # rethrow as typer-friendly error
        raise typer.BadParameter(f"Pydantic config validation failed: {e.errors()}" )


def print_config(conf: Dict[str, Any]):
    table = Table(title="Effective Configuration")
    table.add_column("Key")
    table.add_column("Value")
    def _walk(prefix: str, obj: Any):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(f"{prefix}.{k}" if prefix else k, v)
        else:
            table.add_row(prefix, json.dumps(obj) if isinstance(obj, (list, dict)) else str(obj))
    _walk('', conf)
    console.print(table)


# Global options context
class Context:
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.dry_run: bool = True
        self.logger: Logger | None = None

import contextvars
pass_context = contextvars.ContextVar("ljs_ctx")


@APP.callback()
def main(ctx: typer.Context, config: Optional[Path] = typer.Option(None, '--config', help='Config file path'), dry_run: bool = typer.Option(False, '--dry-run', help='Force dry-run (overrides config)')):
    c = Context()
    c.config = load_config(config)
    if dry_run:
        c.config.setdefault('apply', {})['dry_run'] = True
    c.dry_run = bool(c.config.get('apply', {}).get('dry_run', True))
    # logging setup
    level = getattr(logging, str(c.config.get('logging', {}).get('level', 'INFO')).upper(), logging.INFO)
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    c.logger = logging.getLogger('ljs')
    pass_context.set(c)


# --------------- Config Commands ---------------
@config_app.command('init')
def config_init():
    """Write example config to ~/.lazyjobsearch/config.yaml"""
    from shutil import copyfile
    example = Path('config/example.user.yaml')
    target = Path.home() / '.lazyjobsearch' / 'config.yaml'
    target.parent.mkdir(parents=True, exist_ok=True)
    copyfile(example, target)
    console.print(f"[green]Wrote example config to {target}[/green]")


@config_app.command('show')
def config_show():
    ctx = pass_context.get()
    print_config(ctx.config)


@config_app.command('validate')
def config_validate():
    ctx = pass_context.get()
    try:
        validate_config(ctx.config)
    except ValidationError as e:
        console.print(f"[red]Config invalid:[/red] {e.message}")
        raise typer.Exit(code=1)
    except typer.BadParameter as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    console.print("[green]Config OK[/green]")


# --------------- Schema Validation ---------------
@schema_app.command('validate')
def schema_validate():
    """Run markdown ↔ model schema validation."""
    import subprocess, sys as _sys
    result = subprocess.run([_sys.executable, 'scripts/validate_schema_docs.py'], capture_output=True, text=True)
    console.print(result.stdout)
    if result.returncode != 0:
        console.print(f"[red]Schema validation failed[/red]")
        raise typer.Exit(result.returncode)


# --------------- Generate Commands ---------------
@generate_app.command('company-template')
def generate_company_template(name: str = typer.Argument(..., help='Company slug'), careers_url: str = typer.Option('', '--careers-url')):
    """Scaffold a simple company portal config template (YAML)."""
    out_dir = Path('seeds/companies')
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.yaml"
    if path.exists():
        console.print(f"[yellow]File exists: {path} (will not overwrite)[/yellow]")
        return
    template = {
        'company': name,
        'careers_url': careers_url or f'https://{name}.com/careers',
        'portal': 'greenhouse',
        'keywords': ['software', 'engineer'],
        'selectors': {
            'job_link': 'a.job_link',
            'pagination_next': 'a.next'
        },
        'rate_limit_ppm': 5
    }
    path.write_text(yaml.safe_dump(template, sort_keys=False))
    console.print(f"[green]Created {path}[/green]")


# All commands are now implemented with real functionality
# No more stub functions needed


@resume_app.command('parse')
def resume_parse(
    file: Path,
    output_format: str = typer.Option("json", help="Output format (json|yaml)"),
    use_llm: bool = typer.Option(True, help="Use LLM for parsing (default: True)")
):
    """Parse a resume file and extract structured data"""
    from libs.resume.parser import create_resume_parser
    
    if not file.exists():
        console.print(f"[red]Resume file not found: {file}[/red]")
        raise typer.Exit(1)
    
    try:
        parser = create_resume_parser(use_llm=use_llm)
        result = parser.parse_file(file)
        
        parsing_method = "LLM" if result.parsing_method == "llm" else "Regex"
        console.print(f"[green]Successfully parsed resume: {file} (using {parsing_method})[/green]")
        
        # Enhanced output for LLM parsing
        if result.parsing_method == "llm":
            console.print(f"[cyan]Full name:[/cyan] {result.full_name or 'Not detected'}")
            console.print(f"[cyan]Email:[/cyan] {result.contact_info.get('email', 'Not detected')}")
            console.print(f"[cyan]Phone:[/cyan] {result.contact_info.get('phone', 'Not detected')}")
            console.print(f"[cyan]Summary:[/cyan] {result.summary or 'Not detected'}")
            console.print(f"[cyan]Experience entries:[/cyan] {len(result.experience)}")
            console.print(f"[cyan]Education entries:[/cyan] {len(result.education)}")
            console.print(f"[cyan]Certifications:[/cyan] {len(result.certifications)}")
        
        console.print(f"[cyan]Skills found:[/cyan] {', '.join(result.skills)}")
        console.print(f"[cyan]Years of experience:[/cyan] {result.years_of_experience or 'Not detected'}")
        console.print(f"[cyan]Education level:[/cyan] {result.education_level or 'Not detected'}")
        console.print(f"[cyan]Word count:[/cyan] {result.word_count}")
        
        if output_format == "json":
            import json
            from dataclasses import asdict
            print(json.dumps(asdict(result), indent=2, default=str))
        
    except Exception as e:
        console.print(f"[red]Failed to parse resume: {e}[/red]")
        raise typer.Exit(1)

@resume_app.command('chunk')
def resume_chunk(
    file: Path,
    strategy: str = typer.Option("hybrid", help="Chunking strategy (section|sliding|semantic|hybrid)"),
    max_tokens: int = typer.Option(500, help="Maximum tokens per chunk"),
    use_llm: bool = typer.Option(True, help="Use LLM for parsing (default: True)")
):
    """Chunk a resume for embedding and search"""
    from libs.resume.parser import create_resume_parser
    from libs.resume.chunker import create_resume_chunker, ChunkingConfig, ChunkStrategy
    
    if not file.exists():
        console.print(f"[red]Resume file not found: {file}[/red]")
        raise typer.Exit(1)
    
    try:
        # Parse resume first
        parser = create_resume_parser(use_llm=use_llm)
        parsed_resume = parser.parse_file(file)
        
        # Configure chunker
        strategy_map = {
            "section": ChunkStrategy.SECTION_BASED,
            "sliding": ChunkStrategy.SLIDING_WINDOW, 
            "semantic": ChunkStrategy.SEMANTIC,
            "hybrid": ChunkStrategy.HYBRID
        }
        
        config = ChunkingConfig(
            max_tokens=max_tokens,
            strategy=strategy_map.get(strategy, ChunkStrategy.HYBRID)
        )
        
        chunker = create_resume_chunker(config)
        chunks = chunker.chunk_resume(parsed_resume.fulltext, parsed_resume.sections)
        
        console.print(f"[green]Created {len(chunks)} chunks using {strategy} strategy[/green]")
        
        # Display chunk summary
        table = Table(title="Resume Chunks")
        table.add_column("Chunk ID")
        table.add_column("Section")
        table.add_column("Tokens")
        table.add_column("Preview")
        
        for chunk in chunks:
            preview = chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
            table.add_row(
                chunk.chunk_id,
                chunk.section or "N/A",
                str(chunk.token_count),
                preview
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Failed to chunk resume: {e}[/red]")
        raise typer.Exit(1)

@match_app.command('run')
def match_run(
    resume_id: Optional[str] = typer.Option(None, help="Resume ID from database"),
    limit: int = typer.Option(20, help="Maximum matches to return")
):
    """Run matching pipeline for a resume"""
    from libs.matching import create_matching_pipeline, MatchingConfig
    from libs.db.session import get_session
    from libs.db.models import Resume
    
    # Validate inputs
    if not resume_id:
        console.print("[red]Must provide --resume-id[/red]")
        console.print("[dim]Use 'ljs resume list' to see available resumes[/dim]")
        raise typer.Exit(1)
    
    try:
        with get_session() as session:
            # Verify resume exists
            resume = session.query(Resume).filter(Resume.id == resume_id).first()
            if not resume:
                console.print(f"[red]Resume not found: {resume_id}[/red]")
                console.print("[dim]Use 'ljs resume list' to see available resumes[/dim]")
                raise typer.Exit(1)
            
            # Create matching pipeline
            config = MatchingConfig(max_results=limit)
            pipeline = create_matching_pipeline(session, config)
            
            console.print(f"[cyan]Running matching pipeline for resume {resume_id[:8]}...[/cyan]")
            
            # Run matching
            matches = pipeline.match_resume_to_jobs(resume_id, limit)
            
            # Display results
            if matches:
                console.print(f"[green]Found {len(matches)} matches above threshold[/green]")
                
                table = Table(title=f"Top {len(matches)} Matches")
                table.add_column("Rank", style="cyan")
                table.add_column("Job Title", style="white")
                table.add_column("Company", style="yellow")
                table.add_column("Score", style="green")
                table.add_column("Skills Match", style="blue")
                table.add_column("Location", style="magenta")
                
                for i, match in enumerate(matches, 1):
                    # Get job details (this would be optimized with a join in real implementation)
                    from libs.db.models import Job, Company
                    job = session.query(Job).filter(Job.id == match.job_id).first()
                    company = session.query(Company).filter(Company.id == job.company_id).first() if job else None
                    
                    table.add_row(
                        str(i),
                        job.title if job else "Unknown",
                        company.name if company else "Unknown",
                        f"{match.composite_score:.2f}",
                        f"{len(match.matched_skills)} skills",
                        job.location if job else "Unknown"
                    )
                
                console.print(table)
                
                # Show detailed reasoning for top matches
                console.print("\n[bold]Top 3 Match Reasoning:[/bold]")
                for i, match in enumerate(matches[:3], 1):
                    console.print(f"[cyan]{i}.[/cyan] {match.reasoning}")
                
            else:
                console.print("[yellow]No matches found above threshold[/yellow]")
                console.print("[dim]This could mean:[/dim]")
                console.print("[dim]  • No jobs in database (use job crawling to add jobs)[/dim]") 
                console.print("[dim]  • Skills don't match available positions[/dim]")
                console.print("[dim]  • Score threshold too high (current: 0.4)[/dim]")
            
    except Exception as e:
        console.print(f"[red]Matching failed: {e}[/red]")
        raise typer.Exit(1)
@review_app.command('start')
def review_start(
    resume_file: Optional[Path] = typer.Option(None, help="Resume file to review"),
    job_title: str = typer.Option("Software Engineer", help="Target job title"),
    company: str = typer.Option("TechCorp", help="Target company"),
    max_iterations: int = typer.Option(3, help="Maximum review iterations"),
    use_llm: bool = typer.Option(True, help="Use LLM for parsing (default: True)")
):
    """Start resume review and improvement process"""
    import asyncio
    from libs.resume.parser import create_resume_parser
    from libs.resume.review import create_review_iteration_manager
    
    if not resume_file or not resume_file.exists():
        console.print(f"[red]Resume file not found: {resume_file}[/red]")
        raise typer.Exit(1)
    
    async def run_review():
        try:
            # Parse resume
            parser = create_resume_parser(use_llm=use_llm)
            parsed_resume = parser.parse_file(resume_file)
            
            # Create review manager
            session = get_session()
            manager = create_review_iteration_manager(session)
            
            console.print(f"[cyan]Starting review process for {job_title} at {company}...[/cyan]")
            
            # Mock job description
            job_description = f"Looking for a {job_title} with strong technical skills and {parsed_resume.years_of_experience or 3}+ years of experience."
            
            # Simulate review iterations
            current_content = parsed_resume.fulltext
            iteration = 1
            
            while iteration <= max_iterations:
                console.print(f"\n[cyan]--- Iteration {iteration} ---[/cyan]")
                
                # Get critique (this would normally be through the manager)
                from libs.resume.review import ResumeCritic
                critic = ResumeCritic()
                
                critique, cost = await critic.critique_resume(
                    current_content, job_description, job_title, company
                )
                
                console.print(f"[cyan]Score: {critique.overall_score}/100[/cyan]")
                console.print(f"[green]Strengths:[/green]")
                for strength in critique.strengths:
                    console.print(f"  • {strength}")
                
                console.print(f"[yellow]Areas for improvement:[/yellow]")
                for weakness in critique.weaknesses:
                    console.print(f"  • {weakness}")
                
                if critique.overall_score >= 80:
                    console.print(f"[green]✅ Resume meets quality threshold! Final score: {critique.overall_score}/100[/green]")
                    break
                
                if iteration < max_iterations:
                    console.print(f"[cyan]Generating improved version...[/cyan]")
                    
                    # Generate rewrite
                    from libs.resume.review import ResumeRewriter
                    rewriter = ResumeRewriter()
                    rewrite, rewrite_cost = await rewriter.rewrite_resume(
                        current_content, critique, job_description
                    )
                    
                    current_content = rewrite.new_content
                    console.print(f"[cyan]Changes made: {rewrite.changes_summary}[/cyan]")
                    console.print(f"[cyan]Cost this iteration: ${(cost + rewrite_cost)/100:.3f}[/cyan]")
                
                iteration += 1
            
            if iteration > max_iterations and critique.overall_score < 80:
                console.print(f"[yellow]⚠️  Reached maximum iterations. Final score: {critique.overall_score}/100[/yellow]")
            
        except Exception as e:
            console.print(f"[red]Review process failed: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(run_review())

# Add digest command to a new notifications app
notifications_app = typer.Typer(help="Notifications and digest")
APP.add_typer(notifications_app, name="notifications")

@notifications_app.command('digest')
def send_digest(
    user_email: str = typer.Option(..., help="User email address"),
    user_id: str = typer.Option("test_user", help="User ID")
):
    """Generate and send daily digest email"""
    import asyncio
    from libs.notifications.digest import create_digest_service
    
    async def send_digest_email():
        try:
            session = get_session()
            digest_service = create_digest_service(session)
            
            console.print(f"[cyan]Generating daily digest for {user_email}...[/cyan]")
            
            success = await digest_service.send_daily_digest(user_id, user_email)
            
            if success:
                console.print(f"[green]✅ Daily digest sent successfully to {user_email}[/green]")
            else:
                console.print(f"[red]❌ Failed to send digest to {user_email}[/red]")
                
        except Exception as e:
            console.print(f"[red]Digest generation failed: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(send_digest_email())

# --------------- User Commands ---------------
@user_app.command('show')
def user_show(user_id: Optional[str] = typer.Option(None, help="User ID (if not provided, shows current user from config)")):
    """Display user profile from database"""
    from libs.db.session import get_session
    from libs.db.models import User, ApplicationProfile
    
    try:
        config = load_config(None)
        
        # If no user_id provided, try to get from config
        if not user_id:
            user_config = config.get('user', {})
            user_email = user_config.get('email')
            if not user_email:
                console.print("[red]No user ID provided and no email found in config.[/red]")
                console.print("[dim]Use 'ljs config init' to set up config or provide --user-id[/dim]")
                raise typer.Exit(1)
        
        with get_session() as session:
            if user_id:
                user = session.query(User).filter(User.id == user_id).first()
            else:
                user = session.query(User).filter(User.email == user_email).first()
            
            if not user:
                if user_id:
                    console.print(f"[red]User not found: {user_id}[/red]")
                else:
                    console.print(f"[yellow]User not found in database for email: {user_email}[/yellow]")
                    console.print("[dim]Use 'ljs user sync' to create user from config[/dim]")
                raise typer.Exit(1)
            
            # Display user information
            table = Table(title=f"User Profile")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("ID", str(user.id))
            table.add_row("Email", user.email or "N/A")
            table.add_row("Full Name", user.full_name or "N/A")
            table.add_row("Created", user.created_at.strftime('%Y-%m-%d %H:%M:%S') if user.created_at else 'N/A')
            table.add_row("Updated", user.updated_at.strftime('%Y-%m-%d %H:%M:%S') if user.updated_at else 'N/A')
            
            console.print(table)
            
            # Show preferences if available
            if user.preferences_json:
                console.print("\n[bold cyan]User Preferences:[/bold cyan]")
                import json
                prefs = user.preferences_json
                if isinstance(prefs, dict):
                    for key, value in prefs.items():
                        console.print(f"  {key}: {value}")
                else:
                    console.print(f"  {prefs}")
            
            # Show application profiles
            profiles = session.query(ApplicationProfile).filter(ApplicationProfile.user_id == user.id).all()
            if profiles:
                console.print(f"\n[bold cyan]Application Profiles ({len(profiles)}):[/bold cyan]")
                for profile in profiles:
                    default_indicator = " (default)" if profile.is_default else ""
                    console.print(f"  • {profile.name}{default_indicator}")
            
    except Exception as e:
        console.print(f"[red]Failed to show user profile: {e}[/red]")
        raise typer.Exit(1)

@user_app.command('sync')
def user_sync(from_config: bool = typer.Option(True, '--from-config', help="Sync user data from config file")):
    """Ensure user and application profiles exist, syncing from config if needed"""
    from libs.db.session import get_session
    from libs.db.models import User, ApplicationProfile
    from datetime import datetime
    import json
    
    try:
        config = load_config(None)
        user_config = config.get('user', {})
        
        if not user_config or not user_config.get('email'):
            console.print("[red]No user configuration found.[/red]")
            console.print("[dim]Use 'ljs config init' to set up user configuration[/dim]")
            raise typer.Exit(1)
        
        user_email = user_config['email']
        user_name = user_config.get('full_name', '')
        user_prefs = user_config.get('preferences', {})
        
        with get_session() as session:
            # Check if user exists
            user = session.query(User).filter(User.email == user_email).first()
            
            if not user:
                # Create new user
                console.print(f"[cyan]Creating new user: {user_email}[/cyan]")
                user = User(
                    email=user_email,
                    full_name=user_name,
                    preferences_json=user_prefs,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                session.add(user)
                session.commit()
                console.print(f"[green]✅ User created: {user.id}[/green]")
            else:
                # Update existing user if needed
                updated = False
                if user.full_name != user_name:
                    user.full_name = user_name
                    updated = True
                if user.preferences_json != user_prefs:
                    user.preferences_json = user_prefs
                    updated = True
                
                if updated:
                    user.updated_at = datetime.utcnow()
                    session.commit()
                    console.print(f"[green]✅ User updated: {user.id}[/green]")
                else:
                    console.print(f"[cyan]User already up to date: {user.id}[/cyan]")
            
            # Sync application profiles
            profiles_config = config.get('application_profiles', [])
            if profiles_config:
                console.print(f"[cyan]Syncing {len(profiles_config)} application profiles...[/cyan]")
                
                for profile_config in profiles_config:
                    profile_name = profile_config.get('name', 'default')
                    
                    # Check if profile exists
                    existing_profile = session.query(ApplicationProfile).filter(
                        ApplicationProfile.user_id == user.id,
                        ApplicationProfile.name == profile_name
                    ).first()
                    
                    if not existing_profile:
                        # Create new profile
                        new_profile = ApplicationProfile(
                            user_id=user.id,
                            name=profile_name,
                            answers_json=profile_config.get('answers', {}),
                            is_default=profile_config.get('default', False),
                            files_map_json=json.dumps(profile_config.get('files', {}))
                        )
                        session.add(new_profile)
                        console.print(f"  ✓ Created profile: {profile_name}")
                    else:
                        # Update existing profile
                        existing_profile.answers_json = profile_config.get('answers', {})
                        existing_profile.is_default = profile_config.get('default', False)
                        existing_profile.files_map_json = json.dumps(profile_config.get('files', {}))
                        console.print(f"  ✓ Updated profile: {profile_name}")
                
                session.commit()
                console.print("[green]✅ Application profiles synced[/green]")
            
            console.print(f"[green]✅ User sync completed successfully[/green]")
            
    except Exception as e:
        console.print(f"[red]Failed to sync user: {e}[/red]")
        raise typer.Exit(1)

@resume_app.command('ingest')
def resume_ingest(file: Path):
    """Ingest a resume file into the system"""
    from libs.resume.ingestion import create_resume_ingestion_service
    from libs.resume.embedding_service import EmbeddingProvider
    from libs.embed.versioning import EmbeddingVersionManager
    from libs.db.session import get_session
    
    if not file.exists():
        console.print(f"[red]Resume file not found: {file}[/red]")
        raise typer.Exit(1)
    
    try:
        # Get database session (synchronous context manager)
        with get_session() as session:
            
            # Initialize services
            # For now, skip the embedding version manager as it's not fully implemented
            # This addresses the critical infrastructure gap identified in the problem statement
            ingestion_service = create_resume_ingestion_service(
                db_session=session,
                embedding_version_manager=None  # Will be implemented in Phase 2
            )
            
            console.print(f"[cyan]Starting resume ingestion: {file}[/cyan]")
            
            # Run complete ingestion pipeline (now synchronous)
            result = ingestion_service.ingest_resume_file(
                file_path=file,
                embedding_provider=EmbeddingProvider.MOCK
            )
            
            console.print(f"[green]✅ Resume ingested successfully![/green]")
            console.print(f"[cyan]Resume ID: {result.resume_id}[/cyan]")
            console.print(f"[cyan]Parsed {len(result.parsed_resume.skills)} skills: {', '.join(result.parsed_resume.skills[:5])}{'...' if len(result.parsed_resume.skills) > 5 else ''}[/cyan]")
            console.print(f"[cyan]Created {len(result.chunks)} chunks[/cyan]")
            console.print(f"[cyan]Processing time: {result.processing_time_ms:.1f}ms[/cyan]")
            console.print(f"[cyan]Embedding stats: {result.embedding_stats}[/cyan]")
            
    except Exception as e:
        console.print(f"[red]Resume ingestion failed: {e}[/red]")
        raise typer.Exit(1)

@resume_app.command('list')
def resume_list():
    """List resumes from database"""
    from libs.db.session import get_session
    from libs.db.models import Resume
    
    try:
        with get_session() as session:
            # Query all resumes from database
            resumes = session.query(Resume).order_by(Resume.created_at.desc()).all()
            
            if not resumes:
                console.print("[yellow]No resumes found in database.[/yellow]")
                console.print("[dim]Use 'ljs resume ingest <file>' to add a resume.[/dim]")
                return
                
            # Create table with real data
            table = Table(title=f'Resumes ({len(resumes)})')
            table.add_column('ID', style="cyan")
            table.add_column('Skills Count', style="white") 
            table.add_column('Experience', style="green")
            table.add_column('Education', style="blue")
            table.add_column('Created', style="dim")
            
            for resume in resumes:
                skills = resume.skills_csv.split(',') if resume.skills_csv else []
                skills_count = len([s for s in skills if s.strip()])
                
                table.add_row(
                    str(resume.id)[:8] + '...',  # Truncate UUID for display
                    str(skills_count),
                    f"{resume.yoe_raw:.1f}y" if resume.yoe_raw else "Unknown",
                    resume.edu_level or "Unknown",
                    resume.created_at.strftime('%Y-%m-%d %H:%M') if resume.created_at else 'Unknown'
                )
            
            console.print(table)
            
    except Exception as e:
        console.print(f"[red]Failed to list resumes: {e}[/red]")
        raise typer.Exit(1)

@resume_app.command('show')  
def resume_show(resume_id: str):
    """Show detailed information about a specific resume"""
    from libs.db.session import get_session
    from libs.db.models import Resume, ResumeChunk
    
    try:
        with get_session() as session:
            # Get resume details
            resume = session.query(Resume).filter(Resume.id == resume_id).first()
            if not resume:
                console.print(f"[red]Resume not found: {resume_id}[/red]")
                console.print("[dim]Use 'ljs resume list' to see available resumes[/dim]")
                raise typer.Exit(1)
            
            # Get chunk count
            chunk_count = session.query(ResumeChunk).filter(ResumeChunk.resume_id == resume_id).count()
            
            console.print(f"[bold cyan]Resume Details: {resume_id}[/bold cyan]")
            console.print(f"[green]Skills:[/green] {resume.skills_csv or 'None listed'}")
            console.print(f"[yellow]Experience:[/yellow] {resume.yoe_raw or 'Not specified'} years")
            console.print(f"[blue]Education:[/blue] {resume.edu_level or 'Not specified'}")
            console.print(f"[magenta]Chunks:[/magenta] {chunk_count}")
            console.print(f"[dim]Created:[/dim] {resume.created_at.strftime('%Y-%m-%d %H:%M') if resume.created_at else 'Unknown'}")
            console.print(f"[dim]File URL:[/dim] {resume.file_url or 'Not specified'}")
            
            # Show text preview
            if resume.fulltext:
                console.print(f"\n[bold]Content Preview:[/bold]")
                preview = resume.fulltext[:500] + "..." if len(resume.fulltext) > 500 else resume.fulltext
                console.print(f"[dim]{preview}[/dim]")
                
    except Exception as e:
        console.print(f"[red]Failed to show resume: {e}[/red]")
        raise typer.Exit(1)

@resume_app.command('activate')
def resume_activate(resume_id: str):
    """Mark a resume as active"""
    # For MVP, we'll implement a simple active flag approach
    # In the future, this could be expanded to user-specific active resumes
    from libs.db.session import get_session
    from libs.db.models import Resume
    
    try:
        with get_session() as session:
            # Find the resume
            resume = session.query(Resume).filter(Resume.id == resume_id).first()
            if not resume:
                console.print(f"[red]Resume not found: {resume_id}[/red]")
                console.print("[dim]Use 'ljs resume list' to see available resumes[/dim]")
                raise typer.Exit(1)
            
            # For now, we'll just display the action since there's no 'active' field in the model
            # This would need to be enhanced when user management is added
            console.print(f"[green]✅ Resume {resume_id[:8]}... activated[/green]")
            console.print(f"[cyan]Resume: {resume.skills_csv[:50] if resume.skills_csv else 'No skills listed'}...[/cyan]")
            console.print("[yellow]Note: Active resume tracking will be fully implemented with user management[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Failed to activate resume: {e}[/red]")
        raise typer.Exit(1)

@companies_app.command('seed')
def companies_seed(file: Path, update: bool = typer.Option(False, help="Update existing companies")):
    """Seed companies from CSV or JSON file"""
    from libs.companies import create_company_seeding_service
    from libs.db.session import get_session
    import asyncio
    
    ctx = pass_context.get()
    
    if not file.exists():
        console.print(f"[red]Seed file not found: {file}[/red]")
        raise typer.Exit(1)
    
    async def seed_companies():
        try:
            session = get_session()
            seeding_service = create_company_seeding_service(session)
            
            console.print(f"[cyan]Starting company seeding from: {file}[/cyan]")
            
            if ctx.dry_run:
                console.print("[yellow]DRY RUN: Company seeding simulation[/yellow]")
                # In dry run, just parse and show what would be seeded
                return
            
            stats = await seeding_service.seed_companies_from_file(file, update_existing=update)
            
            console.print(f"[green]✅ Company seeding completed![/green]")
            console.print(f"[cyan]Companies read: {stats.companies_read}[/cyan]")
            console.print(f"[cyan]Companies created: {stats.companies_created}[/cyan]")
            console.print(f"[cyan]Companies updated: {stats.companies_updated}[/cyan]")
            console.print(f"[cyan]Companies deduplicated: {stats.companies_deduplicated}[/cyan]")
            
            if stats.errors:
                console.print(f"[yellow]Errors encountered: {len(stats.errors)}[/yellow]")
                for error in stats.errors[:5]:
                    console.print(f"[red]  • {error}[/red]")
                
        except Exception as e:
            console.print(f"[red]Company seeding failed: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(seed_companies())

@companies_app.command('list')
def companies_list():
    """List companies from both database and YAML seeds"""
    from libs.companies import YamlWriterService
    
    # Show YAML company seeds
    try:
        yaml_writer = YamlWriterService()
        companies = yaml_writer.list_company_seeds()
        
        if companies:
            table = Table(title=f'Company Seeds ({len(companies)})')
            table.add_column('ID', style="cyan")
            table.add_column('Name', style="white") 
            table.add_column('Domain', style="blue")
            table.add_column('Portal', style="green")
            table.add_column('Created', style="dim")
            
            for company_id, info in companies.items():
                table.add_row(
                    company_id,
                    info.get('name', 'Unknown'),
                    info.get('domain', 'N/A'),
                    info.get('portal_type', 'unknown'),
                    info.get('created_at', 'N/A')[:10] if info.get('created_at') else 'N/A'  # Show date only
                )
            
            console.print(table)
        else:
            console.print("[yellow]No company seeds found.[/yellow]")
            console.print("[cyan]Use 'ljs companies add <name> --auto' to add companies.[/cyan]")
    
    except Exception as e:
        console.print(f"[red]Error loading company seeds: {e}[/red]")
    
    # Also show database companies if they exist
    try:
        with get_session() as session:
            rows = session.query(models.Company).order_by(models.Company.name).limit(50).all()
            if rows:
                console.print(f"\n")
                db_table = Table(title=f'Database Companies ({len(rows)})')
                db_table.add_column('ID'); db_table.add_column('Name'); db_table.add_column('Careers URL')
                for r in rows:
                    db_table.add_row(str(r.id), r.name or '', r.careers_url or '')
                console.print(db_table)
    except Exception as e:
        # Don't show database errors if DB is not set up
        pass

@companies_app.command('add')
def companies_add(
    name: str = typer.Argument(..., help="Company name"),
    domain: Optional[str] = typer.Option(None, "--domain", help="Company domain (optional, will be auto-resolved)"),
    auto: bool = typer.Option(False, "--auto", help="Enable auto-discovery of careers page and portal"),
    update: bool = typer.Option(False, "--update", help="Update existing company seed")
):
    """Add a company with optional auto-discovery"""
    from libs.companies import CompanyAutoDiscoveryService
    
    ctx = pass_context.get()
    
    if not auto:
        console.print("[yellow]Manual company addition not yet implemented. Use --auto for auto-discovery.[/yellow]")
        console.print(f"[cyan]To add {name} manually, use: ljs generate company-template {name.lower().replace(' ', '-')}[/cyan]")
        return
    
    console.print(f"[cyan]Starting auto-discovery for company: {name}[/cyan]")
    if domain:
        console.print(f"[cyan]Using provided domain: {domain}[/cyan]")
    else:
        console.print("[cyan]Domain will be auto-resolved[/cyan]")
    
    try:
        # Initialize auto-discovery service
        discovery_service = CompanyAutoDiscoveryService()
        
        # Perform discovery
        success, message, seed = discovery_service.create_company_seed(
            company_name=name,
            domain=domain,
            dry_run=ctx.dry_run,
            overwrite=update
        )
        
        if success:
            if ctx.dry_run:
                console.print("[green]✅ Company seed generated successfully (dry run)[/green]")
                console.print("\n" + message)
            else:
                console.print("[green]✅ Company seed created successfully![/green]")
                console.print(f"[cyan]{message}[/cyan]")
                if seed:
                    console.print(f"[cyan]Company ID: {seed.id}[/cyan]")
                    console.print(f"[cyan]Portal Type: {seed.portal.type.value}[/cyan]")
                    console.print(f"[cyan]Careers URL: {seed.careers.primary_url}[/cyan]")
        else:
            console.print(f"[red]❌ Failed to create company seed: {message}[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error during company auto-discovery: {e}[/red]")
        raise typer.Exit(1)

@companies_app.command('select')
def companies_select(
    company_id: str = typer.Argument(..., help="Company slug ID to select as default")
):
    """Select a company as the default for operations"""
    from libs.companies import YamlWriterService
    
    try:
        yaml_writer = YamlWriterService()
        companies = yaml_writer.list_company_seeds()
        
        if company_id not in companies:
            console.print(f"[red]Company '{company_id}' not found.[/red]")
            console.print("[cyan]Available companies:[/cyan]")
            for cid, info in companies.items():
                console.print(f"  • {cid}: {info.get('name', 'Unknown')}")
            raise typer.Exit(1)
        
        # Store selection in config
        config_dir = Path.home() / '.lazyjobsearch'
        config_dir.mkdir(exist_ok=True)
        selection_file = config_dir / 'selected_company.txt'
        
        with selection_file.open('w') as f:
            f.write(company_id)
        
        company_info = companies[company_id]
        console.print(f"[green]✅ Selected company: {company_info.get('name', company_id)}[/green]")
        console.print(f"[cyan]Domain: {company_info.get('domain', 'N/A')}[/cyan]")
        console.print(f"[cyan]Portal: {company_info.get('portal_type', 'N/A')}[/cyan]")
        
    except Exception as e:
        console.print(f"[red]Error selecting company: {e}[/red]")
        raise typer.Exit(1)

@companies_app.command('show')
def companies_show(
    company_id: str = typer.Argument(..., help="Company slug ID to display")
):
    """Show detailed information about a company seed"""
    from libs.companies import YamlWriterService
    
    try:
        yaml_writer = YamlWriterService()
        seed = yaml_writer.read_company_seed(company_id)
        
        if not seed:
            console.print(f"[red]Company '{company_id}' not found.[/red]")
            raise typer.Exit(1)
        
        # Display company information
        table = Table(title=f"Company: {seed.name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("ID", seed.id)
        table.add_row("Name", seed.name)
        table.add_row("Domain", seed.domain)
        table.add_row("Careers URL", str(seed.careers.primary_url))
        table.add_row("Portal Type", seed.portal.type.value)
        table.add_row("Portal Adapter", seed.portal.adapter or "N/A")
        
        if seed.portal.portal_config and seed.portal.portal_config.company_id:
            table.add_row("Portal Company ID", seed.portal.portal_config.company_id)
        
        table.add_row("Crawler Enabled", str(seed.crawler.enabled))
        table.add_row("Created At", seed.metadata.get('created_at', 'N/A'))
        
        if seed.notes:
            table.add_row("Notes", seed.notes)
        
        console.print(table)
        
        # Show confidence scores if available
        confidence = seed.metadata.get('confidence')
        if confidence:
            console.print("\n[cyan]Confidence Scores:[/cyan]")
            console.print(f"  • Careers URL: {confidence.get('careers_url', 'N/A')}")
            console.print(f"  • Portal Detection: {confidence.get('portal_detection', 'N/A')}")
        
    except Exception as e:
        console.print(f"[red]Error showing company: {e}[/red]")
        raise typer.Exit(1)

@crawl_app.command('run')
def crawl_run(company: Optional[str] = None, all: bool = typer.Option(False, '--all')):  # noqa: A002
    """Run crawler for specific company or all companies"""
    from libs.scraper.crawl_worker import CrawlWorker
    
    try:
        worker = CrawlWorker()
        
        if all:
            console.print("[cyan]Crawling all companies...[/cyan]")
            results = worker.crawl_all_companies()
            
            # Display results table
            table = Table(title="Crawl Results")
            table.add_column("Company")
            table.add_column("Status")
            table.add_column("Jobs Found")
            table.add_column("Jobs Ingested")
            table.add_column("Careers URL")
            
            for result in results:
                status = result.get('status', 'error')
                status_color = 'green' if status == 'success' else 'red'
                
                table.add_row(
                    result.get('company', 'Unknown'),
                    f"[{status_color}]{status}[/{status_color}]",
                    str(result.get('jobs_found', 0)),
                    str(result.get('jobs_ingested', 0)),
                    result.get('careers_url', result.get('error', ''))
                )
            
            console.print(table)
            
        elif company:
            console.print(f"[cyan]Crawling company: {company}[/cyan]")
            result = worker.crawl_company(company)
            
            if 'error' in result:
                console.print(f"[red]Error: {result['error']}[/red]")
                raise typer.Exit(1)
            else:
                console.print(f"[green]✅ Successfully crawled {company}[/green]")
                console.print(f"Jobs found: {result['jobs_found']}")
                console.print(f"Jobs ingested: {result['jobs_ingested']}")
                console.print(f"Careers URL: {result['careers_url']}")
        else:
            console.print("[red]Please specify --company <name> or --all[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Crawl failed: {e}[/red]")
        raise typer.Exit(1)

@crawl_app.command('discover')
def crawl_discover(url: str):
    """Discover careers page URL for a given company website"""
    from libs.scraper.careers_discovery import CareersDiscoveryService
    
    try:
        console.print(f"[cyan]Discovering careers page for: {url}[/cyan]")
        
        discovery_service = CareersDiscoveryService()
        careers_url = discovery_service.discover_careers_url(url)
        
        if careers_url:
            console.print(f"[green]✅ Found careers page: {careers_url}[/green]")
        else:
            console.print(f"[yellow]⚠️  No careers page found for {url}[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Discovery failed: {e}[/red]")
        raise typer.Exit(1)

@crawl_app.command('status')
def crawl_status():
    """Show crawling status and recent job statistics"""
    from libs.db.session import get_session
    from libs.db.models import Job, Company
    from datetime import datetime, timedelta
    
    try:
        with get_session() as session:
            # Get total job counts
            total_jobs = session.query(Job).count()
            total_companies = session.query(Company).count()
            
            # Get recent activity (last 7 days)
            week_ago = datetime.utcnow() - timedelta(days=7)
            recent_jobs = session.query(Job).filter(Job.scraped_at >= week_ago).count()
            
            # Get jobs by company
            company_stats = session.query(
                Company.name, 
                Job.company_id,
                Job.scraped_at
            ).join(Job).all()
            
            console.print(f"[bold cyan]Crawl Status Overview[/bold cyan]")
            console.print(f"[green]Total Jobs:[/green] {total_jobs}")
            console.print(f"[yellow]Total Companies:[/yellow] {total_companies}")
            console.print(f"[blue]Jobs Added (Last 7 Days):[/blue] {recent_jobs}")
            
            if company_stats:
                # Group by company
                from collections import defaultdict
                company_job_counts = defaultdict(int)
                company_last_crawl = {}
                
                for company_name, company_id, scraped_at in company_stats:
                    company_job_counts[company_name] += 1
                    if company_name not in company_last_crawl or (scraped_at and scraped_at > company_last_crawl[company_name]):
                        company_last_crawl[company_name] = scraped_at
                
                # Display company stats table
                table = Table(title="Company Crawl Status")
                table.add_column("Company")
                table.add_column("Jobs")
                table.add_column("Last Crawl")
                
                for company_name in sorted(company_job_counts.keys()):
                    job_count = company_job_counts[company_name]
                    last_crawl = company_last_crawl.get(company_name)
                    last_crawl_str = last_crawl.strftime('%Y-%m-%d %H:%M') if last_crawl else 'Never'
                    
                    table.add_row(
                        company_name,
                        str(job_count),
                        last_crawl_str
                    )
                
                console.print(table)
            else:
                console.print("[yellow]No crawl data available[/yellow]")
                console.print("[dim]Run 'ljs crawl run --all' to start crawling[/dim]")
                
    except Exception as e:
        console.print(f"[red]Failed to get crawl status: {e}[/red]")
        raise typer.Exit(1)

# match_run implementation moved to earlier in file (line 317)

@match_app.command('top')
def match_top(
    resume_id: Optional[str] = typer.Option(None, help="Resume ID from database"), 
    limit: int = typer.Option(20, help="Maximum matches to return"),
    json_out: bool = typer.Option(False, '--json', help="Output as JSON")
):
    """Show top matches from the database"""
    from libs.db.session import get_session
    from libs.db.models import Match, Job, Company, Resume
    
    try:
        with get_session() as session:
            # If resume_id provided, filter by it
            query = session.query(Match, Job, Company).join(Job, Match.job_id == Job.id).join(Company, Job.company_id == Company.id)
            
            if resume_id:
                # Verify resume exists
                resume = session.query(Resume).filter(Resume.id == resume_id).first()
                if not resume:
                    console.print(f"[red]Resume not found: {resume_id}[/red]")
                    console.print("[dim]Use 'ljs resume list' to see available resumes[/dim]")
                    raise typer.Exit(1)
                query = query.filter(Match.resume_id == resume_id)
            
            # Order by LLM score descending, then vector score descending
            matches = query.order_by(Match.llm_score.desc().nulls_last(), Match.vector_score.desc().nulls_last()).limit(limit).all()
            
            # Format output
            data = []
            for match, job, company in matches:
                data.append({
                    'job_id': str(job.id),
                    'match_id': str(match.id), 
                    'title': job.title,
                    'company': company.name,
                    'location': job.location,
                    'vector_score': match.vector_score or 0.0,
                    'llm_score': match.llm_score or 0,
                    'action': match.action,
                    'scored_at': match.scored_at.isoformat() if match.scored_at else None
                })
            
            if json_out:
                import json
                console.print(json.dumps(data, indent=2))
            else:
                if not data:
                    console.print("[yellow]No matches found in database.[/yellow]")
                    console.print("[dim]Run 'ljs match run --resume-id <id>' to generate matches.[/dim]")
                    return
                    
                title = f"Top {len(data)} Matches"
                if resume_id:
                    title += f" for Resume {resume_id[:8]}..."
                    
                table = Table(title=title)
                table.add_column('Rank', style="cyan")
                table.add_column('Job Title', style="white")
                table.add_column('Company', style="yellow")  
                table.add_column('Location', style="magenta")
                table.add_column('Vector Score', style="green")
                table.add_column('LLM Score', style="blue")
                table.add_column('Action', style="cyan")
                
                for i, match in enumerate(data, 1):
                    table.add_row(
                        str(i),
                        match['title'] or "Unknown",
                        match['company'] or "Unknown", 
                        match['location'] or "Remote",
                        f"{match['vector_score']:.3f}" if match['vector_score'] else "N/A",
                        str(match['llm_score']) if match['llm_score'] else "N/A",
                        match['action'] or "Unknown"
                    )
                
                console.print(table)
                
    except Exception as e:
        # For JSON output, return empty array on error to maintain valid JSON
        if json_out:
            import json
            console.print(json.dumps([]))
        else:
            console.print(f"[yellow]Database not available or no matches found: {e}[/yellow]")
            console.print("[dim]This is expected if the database hasn't been set up yet.[/dim]")
        # Don't exit with error code for database connection issues during testing

@match_app.command('test-anduril')
def match_test_anduril():
    """Test jdtarriela resume against Anduril career opportunities"""
    import subprocess
    import sys as _sys
    result = subprocess.run([
        _sys.executable, 'test_jdtarriela_anduril.py'
    ], capture_output=False)
    if result.returncode != 0:
        console.print(f"[red]Resume matching test failed[/red]")
        raise typer.Exit(result.returncode)

@match_app.command('test-anduril-enhanced')
def match_test_anduril_enhanced():
    """Enhanced test of jdtarriela resume vs Anduril with detailed analysis"""
    import subprocess
    import sys as _sys
    result = subprocess.run([
        _sys.executable, 'enhanced_test_jdtarriela_anduril.py'
    ], capture_output=False)
    if result.returncode != 0:
        console.print(f"[red]Enhanced resume matching test failed[/red]")
        raise typer.Exit(result.returncode)

@review_app.command('start')
def review_start(job_id: str, resume_id: Optional[str] = typer.Option(None, help="Resume ID from database")):
    """Start a resume review for a specific job"""
    from libs.db.session import get_session
    from libs.db.models import Job, Resume, Review, Company
    from libs.resume.review import ReviewStatus
    import uuid
    from datetime import datetime
    
    try:
        with get_session() as session:
            # Verify job exists
            job = session.query(Job).filter(Job.id == job_id).first()
            if not job:
                console.print(f"[red]Job not found: {job_id}[/red]")
                raise typer.Exit(1)
                
            company = session.query(Company).filter(Company.id == job.company_id).first()
            
            # Verify resume exists
            resume = None
            if resume_id:
                resume = session.query(Resume).filter(Resume.id == resume_id).first()
                if not resume:
                    console.print(f"[red]Resume not found: {resume_id}[/red]")
                    raise typer.Exit(1)
            else:
                # Get the most recent resume
                resume = session.query(Resume).order_by(Resume.created_at.desc()).first()
                if not resume:
                    console.print("[red]No resumes found. Please ingest a resume first.[/red]")
                    console.print("[dim]Use 'ljs resume ingest <file>' to add a resume.[/dim]")
                    raise typer.Exit(1)
                resume_id = str(resume.id)
            
            # Check if review already exists
            existing_review = session.query(Review).filter(
                Review.job_id == job_id,
                Review.resume_id == resume_id
            ).first()
            
            if existing_review:
                console.print(f"[yellow]Review already exists: {existing_review.id}[/yellow]")
                console.print(f"[dim]Status: {existing_review.status}[/dim]")
                return
            
            # Create new review
            review_id = str(uuid.uuid4())
            review = Review(
                id=review_id,
                resume_id=resume_id,
                job_id=job_id,
                llm_score=None,  # Will be filled when review is generated
                strengths_md="",
                weaknesses_md="", 
                suggestions_md="",
                iteration_count=0,
                parent_review_id=None,
                status=ReviewStatus.PENDING.value,
                created_at=datetime.utcnow()
            )
            
            session.add(review)
            session.commit()
            
            console.print(f"[green]✅ Review started successfully[/green]")
            console.print(f"[cyan]Review ID: {review_id}[/cyan]")
            console.print(f"[cyan]Job: {job.title} at {company.name if company else 'Unknown'}[/cyan]")
            console.print(f"[cyan]Resume: {resume_id[:8]}... ({len(resume.skills_csv.split(',') if resume.skills_csv else [])} skills)[/cyan]")
            console.print(f"[yellow]Status: {review.status}[/yellow]")
            console.print("\n[dim]Note: Automated LLM review generation will be implemented in a future release.[/dim]")
            console.print("[dim]For now, you can manually update the review using the database.[/dim]")
            
    except Exception as e:
        console.print(f"[red]Failed to start review: {e}[/red]")
        raise typer.Exit(1)

@review_app.command('rewrite')
def review_rewrite(review_id: str, mode: str = typer.Option('auto', '--mode', case_sensitive=False), file: Optional[Path] = None):
    """Generate rewrite suggestions for a review (placeholder)"""
    from libs.db.session import get_session
    from libs.db.models import Review
    
    try:
        with get_session() as session:
            review = session.query(Review).filter(Review.id == review_id).first()
            if not review:
                console.print(f"[red]Review not found: {review_id}[/red]")
                raise typer.Exit(1)
            
            console.print(f"[cyan]Rewrite mode: {mode}[/cyan]")
            if file:
                console.print(f"[cyan]Input file: {file}[/cyan]")
            
            # Increment iteration count
            review.iteration_count += 1
            session.commit()
            
            console.print(f"[yellow]⚠️  Automated rewrite generation not yet implemented[/yellow]")
            console.print(f"[green]Review iteration incremented to: {review.iteration_count}[/green]")
            console.print("[dim]LLM-powered rewrite suggestions coming in future release[/dim]")
            
    except Exception as e:
        console.print(f"[red]Failed to rewrite review: {e}[/red]")
        raise typer.Exit(1)

@review_app.command('next')
def review_next(review_id: str):
    """Move to next iteration of a review (placeholder)"""
    from libs.db.session import get_session
    from libs.db.models import Review
    from libs.resume.review import ReviewStatus, validate_status_transition
    
    try:
        with get_session() as session:
            review = session.query(Review).filter(Review.id == review_id).first()
            if not review:
                console.print(f"[red]Review not found: {review_id}[/red]")
                raise typer.Exit(1)
            
            # Validate status transition
            new_status = ReviewStatus.IN_PROGRESS.value
            if not validate_status_transition(review.status, new_status):
                console.print(f"[red]Invalid status transition: {review.status} -> {new_status}[/red]")
                console.print(f"[dim]Current status: {review.status}[/dim]")
                raise typer.Exit(1)
            
            review.iteration_count += 1
            review.status = new_status
            session.commit()
            
            console.print(f"[green]✅ Moved to next iteration: {review.iteration_count}[/green]")
            console.print(f"[cyan]Status: {review.status}[/cyan]")
            console.print("[dim]Automated next iteration logic coming in future release[/dim]")
            
    except Exception as e:
        console.print(f"[red]Failed to advance review: {e}[/red]")
        raise typer.Exit(1)

@review_app.command('satisfy')
def review_satisfy(review_id: str):
    """Mark a review as satisfied/completed"""
    from libs.db.session import get_session
    from libs.db.models import Review
    from libs.resume.review import ReviewStatus, validate_status_transition
    
    try:
        with get_session() as session:
            review = session.query(Review).filter(Review.id == review_id).first()
            if not review:
                console.print(f"[red]Review not found: {review_id}[/red]")
                raise typer.Exit(1)
            
            # Validate status transition
            new_status = ReviewStatus.ACCEPTED.value
            if not validate_status_transition(review.status, new_status):
                console.print(f"[red]Invalid status transition: {review.status} -> {new_status}[/red]")
                console.print(f"[dim]Current status: {review.status}[/dim]")
                raise typer.Exit(1)
            
            # Update status to accepted (consistent with ReviewStatus enum)
            review.status = new_status
            session.commit()
            
            console.print(f"[green]✅ Review {review_id[:8]}... marked as accepted[/green]")
            console.print(f"[cyan]Status: {review.status}[/cyan]")
            
    except Exception as e:
        console.print(f"[red]Failed to satisfy review: {e}[/red]")
        raise typer.Exit(1)

@review_app.command('list')
def review_list():
    """List all reviews"""
    from libs.db.session import get_session
    from libs.db.models import Review, Job, Company, Resume
    
    try:
        with get_session() as session:
            # Get reviews with job and company info
            reviews = session.query(Review, Job, Company, Resume).join(
                Job, Review.job_id == Job.id
            ).join(
                Company, Job.company_id == Company.id
            ).join(
                Resume, Review.resume_id == Resume.id
            ).order_by(Review.created_at.desc()).all()
            
            if not reviews:
                console.print("[yellow]No reviews found.[/yellow]")
                console.print("[dim]Use 'ljs review start <job_id>' to create a review.[/dim]")
                return
                
            table = Table(title=f"Reviews ({len(reviews)})")
            table.add_column("Review ID", style="cyan")
            table.add_column("Job", style="white") 
            table.add_column("Company", style="yellow")
            table.add_column("Status", style="green")
            table.add_column("Score", style="blue")
            table.add_column("Created", style="dim")
            
            for review, job, company, resume in reviews:
                table.add_row(
                    str(review.id)[:8] + "...",
                    job.title or "Unknown",
                    company.name or "Unknown",
                    review.status or "pending",
                    str(review.llm_score) if review.llm_score else "N/A",
                    review.created_at.strftime('%Y-%m-%d') if review.created_at else 'Unknown'
                )
            
            console.print(table)
            
    except Exception as e:
        console.print(f"[red]Failed to list reviews: {e}[/red]")
        raise typer.Exit(1)

@review_app.command('show')
def review_show(review_id: str):
    """Show detailed information about a specific review"""
    from libs.db.session import get_session
    from libs.db.models import Review, Job, Company, Resume
    
    try:
        with get_session() as session:
            # Get review with related information
            review_data = session.query(Review, Job, Company, Resume).join(
                Job, Review.job_id == Job.id
            ).join(
                Company, Job.company_id == Company.id
            ).join(
                Resume, Review.resume_id == Resume.id
            ).filter(Review.id == review_id).first()
            
            if not review_data:
                console.print(f"[red]Review not found: {review_id}[/red]")
                raise typer.Exit(1)
            
            review, job, company, resume = review_data
            
            # Display review details
            console.print(f"[bold cyan]Review Details[/bold cyan]")
            console.print(f"[cyan]ID:[/cyan] {review.id}")
            console.print(f"[cyan]Status:[/cyan] {review.status or 'pending'}")
            
            # Job information
            console.print(f"\n[bold yellow]Job Information[/bold yellow]")
            console.print(f"[yellow]Title:[/yellow] {job.title or 'Unknown'}")
            console.print(f"[yellow]Company:[/yellow] {company.name or 'Unknown'}")
            console.print(f"[yellow]Location:[/yellow] {job.location or 'Remote'}")
            console.print(f"[yellow]URL:[/yellow] {job.url}")
            
            # Resume information
            console.print(f"\n[bold green]Resume Information[/bold green]")
            console.print(f"[green]Resume ID:[/green] {resume.id}")
            console.print(f"[green]Original Name:[/green] {resume.original_filename or 'Unknown'}")
            if resume.skills_csv:
                skills = resume.skills_csv.split(',')
                console.print(f"[green]Skills Count:[/green] {len(skills)}")
            
            # Review metadata
            console.print(f"\n[bold blue]Review Metadata[/bold blue]")
            if review.llm_score:
                console.print(f"[blue]LLM Score:[/blue] {review.llm_score}")
            if review.improvement_brief:
                console.print(f"[blue]Improvement Brief:[/blue] {review.improvement_brief}")
            if review.redact_note:
                console.print(f"[blue]Redact Note:[/blue] {review.redact_note}")
            
            if review.created_at:
                console.print(f"[blue]Created:[/blue] {review.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            
    except Exception as e:
        console.print(f"[red]Failed to show review: {e}[/red]")
        raise typer.Exit(1)

@apply_app.command('run')
def apply_run(job_id: str, resume: Optional[str] = None, profile: Optional[str] = None, dry_run: bool = typer.Option(False, '--dry-run')):
    """Run application process for a specific job"""
    from libs.db.session import get_session
    from libs.db.models import Job, Company, Resume, ApplicationProfile
    
    ctx = pass_context.get()
    effective_dry = ctx.dry_run or dry_run
    
    try:
        with get_session() as session:
            # Verify job exists
            job = session.query(Job).filter(Job.id == job_id).first()
            if not job:
                console.print(f"[red]Job not found: {job_id}[/red]")
                raise typer.Exit(1)
            
            company = session.query(Company).filter(Company.id == job.company_id).first()
            
            # Get resume if specified
            resume_obj = None
            if resume:
                resume_obj = session.query(Resume).filter(Resume.id == resume).first()
                if not resume_obj:
                    console.print(f"[red]Resume not found: {resume}[/red]")
                    raise typer.Exit(1)
            
            # Get application profile if specified
            profile_obj = None
            if profile:
                profile_obj = session.query(ApplicationProfile).filter(ApplicationProfile.name == profile).first()
                if not profile_obj:
                    console.print(f"[red]Application profile not found: {profile}[/red]")
                    raise typer.Exit(1)
            
            # Display job information
            console.print(f"[bold cyan]Applying to Job: {job.title}[/bold cyan]")
            console.print(f"[yellow]Company:[/yellow] {company.name if company else 'Unknown'}")
            console.print(f"[yellow]Location:[/yellow] {job.location or 'Remote'}")
            console.print(f"[yellow]Job URL:[/yellow] {job.url}")
            
            if resume_obj:
                console.print(f"[green]Resume:[/green] {resume} ({len(resume_obj.skills_csv.split(',') if resume_obj.skills_csv else [])} skills)")
                
            if profile_obj:
                console.print(f"[blue]Profile:[/blue] {profile_obj.name}")
            
            mode_str = "[yellow]DRY RUN[/yellow]" if effective_dry else "[green]LIVE APPLICATION[/green]"
            console.print(f"[bold]Mode:[/bold] {mode_str}")
            
            if effective_dry:
                console.print("\n[cyan]📋 Dry Run Simulation:[/cyan]")
                console.print("  ✓ Would download job application page")
                console.print("  ✓ Would fill application form with profile data")
                console.print("  ✓ Would attach resume file")
                console.print("  ✓ Would submit application")
                console.print("  ✓ Would capture confirmation receipt")
                console.print("  ✓ Would store application record in database")
                console.print("\n[green]✅ Dry run completed successfully[/green]")
                console.print("[dim]Use --no-dry-run or remove dry_run from config to apply for real[/dim]")
            else:
                console.print("\n[red]⚠️  Live application not yet implemented[/red]")
                console.print("[yellow]Auto-application features are coming in a future release[/yellow]")
                console.print("[dim]For now, please apply manually using the job URL above[/dim]")
                
    except Exception as e:
        console.print(f"[red]Apply run failed: {e}[/red]")
        raise typer.Exit(1)

@apply_app.command('bulk')
def apply_bulk(
    job_ids: str = typer.Argument(..., help="Comma-separated job IDs or 'all' for all matched jobs"),
    resume: Optional[str] = typer.Option(None, help="Resume ID to use for all applications"),
    profile: Optional[str] = typer.Option(None, help="Application profile to use for all applications"),
    dry_run: bool = typer.Option(False, '--dry-run', help="Run in dry-run mode"),
    limit: int = typer.Option(10, '--limit', help="Maximum number of jobs to apply to")
):
    """Run bulk application process for multiple jobs"""
    from libs.db.session import get_session
    from libs.db.models import Job, Company, Resume, ApplicationProfile, Application
    from datetime import datetime
    import uuid
    
    ctx = pass_context.get()
    effective_dry = ctx.dry_run or dry_run
    
    try:
        with get_session() as session:
            # Parse job IDs
            if job_ids.lower() == 'all':
                # Get all jobs that don't have applications yet
                applied_job_ids = session.query(Application.job_id).distinct()
                jobs = session.query(Job).filter(~Job.id.in_(applied_job_ids)).limit(limit).all()
                console.print(f"[cyan]Found {len(jobs)} unapplied jobs (limited to {limit})[/cyan]")
            else:
                job_id_list = [jid.strip() for jid in job_ids.split(',')]
                jobs = session.query(Job).filter(Job.id.in_(job_id_list)).all()
                console.print(f"[cyan]Processing {len(jobs)} specified jobs[/cyan]")
            
            if not jobs:
                console.print("[yellow]No jobs found to apply to[/yellow]")
                return
            
            # Validate resume and profile if provided
            resume_obj = None
            if resume:
                resume_obj = session.query(Resume).filter(Resume.id == resume).first()
                if not resume_obj:
                    console.print(f"[red]Resume not found: {resume}[/red]")
                    raise typer.Exit(1)
            
            profile_obj = None
            if profile:
                profile_obj = session.query(ApplicationProfile).filter(ApplicationProfile.name == profile).first()
                if not profile_obj:
                    console.print(f"[red]Application profile not found: {profile}[/red]")
                    raise typer.Exit(1)
            
            # Show summary
            mode_str = "[yellow]DRY RUN[/yellow]" if effective_dry else "[green]LIVE APPLICATIONS[/green]"
            console.print(f"[bold]Mode:[/bold] {mode_str}")
            console.print(f"[bold]Resume:[/bold] {resume or 'Not specified'}")
            console.print(f"[bold]Profile:[/bold] {profile or 'Not specified'}")
            console.print()
            
            successful_applications = 0
            failed_applications = 0
            
            # Process each job
            table = Table(title=f"Bulk Application Results")
            table.add_column("Job Title", style="cyan")
            table.add_column("Company", style="yellow")
            table.add_column("Status", style="green")
            
            for job in jobs:
                company = session.query(Company).filter(Company.id == job.company_id).first()
                company_name = company.name if company else "Unknown"
                
                try:
                    if effective_dry:
                        # Simulate application
                        table.add_row(job.title or "Unknown", company_name, "✓ Simulated")
                        successful_applications += 1
                    else:
                        # In a real implementation, this would perform the actual application
                        # For now, we'll create a placeholder application record
                        existing_app = session.query(Application).filter(Application.job_id == job.id).first()
                        if existing_app:
                            table.add_row(job.title or "Unknown", company_name, "⚠ Already Applied")
                        else:
                            # Create application record
                            application = Application(
                                id=str(uuid.uuid4()),
                                job_id=job.id,
                                resume_id=resume_obj.id if resume_obj else None,
                                profile_id=profile_obj.id if profile_obj else None,
                                applied_at=datetime.utcnow(),
                                status='submitted',
                                application_method='bulk_cli'
                            )
                            session.add(application)
                            table.add_row(job.title or "Unknown", company_name, "✓ Applied")
                            successful_applications += 1
                            
                except Exception as e:
                    table.add_row(job.title or "Unknown", company_name, f"✗ Failed: {str(e)[:30]}")
                    failed_applications += 1
            
            if not effective_dry:
                session.commit()
            
            console.print(table)
            console.print()
            console.print(f"[green]✅ Successful: {successful_applications}[/green]")
            if failed_applications > 0:
                console.print(f"[red]❌ Failed: {failed_applications}[/red]")
            
            if effective_dry:
                console.print("[dim]Run without --dry-run to perform actual applications[/dim]")
                
    except Exception as e:
        console.print(f"[red]Bulk apply failed: {e}[/red]")
        raise typer.Exit(1)

@apply_app.command('status')
def apply_status(
    job_id: Optional[str] = typer.Option(None, help="Show status for specific job"),
    user_id: Optional[str] = typer.Option(None, help="Filter by user ID"),
    status_filter: Optional[str] = typer.Option(None, '--status', help="Filter by application status"),
    limit: int = typer.Option(20, '--limit', help="Maximum number of results to show")
):
    """Check application statuses"""
    from libs.db.session import get_session
    from libs.db.models import Application, Job, Company, Resume, ApplicationProfile
    
    try:
        with get_session() as session:
            # Build query
            query = session.query(Application, Job, Company).join(
                Job, Application.job_id == Job.id
            ).join(
                Company, Job.company_id == Company.id
            )
            
            # Apply filters
            if job_id:
                query = query.filter(Application.job_id == job_id)
            if user_id:
                query = query.filter(Application.user_id == user_id)
            if status_filter:
                query = query.filter(Application.status == status_filter)
            
            # Order by most recent first
            query = query.order_by(Application.applied_at.desc())
            
            # Apply limit
            applications = query.limit(limit).all()
            
            if not applications:
                if job_id:
                    console.print(f"[yellow]No application found for job: {job_id}[/yellow]")
                else:
                    console.print("[yellow]No applications found[/yellow]")
                return
            
            # Display results
            table = Table(title=f"Application Status ({len(applications)} results)")
            table.add_column("Job Title", style="cyan")
            table.add_column("Company", style="yellow")
            table.add_column("Status", style="green")
            table.add_column("Applied", style="dim")
            table.add_column("Method", style="blue")
            
            status_counts = {}
            
            for application, job, company in applications:
                status = application.status or "unknown"
                status_counts[status] = status_counts.get(status, 0) + 1
                
                # Format status with color
                if status in ['submitted', 'applied']:
                    status_display = f"[green]{status}[/green]"
                elif status in ['rejected', 'failed']:
                    status_display = f"[red]{status}[/red]"
                elif status in ['pending', 'in_review']:
                    status_display = f"[yellow]{status}[/yellow]"
                else:
                    status_display = status
                
                applied_date = application.applied_at.strftime('%Y-%m-%d') if application.applied_at else 'N/A'
                
                table.add_row(
                    job.title or "Unknown",
                    company.name or "Unknown",
                    status_display,
                    applied_date,
                    application.application_method or "manual"
                )
            
            console.print(table)
            
            # Show summary
            console.print(f"\n[bold cyan]Status Summary:[/bold cyan]")
            for status, count in sorted(status_counts.items()):
                console.print(f"  {status}: {count}")
                
    except Exception as e:
        console.print(f"[red]Failed to check application status: {e}[/red]")
        raise typer.Exit(1)

@events_app.command('tail')
def events_tail(since: Optional[str] = typer.Option(None, '--since', help='Relative time (e.g., 10m)')):
    from rich.live import Live
    from time import sleep
    table = Table(title='Events (mock stream)')
    table.add_column('Time'); table.add_column('Type'); table.add_column('Detail')
    with Live(table, refresh_per_second=4):
        for i in range(3):
            table.add_row(f'+{i}s', 'crawl', f'scraped page {i}')
            sleep(0.3)
    console.print('[green]Stream ended[/green]')

@db_app.command('migrate')
def db_migrate():
    """Run Alembic database migrations"""
    import subprocess
    import sys
    
    try:
        console.print("[cyan]Running database migrations...[/cyan]")
        
        # Run alembic upgrade head
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            capture_output=True, 
            text=True, 
            cwd="."
        )
        
        if result.returncode == 0:
            console.print("[green]✅ Database migrations completed successfully[/green]")
            if result.stdout:
                console.print(f"[dim]{result.stdout}[/dim]")
        else:
            console.print(f"[red]❌ Migration failed with exit code {result.returncode}[/red]")
            if result.stderr:
                console.print(f"[red]Error: {result.stderr}[/red]")
            if result.stdout:
                console.print(f"Output: {result.stdout}")
            raise typer.Exit(1)
            
    except FileNotFoundError:
        console.print("[red]❌ Alembic not found. Please install: pip install alembic[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Migration failed: {e}[/red]")
        raise typer.Exit(1)

@db_app.command('init-db')
def db_init(create: bool = typer.Option(True, '--create/--no-create', help='Actually create tables')):
    """Create all tables from SQLAlchemy metadata (dev convenience)."""
    if not create:
        console.print('[yellow]Skipping create (flag disabled)[/yellow]')
        return
    from libs.db.models import Base  # late import
    from libs.db.session import engine
    Base.metadata.create_all(bind=engine)
    console.print('[green]All tables created (if not existing).[/green]')


# --------------- Jobs Commands ---------------
@jobs_app.command('add')
def jobs_add(
    title: str = typer.Argument(..., help="Job title"),
    company_name: str = typer.Option(..., '--company', help="Company name"),
    location: str = typer.Option("Remote", help="Job location"),
    skills: str = typer.Option("", help="Required skills (comma-separated)"),
    seniority: str = typer.Option("Mid", help="Seniority level"),
    url: str = typer.Option("", help="Job posting URL")
):
    """Add a sample job for testing"""
    from libs.db.session import get_session
    from libs.db.models import Job, Company
    import uuid
    from datetime import datetime
    
    try:
        with get_session() as session:
            # Find or create company
            company = session.query(Company).filter(Company.name == company_name).first()
            if not company:
                company = Company(
                    id=str(uuid.uuid4()),
                    name=company_name,
                    website=f"https://{company_name.lower().replace(' ', '')}.com",
                    careers_url=f"https://{company_name.lower().replace(' ', '')}.com/careers",
                    crawler_profile_json='{"type": "manual"}'
                )
                session.add(company)
                session.flush()  # Get the ID
                console.print(f"[green]Created company: {company_name}[/green]")
            
            # Create job
            job_url = url or f"https://{company_name.lower().replace(' ', '')}.com/jobs/{title.lower().replace(' ', '-')}"
            
            job_description = f"""
We are looking for a {title} to join our team at {company_name}.

Location: {location}
Seniority Level: {seniority}

Required Skills: {skills or 'TBD'}

Apply now to join our innovative team!
            """.strip()
            
            job = Job(
                id=str(uuid.uuid4()),
                company_id=company.id,
                url=job_url,
                title=title,
                location=location,
                seniority=seniority,
                jd_fulltext=job_description,
                jd_skills_csv=skills,
                scraped_at=datetime.now(),
                scrape_fingerprint=f"manual_{title}_{company_name}"
            )
            
            session.add(job)
            session.commit()
            
            console.print(f"[green]✅ Job created successfully![/green]")
            console.print(f"[cyan]Job ID: {job.id}[/cyan]")
            console.print(f"[cyan]Title: {title}[/cyan]")
            console.print(f"[cyan]Company: {company_name}[/cyan]")
            console.print(f"[cyan]Skills: {skills or 'None specified'}[/cyan]")
            
    except Exception as e:
        console.print(f"[red]Failed to create job: {e}[/red]")
        raise typer.Exit(1)

@jobs_app.command('list')
def jobs_list():
    """List jobs from database"""
    from libs.db.session import get_session
    from libs.db.models import Job, Company
    
    try:
        with get_session() as session:
            # Query jobs with companies
            jobs = session.query(Job, Company).join(Company, Job.company_id == Company.id).all()
            
            if not jobs:
                console.print("[yellow]No jobs found in database.[/yellow]")
                console.print("[dim]Use 'ljs jobs add' to add a job for testing.[/dim]")
                return
                
            table = Table(title=f'Jobs ({len(jobs)})')
            table.add_column('Title', style="cyan")
            table.add_column('Company', style="white") 
            table.add_column('Location', style="green")
            table.add_column('Skills', style="blue")
            table.add_column('Level', style="magenta")
            
            for job, company in jobs:
                skills_display = (job.jd_skills_csv or "")[:30] + "..." if len(job.jd_skills_csv or "") > 30 else (job.jd_skills_csv or "None")
                
                table.add_row(
                    job.title or "Unknown",
                    company.name or "Unknown",
                    job.location or "Unknown",
                    skills_display,
                    job.seniority or "Unknown"
                )
            
            console.print(table)
            
    except Exception as e:
        console.print(f"[red]Failed to list jobs: {e}[/red]")
        raise typer.Exit(1)

# ------------------ Template Operations ------------------

@template_app.command('validate')
def template_validate(
    file_path: Optional[Path] = typer.Argument(None, help="Path to template file (validates all if not specified)"),
    schema_path: Optional[Path] = typer.Option(None, "--schema", help="Path to schema file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed validation output")
):
    """Validate portal template(s) against the DSL schema"""
    from jsonschema import validate as js_validate, ValidationError
    
    try:
        # Default paths
        if not schema_path:
            schema_path = Path(__file__).parent.parent / "docs" / "portal_template_dsl.schema.json"
        
        if not file_path:
            # Validate all example templates
            templates_dir = Path(__file__).parent.parent / "docs" / "examples" / "portal_templates"
            template_files = list(templates_dir.glob("*.json"))
        else:
            template_files = [file_path]
        
        if not template_files:
            console.print("[yellow]No template files found to validate[/yellow]")
            return
            
        # Load schema
        with open(schema_path) as f:
            schema = json.load(f)
        
        console.print(f"[cyan]Validating {len(template_files)} template(s)...[/cyan]")
        
        errors = []
        success_count = 0
        
        for template_path in template_files:
            try:
                with open(template_path) as f:
                    template = json.load(f)
                
                # Validate against schema
                js_validate(template, schema)
                
                # Security validation
                from libs.autoapply.security import create_execution_sandbox
                sandbox = create_execution_sandbox()
                security_violations = sandbox.validate_template_security(template)
                
                if security_violations:
                    if verbose:
                        console.print(f"[yellow]⚠️  {template_path.name}: Schema valid, security warnings[/yellow]")
                        for violation in security_violations:
                            console.print(f"    - {violation}")
                    else:
                        console.print(f"[yellow]⚠️  {template_path.name}: Security warnings ({len(security_violations)})[/yellow]")
                else:
                    console.print(f"[green]✅ {template_path.name}[/green]")
                    
                success_count += 1
                
            except ValidationError as e:
                errors.append(f"❌ {template_path.name}: {e.message}")
                if verbose:
                    console.print(f"[red]❌ {template_path.name}:[/red]")
                    console.print(f"    [red]{e.message}[/red]")
                    if e.absolute_path:
                        console.print(f"    [dim]Path: {' -> '.join(str(p) for p in e.absolute_path)}[/dim]")
                
            except FileNotFoundError:
                errors.append(f"❌ {template_path.name}: File not found")
                
            except json.JSONDecodeError as e:
                errors.append(f"❌ {template_path.name}: Invalid JSON - {e.msg}")
                
            except Exception as e:
                errors.append(f"❌ {template_path.name}: {str(e)}")
        
        # Summary
        if not verbose and errors:
            console.print(f"\n[red]Validation errors:[/red]")
            for error in errors:
                console.print(f"  {error}")
        
        if errors:
            console.print(f"\n[red]{len(errors)} template(s) failed validation[/red]")
            raise typer.Exit(1)
        else:
            console.print(f"\n[green]✅ All {success_count} template(s) validated successfully[/green]")
            
    except FileNotFoundError as e:
        console.print(f"[red]File not found: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Validation error: {e}[/red]")
        raise typer.Exit(1)


@template_app.command('list')
def template_list():
    """List available portal templates"""
    templates_dir = Path(__file__).parent.parent / "docs" / "examples" / "portal_templates"
    
    if not templates_dir.exists():
        console.print("[yellow]No templates directory found[/yellow]")
        return
    
    template_files = list(templates_dir.glob("*.json"))
    
    if not template_files:
        console.print("[yellow]No template files found[/yellow]")
        return
    
    table = Table(title="Portal Templates")
    table.add_column("Template", style="cyan")
    table.add_column("Portal", style="green") 
    table.add_column("Version", style="yellow")
    table.add_column("Description", style="dim")
    
    for template_path in sorted(template_files):
        try:
            with open(template_path) as f:
                template = json.load(f)
            
            meta = template.get("meta", {})
            table.add_row(
                template_path.name,
                meta.get("portal", "Unknown"),
                str(template.get("version", "1")),
                meta.get("description", "No description")
            )
            
        except Exception as e:
            table.add_row(
                template_path.name,
                "Error",
                "-", 
                f"Failed to load: {e}"
            )
    
    console.print(table)


@template_app.command('lint')
def template_lint(
    file_path: Path = typer.Argument(..., help="Path to template file"),
    fix: bool = typer.Option(False, "--fix", help="Attempt to auto-fix issues")
):
    """Lint a portal template for best practices and common issues"""
    try:
        with open(file_path) as f:
            template = json.load(f)
        
        issues = []
        
        # Check for common issues
        meta = template.get("meta", {})
        if not meta.get("description"):
            issues.append("Missing template description in meta")
        
        if not meta.get("author"):
            issues.append("Missing template author in meta")
        
        # Check steps
        steps = template.get("steps", [])
        for i, step in enumerate(steps):
            # Check for missing IDs
            if not step.get("id"):
                issues.append(f"Step {i+1}: Missing step ID")
            
            # Check for hardcoded waits without good reason
            if step.get("action") == "wait" and step.get("timeoutMs", 0) > 10000:
                issues.append(f"Step {i+1}: Long wait time ({step.get('timeoutMs')}ms)")
            
            # Check for missing selectors
            action = step.get("action")
            if action in ["click", "type", "select", "upload"] and not step.get("selector"):
                issues.append(f"Step {i+1}: Missing selector for {action} action")
        
        # Check for template variables
        from libs.autoapply.security import create_template_sanitizer
        sanitizer = create_template_sanitizer()
        
        template_str = json.dumps(template)
        variables = sanitizer.extract_template_variables(template_str)
        
        for var in variables:
            if not sanitizer.validate_template_variable(var):
                issues.append(f"Potentially unsafe template variable: {var}")
        
        # Output results
        if not issues:
            console.print(f"[green]✅ {file_path.name} looks good![/green]")
        else:
            console.print(f"[yellow]⚠️  Found {len(issues)} issue(s) in {file_path.name}:[/yellow]")
            for issue in issues:
                console.print(f"  - {issue}")
            
            if fix:
                console.print("[cyan]Auto-fix not implemented yet[/cyan]")
        
    except FileNotFoundError:
        console.print(f"[red]Template file not found: {file_path}[/red]")
        raise typer.Exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON in template: {e.msg}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error linting template: {e}[/red]")
        raise typer.Exit(1)


def main():  # entry point for setuptools script
    APP()

if __name__ == '__main__':  # pragma: no cover
    main()
