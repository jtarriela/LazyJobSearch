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

APP = typer.Typer(add_completion=False, help="LazyJobSearch CLI")
console = Console()

# Sub-apps
config_app = typer.Typer(help="Config management")
resume_app = typer.Typer(help="Resume operations")
companies_app = typer.Typer(help="Company seeding & listing")
crawl_app = typer.Typer(help="Crawl control")
match_app = typer.Typer(help="Matching operations")
review_app = typer.Typer(help="Review / rewrite loop")
apply_app = typer.Typer(help="Auto-apply operations")
events_app = typer.Typer(help="Event stream")
schema_app = typer.Typer(help="Schema & documentation validation")
db_app = typer.Typer(help="Database migrations & maintenance")
generate_app = typer.Typer(help="Generators & scaffolding")

APP.add_typer(config_app, name="config")
APP.add_typer(resume_app, name="resume")
APP.add_typer(companies_app, name="companies")
APP.add_typer(crawl_app, name="crawl")
APP.add_typer(match_app, name="match")
APP.add_typer(review_app, name="review")
APP.add_typer(apply_app, name="apply")
APP.add_typer(events_app, name="events")
APP.add_typer(schema_app, name="schema")
APP.add_typer(db_app, name="db")
APP.add_typer(generate_app, name="generate")


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


# --------------- Placeholder Domain Commands ---------------
def _stub(name: str, **kwargs):
    ctx = pass_context.get()
    mode = '[DRY]' if ctx.dry_run else '[LIVE]'
    console.print(f"[cyan]{mode} {name}[/cyan] {kwargs}")


@resume_app.command('parse')
def resume_parse(
    file: Path,
    output_format: str = typer.Option("json", help="Output format (json|yaml)")
):
    """Parse a resume file and extract structured data"""
    from libs.resume.parser import create_resume_parser
    
    if not file.exists():
        console.print(f"[red]Resume file not found: {file}[/red]")
        raise typer.Exit(1)
    
    try:
        parser = create_resume_parser()
        result = parser.parse_file(file)
        
        console.print(f"[green]Successfully parsed resume: {file}[/green]")
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
    max_tokens: int = typer.Option(500, help="Maximum tokens per chunk")
):
    """Chunk a resume for embedding and search"""
    from libs.resume.parser import create_resume_parser
    from libs.resume.chunker import create_resume_chunker, ChunkingConfig, ChunkStrategy
    
    if not file.exists():
        console.print(f"[red]Resume file not found: {file}[/red]")
        raise typer.Exit(1)
    
    try:
        # Parse resume first
        parser = create_resume_parser()
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
    resume_file: Optional[Path] = typer.Option(None, help="Resume file to match"),
    resume_id: Optional[str] = typer.Option(None, help="Resume ID from database"),
    limit: int = typer.Option(20, help="Maximum matches to return")
):
    """Run matching pipeline for a resume"""
    import asyncio
    from libs.resume.parser import create_resume_parser
    from libs.resume.embedding_service import create_embedding_service, EmbeddingProvider
    from libs.matching.pipeline import create_matching_pipeline, ResumeProfile, MatchingConfig
    
    # Validate inputs
    if not resume_file and not resume_id:
        console.print("[red]Must provide either --resume-file or --resume-id[/red]")
        raise typer.Exit(1)
    
    async def run_matching():
        try:
            # Create services
            session = get_session()
            embedding_service = create_embedding_service(provider=EmbeddingProvider.MOCK)
            config = MatchingConfig(llm_limit=limit)
            pipeline = create_matching_pipeline(session, embedding_service, config)
            
            # Prepare resume profile
            if resume_file:
                if not resume_file.exists():
                    console.print(f"[red]Resume file not found: {resume_file}[/red]")
                    return
                
                # Parse resume file
                parser = create_resume_parser()
                parsed_resume = parser.parse_file(resume_file)
                
                profile = ResumeProfile(
                    resume_id=f"file_{resume_file.stem}",
                    fulltext=parsed_resume.fulltext,
                    skills=parsed_resume.skills,
                    years_experience=parsed_resume.years_of_experience,
                    education_level=parsed_resume.education_level
                )
            else:
                # TODO: Load from database
                console.print("[yellow]Database resume loading not implemented yet[/yellow]")
                return
            
            console.print(f"[cyan]Running matching pipeline for resume...[/cyan]")
            
            # Run matching
            result = await pipeline.match_resume_to_jobs(profile)
            
            # Display results
            console.print(f"[green]Matching completed in {result.processing_time_seconds:.2f}s[/green]")
            console.print(f"[cyan]Cost: ${result.total_cost_cents/100:.3f}[/cyan]")
            console.print(f"[cyan]Stages completed: {[s.value for s in result.stages_completed]}[/cyan]")
            
            if result.matches:
                table = Table(title=f"Top {len(result.matches)} Matches")
                table.add_column("Rank")
                table.add_column("Job Title")
                table.add_column("Company")
                table.add_column("LLM Score")
                table.add_column("Action")
                table.add_column("Reasoning")
                
                for i, match in enumerate(result.matches[:10], 1):
                    score_color = "green" if (match.llm_score or 0) >= 80 else "yellow" if (match.llm_score or 0) >= 60 else "red"
                    action_color = "green" if "HIGH" in (match.action or "") else "yellow" if "MEDIUM" in (match.action or "") else "white"
                    
                    table.add_row(
                        str(i),
                        match.title,
                        match.company,
                        f"[{score_color}]{match.llm_score or 0}[/{score_color}]",
                        f"[{action_color}]{match.action or 'N/A'}[/{action_color}]",
                        (match.llm_reasoning or "")[:60] + "..." if len(match.llm_reasoning or "") > 60 else (match.llm_reasoning or "")
                    )
                
                console.print(table)
            else:
                console.print("[yellow]No matches found[/yellow]")
            
        except Exception as e:
            console.print(f"[red]Matching failed: {e}[/red]")
            raise typer.Exit(1)
    
    asyncio.run(run_matching())

@review_app.command('start')
def review_start(
    resume_file: Optional[Path] = typer.Option(None, help="Resume file to review"),
    job_title: str = typer.Option("Software Engineer", help="Target job title"),
    company: str = typer.Option("TechCorp", help="Target company"),
    max_iterations: int = typer.Option(3, help="Maximum review iterations")
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
            parser = create_resume_parser()
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

@resume_app.command('ingest')
def resume_ingest(file: Path):
    """Ingest a resume file into the system"""
    from libs.resume.ingestion import create_resume_ingestion_service
    from libs.resume.embedding_service import EmbeddingProvider
    from libs.embed.versioning import EmbeddingVersionManager
    from libs.db.session import get_session
    import asyncio
    
    if not file.exists():
        console.print(f"[red]Resume file not found: {file}[/red]")
        raise typer.Exit(1)
    
    async def ingest_resume():
        try:
            # Get database session
            session = get_session()
            
            # Initialize services
            embedding_version_manager = EmbeddingVersionManager(session)
            ingestion_service = create_resume_ingestion_service(
                db_session=session,
                embedding_version_manager=embedding_version_manager
            )
            
            console.print(f"[cyan]Starting resume ingestion: {file}[/cyan]")
            
            # Run complete ingestion pipeline
            result = await ingestion_service.ingest_resume_file(
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
    
    asyncio.run(ingest_resume())

@resume_app.command('list')
def resume_list():
    table = Table(title='Resumes (mock)')
    table.add_column('ID'); table.add_column('Version'); table.add_column('Active')
    table.add_row('r1', '1', 'True')
    console.print(table)

@resume_app.command('activate')
def resume_activate(resume_id: str):
    _stub('resume.activate', resume_id=resume_id)

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

@match_app.command('run')
def match_run(resume: Optional[str] = None, limit: int = 200):
    _stub('match.run', resume=resume, limit=limit)

@match_app.command('top')
def match_top(resume: Optional[str] = None, limit: int = 20, json_out: bool = typer.Option(False, '--json')):
    data = [
        {'job_id': 'j1', 'score': 87, 'title': 'ML Engineer'},
        {'job_id': 'j2', 'score': 82, 'title': 'Data Engineer'}
    ][:limit]
    if json_out:
        console.print_json(data=data)
    else:
        table = Table(title='Top Matches (mock)')
        table.add_column('Job ID'); table.add_column('Score'); table.add_column('Title')
        for row in data:
            table.add_row(row['job_id'], str(row['score']), row['title'])
        console.print(table)

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
def review_start(job_id: str, resume: Optional[str] = None):
    _stub('review.start', job_id=job_id, resume=resume)

@review_app.command('rewrite')
def review_rewrite(review_id: str, mode: str = typer.Option('auto', '--mode', case_sensitive=False), file: Optional[Path] = None):
    _stub('review.rewrite', review_id=review_id, mode=mode, file=str(file) if file else None)

@review_app.command('next')
def review_next(review_id: str):
    _stub('review.next', review_id=review_id)

@review_app.command('satisfy')
def review_satisfy(review_id: str):
    _stub('review.satisfy', review_id=review_id)

@apply_app.command('run')
def apply_run(job_id: str, resume: Optional[str] = None, profile: Optional[str] = None, dry_run: bool = typer.Option(False, '--dry-run')):
    ctx = pass_context.get()
    effective_dry = ctx.dry_run or dry_run
    _stub('apply.run', job_id=job_id, resume=resume, profile=profile, dry_run=effective_dry)

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
    _stub('db.migrate')

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


def main():  # entry point for setuptools script
    APP()

if __name__ == '__main__':  # pragma: no cover
    main()
