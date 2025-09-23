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

# pass_context = typer.ContextVar("ljs_ctx")  # Deprecated in newer typer versions


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
    # Store context in typer context
    ctx.obj = c


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
    config = load_config(None)
    print_config(config)


@config_app.command('validate')
def config_validate():
    config = load_config(None)
    try:
        validate_config(config)
        console.print("[green]Config is valid[/green]")
    except ValidationError as e:
        console.print(f"[red]Config invalid:[/red] {e.message}")
        raise typer.Exit(1)
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
    # Simple stub for MVP - just print the action
    console.print(f"[cyan][STUB] {name}[/cyan] {kwargs}")


@resume_app.command('ingest')
def resume_ingest(file: Path):
    _stub('resume.ingest', file=str(file))

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
def companies_seed(file: Path = typer.Option(..., '--file', help='Path to file containing company names')):
    if not file.exists():
        console.print(f"[red]Seed file not found: {file}[/red]")
        raise typer.Exit(1)
    names = [l.strip() for l in file.read_text().splitlines() if l.strip()]
    if not names:
        console.print('[yellow]No company names found in seed file[/yellow]')
        return
    console.print(f"[cyan][DRY] Seeding {len(names)} companies[/cyan]")
    for n in names[:5]:
        console.print(f"[cyan]DRY[/cyan] would insert: {n}")
        # TODO: Remove DRY mode and implement actual database insertion

@companies_app.command('list')
def companies_list():
    with get_session() as session:
        rows = session.query(models.Company).order_by(models.Company.name).limit(50).all()
        table = Table(title=f'Companies ({len(rows)})')
        table.add_column('ID'); table.add_column('Name'); table.add_column('Careers URL')
        for r in rows:
            table.add_row(str(r.id), r.name or '', r.careers_url or '')
        console.print(table)

@crawl_app.command('run')
def crawl_run(company: Optional[str] = None, all: bool = typer.Option(False, '--all')):  # noqa: A002
    _stub('crawl.run', company=company, all=all)

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
    _stub('apply.run', job_id=job_id, resume=resume, profile=profile, dry_run=dry_run)

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
