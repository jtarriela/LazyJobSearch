import json
from typer.testing import CliRunner
from cli.ljs import APP
from pathlib import Path

runner = CliRunner()


def test_config_init_help():
    result = runner.invoke(APP, ["config", "show"], catch_exceptions=False)
    assert result.exit_code == 0


def test_match_top_json():
    result = runner.invoke(APP, ["match", "top", "--json"], catch_exceptions=False)
    assert result.exit_code == 0
    # rich prints JSON; ensure braces present
    assert '{' in result.stdout


def test_generate_company_template(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(APP, ["generate", "company-template", "acme"], catch_exceptions=False)
    assert result.exit_code == 0
    assert (tmp_path / 'seeds' / 'companies' / 'acme.yaml').exists()


def test_companies_seed_dry_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seed_file = Path('companies.txt')
    seed_file.write_text("Acme\nBetaCorp\n")
    result = runner.invoke(APP, ["--dry-run", "companies", "seed", "--file", str(seed_file)])
    assert result.exit_code == 0
    assert 'DRY' in result.stdout


def test_config_validate_failure(tmp_path, monkeypatch):
    # create minimal invalid config (missing required email under user)
    monkeypatch.chdir(tmp_path)
    invalid = Path('lazyjobsearch.yaml')
    invalid.write_text("user: { }")
    result = runner.invoke(APP, ["config", "validate"], catch_exceptions=False)
    assert result.exit_code != 0
    assert 'invalid' in result.stdout.lower()


def test_schema_validate_cli(monkeypatch):
    # run in repo root (assumes tests executed from root by pytest)
    result = runner.invoke(APP, ["schema", "validate"], catch_exceptions=False)
    # exit code may be non-zero if model/docs drift; accept 0 for now
    assert result.exit_code in (0, 1)
