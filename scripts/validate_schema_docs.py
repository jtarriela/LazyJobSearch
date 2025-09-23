#!/usr/bin/env python
"""Validate markdown schema docs against SQLAlchemy model metadata.

Checks:
1. Each documented table exists in SQLAlchemy metadata.
2. Each column in markdown appears in model.
3. Warn if model columns not documented.

Markdown format expectation: Table header '| Column | Type | Notes |' after a '# TABLE_NAME Table' H1.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Import models
sys.path.append(str(Path(__file__).resolve().parents[1]))  # repo root
from libs.db import models  # type: ignore
from sqlalchemy import inspect

SCHEMA_DIR = Path('docs/schema')

TABLE_NAME_PATTERN = re.compile(r'^#\s+([A-Z_]+) Table')
COLUMN_ROW_PATTERN = re.compile(r'^\|\s*([a-z0-9_]+)\s*\|')

IGNORED_MD_COLUMNS = {"indexes", "constraints"}


def parse_markdown_table_columns(md_path: Path) -> Tuple[str, List[str]]:
    table_name = None
    columns: List[str] = []
    with md_path.open('r', encoding='utf-8') as f:
        for line in f:
            m = TABLE_NAME_PATTERN.match(line.strip())
            if m:
                table_name = m.group(1).lower()
            if line.startswith('|') and not line.startswith('| Column') and not line.startswith('|--------'):
                m2 = COLUMN_ROW_PATTERN.match(line)
                if m2:
                    col = m2.group(1)
                    if col not in ("id",) and col not in IGNORED_MD_COLUMNS:
                        columns.append(col)
    if not table_name:
        raise ValueError(f"Could not find table name header in {md_path}")
    return table_name, columns


def collect_doc_schema() -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for md in SCHEMA_DIR.glob('*.md'):
        tbl, cols = parse_markdown_table_columns(md)
        mapping[tbl] = cols
    return mapping


def collect_model_schema() -> Dict[str, List[str]]:
    insp = inspect(models.Base.metadata)
    mapping: Dict[str, List[str]] = {}
    for table_name, table in models.Base.metadata.tables.items():
        cols = [c.name for c in table.columns if c.name != 'id']
        mapping[table_name] = cols
    return mapping


def main() -> int:
    doc = collect_doc_schema()
    model = collect_model_schema()
    exit_code = 0

    # Check documented tables exist
    for table in sorted(doc.keys()):
        if table not in model:
            print(f"ERROR: Documented table '{table}' missing in models")
            exit_code = 1

    # Check model tables documented
    for table in sorted(model.keys()):
        if table not in doc:
            print(f"WARNING: Model table '{table}' has no markdown doc")

    # Column checks
    for table, doc_cols in doc.items():
        model_cols = set(model.get(table, []))
        for col in doc_cols:
            if col not in model_cols:
                print(f"ERROR: Table '{table}' documented column '{col}' not in model")
                exit_code = 1
        for col in sorted(model_cols):
            if col not in doc_cols:
                print(f"WARNING: Table '{table}' model has undocumented column '{col}'")

    if exit_code == 0:
        print("Schema docs validation PASSED")
    else:
        print("Schema docs validation FAILED")
    return exit_code

if __name__ == '__main__':
    raise SystemExit(main())
