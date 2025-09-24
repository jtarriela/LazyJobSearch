#!/usr/bin/env bash
set -euo pipefail

echo "Listing companies (expect 0 on fresh DB):"
ljs companies list || true

echo "Seeding sample companies..."
printf "Anduril\nPalantir\nOpenAI\n" > seeds/companies.txt
ljs companies seed --file seeds/companies.txt
ljs companies list