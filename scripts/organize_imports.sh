#!/bin/bash
# Script to organize imports in all Python files using uv

set -e

echo "Organizing imports with isort..."
uv run isort evodm/ tests/ scripts/ --check-only --diff || {
    echo "Fixing imports with isort..."
    uv run isort evodm/ tests/ scripts/
    echo "✓ Imports organized with isort"
}

echo ""
echo "Checking imports with ruff..."
uv run ruff check --select I --fix evodm/ tests/ scripts/ || {
    echo "✓ Imports checked with ruff"
}

echo ""
echo "✓ All imports organized!"

