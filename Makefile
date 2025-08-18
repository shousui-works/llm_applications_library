# Makefile for dudeligence project
.PHONY: help lint format lint-fix format-check check fix clean install

# Default target
help:
	@echo "Dudeligence Project Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  lint         - Run ruff linting"
	@echo "  format       - Fix code formatting with ruff"
	@echo "  format-check - Check code formatting with ruff"
	@echo "  lint-fix     - Fix linting issues with ruff"
	@echo "  check        - Run all checks (lint + format-check)"
	@echo "  fix          - Fix all issues (format + lint-fix)"
	@echo "  clean        - Clean cache files"
	@echo "  install      - Install dependencies with uv"
	@echo "  help         - Show this help message"

# Linting
lint:
	@echo "Running ruff linting..."
	uv run ruff check .

# Formatting
format:
	@echo "Fixing code formatting..."
	uv run ruff format .

format-check:
	@echo "Checking code formatting..."
	uv run ruff format --check .

# Fix linting issues
lint-fix:
	@echo "Fixing linting issues..."
	uv run ruff check --fix .

# Combined targets
check: lint format-check
	@echo "✅ All checks completed"

fix: format lint-fix
	@echo "✅ All fixes applied"

# Clean cache files
clean:
	@echo "Cleaning cache files..."
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# Install dependencies
install:
	@echo "Installing dependencies..."
	uv sync