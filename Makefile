.PHONY: help install dev-install format lint test clean run docker-build docker-run docker-stop migrate seed

# Variables
PYTHON := python3.11
VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python
UVICORN := $(VENV)/bin/uvicorn
PYTEST := $(VENV)/bin/pytest
BLACK := $(VENV)/bin/black
RUFF := $(VENV)/bin/ruff
ISORT := $(VENV)/bin/isort
ALEMBIC := $(VENV)/bin/alembic

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)Face Recognition Service - Available Commands$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

$(VENV): ## Create virtual environment
	@echo "$(GREEN)Creating virtual environment...$(NC)"
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel

install: $(VENV) ## Install production dependencies
	@echo "$(GREEN)Installing production dependencies...$(NC)"
	$(PIP) install -e .

dev-install: $(VENV) ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	$(PIP) install -e ".[dev]"
	$(VENV)/bin/pre-commit install

gpu-install: $(VENV) ## Install GPU dependencies
	@echo "$(GREEN)Installing GPU dependencies...$(NC)"
	$(PIP) install -e ".[gpu]"

format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code...$(NC)"
	$(BLACK) .
	$(ISORT) .

lint: ## Run linting with ruff
	@echo "$(GREEN)Running linter...$(NC)"
	$(RUFF) check .
	$(BLACK) --check .
	$(ISORT) --check-only .

test: ## Run tests with pytest
	@echo "$(GREEN)Running tests...$(NC)"
	$(PYTEST) tests/ -v --cov=api --cov=services --cov=core --cov-report=term-missing

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(NC)"
	$(PYTEST) tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "$(GREEN)Running integration tests...$(NC)"
	$(PYTEST) tests/integration/ -v

clean: ## Clean up generated files
	@echo "$(RED)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf build dist *.egg-info

run: ## Run the application locally
	@echo "$(GREEN)Starting application...$(NC)"
	$(UVICORN) api.main:app --reload --host 0.0.0.0 --port 8000

run-prod: ## Run the application in production mode
	@echo "$(GREEN)Starting application in production mode...$(NC)"
	$(UVICORN) api.main:app --host 0.0.0.0 --port 8000 --workers 4

migrate: ## Run database migrations
	@echo "$(GREEN)Running database migrations...$(NC)"
	$(ALEMBIC) upgrade head

migrate-create: ## Create a new migration
	@echo "$(GREEN)Creating new migration...$(NC)"
	@read -p "Enter migration message: " msg; \
	$(ALEMBIC) revision --autogenerate -m "$$msg"

migrate-down: ## Rollback last migration
	@echo "$(YELLOW)Rolling back last migration...$(NC)"
	$(ALEMBIC) downgrade -1

seed: ## Seed database with sample data
	@echo "$(GREEN)Seeding database...$(NC)"
	$(PYTHON_VENV) scripts/seed_data.py

reindex: ## Rebuild the vector index
	@echo "$(GREEN)Rebuilding vector index...$(NC)"
	$(PYTHON_VENV) scripts/reindex.py

docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build -t face-recognition:latest .

docker-build-gpu: ## Build Docker image with GPU support
	@echo "$(GREEN)Building Docker image with GPU support...$(NC)"
	docker build --target gpu-runtime -t face-recognition:gpu .

docker-run: ## Run with docker-compose
	@echo "$(GREEN)Starting services with docker-compose...$(NC)"
	docker-compose up -d

docker-run-monitoring: ## Run with monitoring stack
	@echo "$(GREEN)Starting services with monitoring...$(NC)"
	docker-compose --profile monitoring up -d

docker-stop: ## Stop docker-compose services
	@echo "$(RED)Stopping services...$(NC)"
	docker-compose down

docker-logs: ## Show docker logs
	docker-compose logs -f

docker-clean: ## Clean docker resources
	@echo "$(RED)Cleaning Docker resources...$(NC)"
	docker-compose down -v
	docker system prune -f

env-setup: ## Setup environment file
	@echo "$(GREEN)Setting up environment file...$(NC)"
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file from .env.example"; \
	else \
		echo ".env file already exists"; \
	fi

pre-commit: ## Run pre-commit hooks
	@echo "$(GREEN)Running pre-commit hooks...$(NC)"
	$(VENV)/bin/pre-commit run --all-files

setup: env-setup dev-install migrate ## Complete development setup
	@echo "$(GREEN)Setup complete! Run 'make run' to start the application.$(NC)"

monitoring-setup: ## Setup monitoring configuration
	@echo "$(GREEN)Setting up monitoring...$(NC)"
	mkdir -p monitoring/grafana/dashboards monitoring/grafana/datasources
	@echo "Monitoring directories created"

check: lint test ## Run all checks (lint + test)
	@echo "$(GREEN)All checks passed!$(NC)"

.DEFAULT_GOAL := help