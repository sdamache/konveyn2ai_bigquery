# KonveyN2AI BigQuery Vector Backend Makefile
# 
# Main commands:
#   make setup    - Create BigQuery tables and indexes
#   make migrate  - Migrate from Vertex AI to BigQuery
#   make run      - Start BigQuery vector backend
#   make test     - Run all tests
#   make clean    - Clean temporary files

.PHONY: help setup migrate run test clean install lint format check-env

# Default target
help:
	@echo "KonveyN2AI BigQuery Vector Backend"
	@echo "=================================="
	@echo ""
	@echo "Available commands:"
	@echo "  make setup     - Create BigQuery dataset, tables and indexes"
	@echo "  make migrate   - Migrate vectors from Vertex AI to BigQuery"
	@echo "  make run       - Start BigQuery vector backend API server"
	@echo "  make test      - Run all tests (contract, integration, unit)"
	@echo "  make install   - Install all dependencies"
	@echo "  make lint      - Run linting and formatting checks"
	@echo "  make format    - Format code with black and ruff"
	@echo "  make clean     - Clean temporary files and caches"
	@echo ""
	@echo "Development commands:"
	@echo "  make dev-setup - Complete development environment setup"
	@echo "  make logs      - Show application logs"
	@echo "  make diagnose  - Run system diagnostics"

# Environment check
check-env:
	@echo "🔍 Checking environment..."
	@if [ -z "$$GOOGLE_CLOUD_PROJECT" ]; then \
		echo "❌ GOOGLE_CLOUD_PROJECT environment variable not set"; \
		echo "   Run: export GOOGLE_CLOUD_PROJECT=konveyn2ai"; \
		exit 1; \
	fi
	@if [ -z "$$BIGQUERY_DATASET_ID" ]; then \
		echo "❌ BIGQUERY_DATASET_ID environment variable not set"; \
		echo "   Run: export BIGQUERY_DATASET_ID=semantic_gap_detector"; \
		exit 1; \
	fi
	@echo "✅ Environment variables configured"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Development setup
dev-setup: install
	@echo "🛠️ Setting up development environment..."
	pip install -e .
	pre-commit install
	@echo "✅ Development environment ready"

# BigQuery setup
setup: check-env
	@echo "🚀 Setting up BigQuery infrastructure..."
	@echo "Project: $$GOOGLE_CLOUD_PROJECT"
	@echo "Dataset: $$BIGQUERY_DATASET_ID"
	python -m src.janapada_memory.schema_manager create-all
	@echo "✅ BigQuery setup completed"

# Migration from Vertex AI
migrate: check-env setup
	@echo "🔄 Starting migration from Vertex AI to BigQuery..."
	python scripts/migrate_to_bigquery.py
	@echo "✅ Migration completed"

# Run the vector backend API
run: check-env
	@echo "🚀 Starting BigQuery vector backend..."
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Run all tests
test:
	@echo "🧪 Running all tests..."
	pytest tests/ -v --tb=short
	@echo "✅ All tests completed"

# Run contract tests only
test-contract:
	@echo "🧪 Running contract tests..."
	pytest tests/contract/ -v --tb=short

# Run integration tests only  
test-integration:
	@echo "🧪 Running integration tests..."
	pytest tests/integration/ -v --tb=short

# Run unit tests only
test-unit:
	@echo "🧪 Running unit tests..."
	pytest tests/unit/ -v --tb=short

# Linting and formatting
lint:
	@echo "🔍 Running linting checks..."
	ruff check src/ tests/ scripts/
	black --check src/ tests/ scripts/
	mypy src/

format:
	@echo "🎨 Formatting code..."
	black src/ tests/ scripts/
	ruff --fix src/ tests/ scripts/

# Utility commands
logs:
	@echo "📋 Application logs..."
	@if [ -f "logs/bigquery_vector.log" ]; then \
		tail -f logs/bigquery_vector.log; \
	else \
		echo "No log file found. Run 'make run' first."; \
	fi

diagnose: check-env
	@echo "🩺 Running system diagnostics..."
	python -m src.common.diagnostics

clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage
	@echo "✅ Cleanup completed"

# Performance and monitoring
benchmark: check-env
	@echo "⚡ Running performance benchmarks..."
	python scripts/benchmark_performance.py

verify-migration: check-env
	@echo "✅ Verifying migration quality..."
	python scripts/verify_migration_quality.py

# Demo and validation
demo: check-env setup
	@echo "🎪 Running end-to-end demo..."
	python demo.py

quickstart-validate:
	@echo "📋 Validating quickstart guide..."
	@echo "This will execute the complete quickstart guide end-to-end"
	bash scripts/validate_quickstart.sh

# Docker support
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t konveyn2ai-bigquery-backend .

docker-run: check-env
	@echo "🐳 Running Docker container..."
	docker run -d \
		-p 8000:8000 \
		-e GOOGLE_CLOUD_PROJECT=$$GOOGLE_CLOUD_PROJECT \
		-e BIGQUERY_DATASET_ID=$$BIGQUERY_DATASET_ID \
		-e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/service-account.json \
		-v $(PWD)/credentials:/app/credentials \
		konveyn2ai-bigquery-backend

# Cloud deployment
deploy-cloud-run:
	@echo "☁️ Deploying to Google Cloud Run..."
	bash deployment/scripts/deploy-to-cloud-run.sh

# Development helpers
shell:
	@echo "🐚 Starting development shell with environment loaded..."
	python -c "from src.common.config import load_config; load_config(); import IPython; IPython.start_ipython()"

# Monitoring and cost analysis
monitor-costs:
	@echo "💰 Analyzing BigQuery costs..."
	python -c "from src.janapada_memory.cost_monitor import CostMonitor; CostMonitor().generate_report()"

monitor-performance:
	@echo "📊 Monitoring BigQuery performance..."
	python -c "from src.janapada_memory.performance_monitor import PerformanceMonitor; PerformanceMonitor().generate_report()"

# Emergency procedures
rollback: check-env
	@echo "⏪ Rolling back to Vertex AI..."
	@echo "WARNING: This will switch back to the previous vector backend"
	@read -p "Are you sure you want to rollback? [y/N]: " confirm && [ $$confirm = "y" ]
	python scripts/rollback_to_vertex.py

# Git workflow helpers
commit-setup:
	@echo "💾 Committing setup changes..."
	git add Makefile requirements.txt .env.example
	git commit -m "feat: add BigQuery vector backend infrastructure setup (T001-T003)"

# Health checks
health-check:
	@echo "❤️ Checking system health..."
	@curl -f http://localhost:8000/health || echo "API server not running. Run 'make run' first."

# Export/backup utilities  
export-schema:
	@echo "📤 Exporting BigQuery schema..."
	python -c "from src.janapada_memory.schema_manager import SchemaManager; SchemaManager().export_schema('backups/schema_backup.json')"

backup-data:
	@echo "💾 Creating data backup..."
	python scripts/backup_bigquery_data.py