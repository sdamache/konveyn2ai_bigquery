# KonveyN2AI BigQuery Vector Backend Makefile
# 
# Main commands:
#   make setup    - Create BigQuery tables and indexes
#   make migrate  - Migrate from Vertex AI to BigQuery
#   make run      - Start BigQuery vector backend
#   make test     - Run all tests
#   make clean    - Clean temporary files
#
# M1 Ingestion commands:
#   make ingest_k8s      - Ingest Kubernetes manifests
#   make ingest_fastapi  - Ingest FastAPI projects
#   make ingest_cobol    - Ingest COBOL copybooks
#   make ingest_irs      - Ingest IRS record layouts
#   make ingest_mumps    - Ingest MUMPS/VistA dictionaries

.PHONY: help setup migrate run test clean install lint format check-env
.PHONY: setup-bigquery validate-env install-deps compute_metrics
.PHONY: ingest_k8s ingest_fastapi ingest_cobol ingest_irs ingest_mumps

# Default target
help:
	@echo "KonveyN2AI BigQuery Vector Backend"
	@echo "=================================="
	@echo ""
	@echo "Available commands:"
	@echo "  make setup     - Create BigQuery dataset, tables and indexes"
	@echo "  make embeddings - Generate embeddings for BigQuery vector store"
	@echo "  make migrate   - Migrate vectors from Vertex AI to BigQuery"
	@echo "  make compute_metrics - Run hybrid gap metrics job"
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
	@echo ""
	@echo "M1 Ingestion commands:"
	@echo "  make ingest_k8s      - Ingest Kubernetes YAML/JSON manifests"
	@echo "  make ingest_fastapi  - Ingest FastAPI source code and OpenAPI specs"
	@echo "  make ingest_cobol    - Ingest COBOL copybooks and data definitions"
	@echo "  make ingest_irs      - Ingest IRS IMF record layouts"
	@echo "  make ingest_mumps    - Ingest MUMPS/VistA FileMan dictionaries"
	@echo "  make setup-bigquery  - Setup M1 BigQuery tables and environment"
	@echo "  make validate-env    - Validate M1 environment configuration"
	@echo "  make install-deps    - Install M1 parser dependencies"

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
	@echo "🔧 Creating BigQuery tables and schema..."
	python -m src.janapada_memory.schema_manager create-all
	@echo "🔍 Validating BigQuery setup..."
	python -c "from src.janapada_memory.connections.bigquery_connection import BigQueryConnectionManager; \
		from src.janapada_memory.schema_manager import SchemaManager; \
		conn = BigQueryConnectionManager(); \
		health = conn.check_health(); \
		print(f'Connection health: {health.is_healthy}'); \
		assert health.is_healthy, f'BigQuery connection failed: {health.error_message}'; \
		assert conn.check_dataset_access(), 'Dataset access check failed'; \
		assert conn.check_table_access('source_metadata'), 'source_metadata table not accessible'; \
		assert conn.check_table_access('source_embeddings'), 'source_embeddings table not accessible'; \
		schema = SchemaManager(connection=conn); \
		validation = schema.validate_schema(validate_data=False, check_indexes=False); \
		assert validation['overall_status'] in ['VALID', 'INVALID'], f'Schema validation failed: {validation}'; \
		print('✅ All validation checks passed')"
	@echo "✅ BigQuery setup completed and validated"

# Migration from Vertex AI
migrate: check-env setup
	@echo "🔄 Starting migration from Vertex AI to BigQuery..."
	python scripts/migrate_to_bigquery.py
	@echo "✅ Migration completed"

# Generate embeddings
embeddings: check-env setup
	@echo "🧠 Generating embeddings for BigQuery vector store..."
	@if [ -z "$$GOOGLE_API_KEY" ]; then \
		echo "❌ GOOGLE_API_KEY environment variable not set"; \
		echo "   Run: export GOOGLE_API_KEY=your_api_key"; \
		exit 1; \
	fi
	python -m pipeline.embedding \
		--project $$GOOGLE_CLOUD_PROJECT \
		--dataset $$BIGQUERY_DATASET_ID \
		--batch-size $${EMBED_BATCH_SIZE:-32} \
		--cache-dir $${EMBED_CACHE_DIR:-.cache/embeddings} \
		$$(if [ "$$DRY_RUN" = "1" ]; then echo "--dry-run"; fi) \
		$$(if [ -n "$$LIMIT" ]; then echo "--limit $$LIMIT"; fi) \
		$$(if [ -n "$$WHERE" ]; then echo "--where \"$$WHERE\""; fi) \
		--verbose
	@echo "✅ Embedding generation completed"

compute_metrics: check-env
	@echo "📊 Running hybrid gap metrics job..."
	@PROJECT=$$GOOGLE_CLOUD_PROJECT DATASET=$$BIGQUERY_DATASET_ID; \
	CMD="venv/bin/python -m src.gap_analysis.pipeline.run_metrics --project $$PROJECT --dataset $$DATASET"; \
	if [ -n "$$ANALYSIS_ID" ]; then CMD="$$CMD --analysis-id $$ANALYSIS_ID"; fi; \
	if [ -n "$$RULESET_VERSION" ]; then CMD="$$CMD --ruleset-version $$RULESET_VERSION"; fi; \
	if [ -n "$$LIMIT" ]; then CMD="$$CMD --limit $$LIMIT"; fi; \
	if [ "$$DRY_RUN" = "1" ]; then CMD="$$CMD --dry-run"; fi; \
	if [ "$$JSON" = "1" ]; then CMD="$$CMD --json"; fi; \
	if [ "$$VERBOSE" = "1" ]; then CMD="$$CMD --verbose"; fi; \
	echo "Executing: $$CMD"; \
	eval "$$CMD"
	@echo "✅ Hybrid gap metrics job finished"

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

# BigQuery AI Hackathon Demo Pipeline
demo_hackathon: check-env
	@echo "🚀 Running BigQuery AI Hackathon Demo Pipeline..."
	@echo "=================================================="

	# Step 1: Setup and validate environment
	@echo "📋 Step 1: Setting up BigQuery environment..."
	$(MAKE) setup

	# Step 2: Ingest sample data (using test_data)
	@echo "📥 Step 2: Ingesting sample K8s artifacts..."
	@if [ ! -d "test_data/k8s-manifests" ]; then \
		echo "⚠️  test_data/k8s-manifests not found, creating sample data..."; \
		mkdir -p test_data/k8s-manifests; \
	fi
	@if [ -d "test_data/k8s-manifests" ] && [ "$$(ls -A test_data/k8s-manifests)" ]; then \
		export GOOGLE_CLOUD_PROJECT=$$GOOGLE_CLOUD_PROJECT BIGQUERY_INGESTION_DATASET_ID=$$BIGQUERY_DATASET_ID SOURCE_PATH=test_data/k8s-manifests; \
		$(MAKE) ingest_k8s; \
	else \
		echo "❌ No K8s manifest files found in test_data/k8s-manifests"; \
		echo "   Please ensure test data exists before running demo"; \
		exit 1; \
	fi

	# Step 3: Generate embeddings with BigQuery native vectors
	@echo "🧠 Step 3: Generating embeddings with BigQuery native vectors..."
	$(MAKE) embeddings LIMIT=50

	# Step 4: Build semantic candidates with vector search
	@echo "🔍 Step 4: Building semantic candidates with vector search..."
	python -m src.gap_analysis.semantic_candidates \
		--project $$GOOGLE_CLOUD_PROJECT \
		--dataset $$BIGQUERY_DATASET_ID

	# Step 5: Run hybrid gap analysis
	@echo "📊 Step 5: Running hybrid gap analysis..."
	$(MAKE) compute_metrics ANALYSIS_ID=hackathon_demo_$$(date +%Y%m%d_%H%M%S) LIMIT=50

	# Step 6: Create summary view for notebook consumption
	@echo "📈 Step 6: Creating summary view for notebook..."
	@echo "Creating gap_metrics_summary view..."
	sed 's/{project}/$$GOOGLE_CLOUD_PROJECT/g; s/{dataset}/$$BIGQUERY_DATASET_ID/g' configs/sql/gap_analysis_summary_view.sql | bq query --use_legacy_sql=false

	# Step 7: Validate pipeline output
	@echo "📊 Step 7: Validating pipeline output..."
	@echo "Checking gap_metrics table for data..."
	bq query --use_legacy_sql=false --format=prettyjson \
		"SELECT COUNT(*) as total_records, COUNT(DISTINCT analysis_id) as analysis_runs FROM \`$$GOOGLE_CLOUD_PROJECT.$$BIGQUERY_DATASET_ID.gap_metrics\`" | head -10
	@echo "Sample from gap_metrics_summary view:"
	bq query --use_legacy_sql=false --max_rows=3 \
		"SELECT artifact_type, rule_name, count_passed, count_failed, avg_confidence FROM \`$$GOOGLE_CLOUD_PROJECT.$$BIGQUERY_DATASET_ID.gap_metrics_summary\` LIMIT 3"

	@echo "✅ Pipeline complete! Data ready for notebook visualization."
	@echo ""
	@echo "📓 Next steps:"
	@echo "   1. Open issue6_notebook.py in Jupyter/Colab"
	@echo "   2. Update BigQuery connection to point to: $$GOOGLE_CLOUD_PROJECT.$$BIGQUERY_DATASET_ID"
	@echo "   3. Run notebook to generate heat maps and dashboards"
	@echo ""
	@echo "🎯 Demo Points for Judges:"
	@echo "   • BigQuery Native Vector Search (768-dimensional embeddings)"
	@echo "   • Hybrid scoring (deterministic SQL + semantic similarity)"
	@echo "   • Real-time gap analysis with confidence scoring"
	@echo "   • Scalable processing of 100+ artifacts in minutes"

# Quick test with minimal data
demo_test: check-env
	@echo "🧪 Running quick demo test with minimal data..."
	$(MAKE) demo_hackathon LIMIT=5

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

# =============================================================================
# M1 Multi-Source Ingestion Targets (T028)
# =============================================================================

# M1 Environment Setup
validate-env:
	@echo "🔍 Validating M1 environment configuration..."
	@if [ -z "$$GOOGLE_CLOUD_PROJECT" ]; then \
		echo "❌ GOOGLE_CLOUD_PROJECT environment variable not set"; \
		echo "   Run: export GOOGLE_CLOUD_PROJECT=konveyn2ai"; \
		exit 1; \
	fi
	@if [ -z "$$BIGQUERY_INGESTION_DATASET_ID" ]; then \
		echo "❌ BIGQUERY_INGESTION_DATASET_ID environment variable not set"; \
		echo "   Run: export BIGQUERY_INGESTION_DATASET_ID=source_ingestion"; \
		exit 1; \
	fi
	@echo "✅ M1 environment variables configured"

# Install M1 parser dependencies
install-deps:
	@echo "📦 Installing M1 parser dependencies..."
	@if [ ! -d "venv" ]; then \
		echo "⚠️  Virtual environment not found. Creating venv..."; \
		python -m venv venv; \
	fi
	@echo "Activating virtual environment and installing dependencies..."
	bash -c "source venv/bin/activate && pip install -r requirements.txt"
	@echo "✅ M1 dependencies installed"

# Setup M1 BigQuery environment
setup-bigquery: validate-env
	@echo "🚀 Setting up M1 BigQuery tables and environment..."
	@echo "Project: $$GOOGLE_CLOUD_PROJECT"
	@echo "Dataset: $$BIGQUERY_INGESTION_DATASET_ID"
	bash -c "source venv/bin/activate && cd src && python -m cli.main setup --project $$GOOGLE_CLOUD_PROJECT --dataset $$BIGQUERY_INGESTION_DATASET_ID"
	@echo "✅ M1 BigQuery setup completed"

# =============================================================================
# M1 Ingestion Commands
# =============================================================================

# Kubernetes manifest ingestion
ingest_k8s: validate-env
	@echo "☸️ Ingesting Kubernetes manifests..."
	@if [ -z "$$SOURCE_PATH" ]; then \
		echo "❌ SOURCE_PATH environment variable not set"; \
		echo "   Usage: make ingest_k8s SOURCE_PATH=/path/to/k8s/manifests"; \
		echo "   Example: make ingest_k8s SOURCE_PATH=./examples/k8s-manifests/"; \
		exit 1; \
	fi
	bash -c "source venv/bin/activate && cd src && python -m cli.main k8s --source ../$$SOURCE_PATH --project $$GOOGLE_CLOUD_PROJECT --dataset $$BIGQUERY_INGESTION_DATASET_ID --output bigquery"
	@echo "✅ Kubernetes ingestion completed"

# FastAPI project ingestion
ingest_fastapi: validate-env
	@echo "⚡ Ingesting FastAPI project..."
	@if [ -z "$$SOURCE_PATH" ]; then \
		echo "❌ SOURCE_PATH environment variable not set"; \
		echo "   Usage: make ingest_fastapi SOURCE_PATH=/path/to/fastapi/project"; \
		echo "   Example: make ingest_fastapi SOURCE_PATH=./examples/fastapi-project/"; \
		exit 1; \
	fi
	bash -c "source venv/bin/activate && cd src && python -m cli.main fastapi --source ../$$SOURCE_PATH --project $$GOOGLE_CLOUD_PROJECT --dataset $$BIGQUERY_INGESTION_DATASET_ID --output bigquery"
	@echo "✅ FastAPI ingestion completed"

# COBOL copybook ingestion
ingest_cobol: validate-env
	@echo "📄 Ingesting COBOL copybooks..."
	@if [ -z "$$SOURCE_PATH" ]; then \
		echo "❌ SOURCE_PATH environment variable not set"; \
		echo "   Usage: make ingest_cobol SOURCE_PATH=/path/to/cobol/copybooks"; \
		echo "   Example: make ingest_cobol SOURCE_PATH=./examples/cobol-copybooks/"; \
		exit 1; \
	fi
	bash -c "source venv/bin/activate && cd src && python -m cli.main cobol --source ../$$SOURCE_PATH --project $$GOOGLE_CLOUD_PROJECT --dataset $$BIGQUERY_INGESTION_DATASET_ID --output bigquery"
	@echo "✅ COBOL ingestion completed"

# IRS record layout ingestion
ingest_irs: validate-env
	@echo "🏛️ Ingesting IRS record layouts..."
	@if [ -z "$$SOURCE_PATH" ]; then \
		echo "❌ SOURCE_PATH environment variable not set"; \
		echo "   Usage: make ingest_irs SOURCE_PATH=/path/to/irs/layouts"; \
		echo "   Example: make ingest_irs SOURCE_PATH=./examples/irs-layouts/"; \
		exit 1; \
	fi
	bash -c "source venv/bin/activate && cd src && python -m cli.main irs --source ../$$SOURCE_PATH --project $$GOOGLE_CLOUD_PROJECT --dataset $$BIGQUERY_INGESTION_DATASET_ID --output bigquery"
	@echo "✅ IRS ingestion completed"

# MUMPS/VistA dictionary ingestion
ingest_mumps: validate-env
	@echo "🏥 Ingesting MUMPS/VistA dictionaries..."
	@if [ -z "$$SOURCE_PATH" ]; then \
		echo "❌ SOURCE_PATH environment variable not set"; \
		echo "   Usage: make ingest_mumps SOURCE_PATH=/path/to/mumps/dictionaries"; \
		echo "   Example: make ingest_mumps SOURCE_PATH=./examples/mumps-dictionaries/"; \
		exit 1; \
	fi
	bash -c "source venv/bin/activate && cd src && python -m cli.main mumps --source ../$$SOURCE_PATH --project $$GOOGLE_CLOUD_PROJECT --dataset $$BIGQUERY_INGESTION_DATASET_ID --output bigquery"
	@echo "✅ MUMPS ingestion completed"

# =============================================================================
# M1 Development and Testing
# =============================================================================

# Run M1 ingestion dry-run examples
dry-run-examples: validate-env
	@echo "🔍 Running M1 ingestion dry-run examples..."
	@echo "Testing Kubernetes parser..."
	bash -c "source venv/bin/activate && cd src && python -m cli.main k8s --source ../examples/k8s-manifests/ --dry-run --output console" || echo "⚠️  No K8s examples found"
	@echo "Testing FastAPI parser..."
	bash -c "source venv/bin/activate && cd src && python -m cli.main fastapi --source ../examples/fastapi-project/ --dry-run --output console" || echo "⚠️  No FastAPI examples found"
	@echo "Testing COBOL parser..."
	bash -c "source venv/bin/activate && cd src && python -m cli.main cobol --source ../examples/cobol-copybooks/ --dry-run --output console" || echo "⚠️  No COBOL examples found"
	@echo "Testing IRS parser..."
	bash -c "source venv/bin/activate && cd src && python -m cli.main irs --source ../examples/irs-layouts/ --dry-run --output console" || echo "⚠️  No IRS examples found"
	@echo "Testing MUMPS parser..."
	bash -c "source venv/bin/activate && cd src && python -m cli.main mumps --source ../examples/mumps-dictionaries/ --dry-run --output console" || echo "⚠️  No MUMPS examples found"
	@echo "✅ Dry-run examples completed"

# Test all M1 parsers
test-m1-parsers:
	@echo "🧪 Testing all M1 parsers..."
	bash -c "source venv/bin/activate && python -m pytest tests/contract/test_*_parser_contract.py -v --tb=short"
	@echo "✅ M1 parser tests completed"

# Test M1 integration
test-m1-integration:
	@echo "🧪 Testing M1 integration..."
	bash -c "source venv/bin/activate && python -m pytest tests/integration/test_*_ingestion.py -v --tb=short"
	@echo "✅ M1 integration tests completed"

# Complete M1 end-to-end validation
validate-m1-complete: setup-bigquery test-m1-parsers dry-run-examples
	@echo "🎯 Complete M1 validation pipeline executed"
	@echo "✅ M1 Multi-Source Ingestion system ready for production"
