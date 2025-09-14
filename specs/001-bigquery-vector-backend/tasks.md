# Tasks: BigQuery Vector Backend

**Input**: Design documents from `/specs/001-bigquery-vector-backend/`
**Prerequisites**: plan.md (✓), research.md (✓), data-model.md (✓), contracts/ (✓)

## Implementation Overview

Replace KonveyN2AI's Vertex AI vector store with BigQuery VECTOR tables. Implement three-table schema (source_metadata, source_embeddings, gap_metrics) with 768-dimensional embeddings while maintaining API compatibility.

**Tech Stack**: Python 3.11, google-cloud-bigquery, scikit-learn, pytest
**Architecture**: Single project with libraries: bigquery_vector_store, schema_manager

## GitHub Checklist Format

Each task includes concrete acceptance criteria for GitHub issue tracking.

## Phase 3.1: Setup & Infrastructure

- [ ] **T001** Create project structure and Makefile for BigQuery backend
  - **File**: `/Makefile`
  - **Acceptance**: `make setup`, `make migrate`, `make run` commands work
  - **Dependencies**: None

- [ ] **T002** [P] Install BigQuery and ML dependencies
  - **File**: `/requirements.txt` 
  - **Acceptance**: Added google-cloud-bigquery>=3.0.0, scikit-learn>=1.3.0
  - **Dependencies**: None

- [ ] **T003** [P] Configure environment variables and authentication
  - **File**: `/.env.example`
  - **Acceptance**: GOOGLE_CLOUD_PROJECT, BIGQUERY_DATASET_ID, GOOGLE_APPLICATION_CREDENTIALS variables documented
  - **Dependencies**: None

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Vector Store API Contract Tests

- [ ] **T004** [P] Contract test POST /vector-store/embeddings
  - **File**: `/tests/contract/test_vector_store_post_embeddings.py`
  - **Acceptance**: Test validates embedding insertion with 768-dim vectors, returns 201/400/409
  - **Dependencies**: T001

- [ ] **T005** [P] Contract test GET /vector-store/embeddings 
  - **File**: `/tests/contract/test_vector_store_get_embeddings.py`
  - **Acceptance**: Test validates pagination, filtering by artifact_type, returns 200
  - **Dependencies**: T001

- [ ] **T006** [P] Contract test POST /vector-store/search
  - **File**: `/tests/contract/test_vector_store_search.py`
  - **Acceptance**: Test validates similarity search with text and vector queries, returns 200/400
  - **Dependencies**: T001

- [ ] **T007** [P] Contract test GET /vector-store/embeddings/{chunk_id}
  - **File**: `/tests/contract/test_vector_store_get_by_id.py`
  - **Acceptance**: Test validates chunk retrieval, returns 200/404
  - **Dependencies**: T001

### Schema Manager API Contract Tests

- [ ] **T008** [P] Contract test POST /schema/tables
  - **File**: `/tests/contract/test_schema_manager_create_tables.py`
  - **Acceptance**: Test validates table creation, returns 201/409
  - **Dependencies**: T001

- [ ] **T009** [P] Contract test POST /schema/indexes
  - **File**: `/tests/contract/test_schema_manager_create_indexes.py`
  - **Acceptance**: Test validates vector index creation, returns 201/400
  - **Dependencies**: T001

- [ ] **T010** [P] Contract test POST /schema/validate
  - **File**: `/tests/contract/test_schema_manager_validate.py`
  - **Acceptance**: Test validates schema compliance check, returns 200
  - **Dependencies**: T001

### Integration Test Scenarios

- [ ] **T011** [P] Integration test: End-to-end vector workflow
  - **File**: `/tests/integration/test_vector_workflow.py`
  - **Acceptance**: Insert vectors → Search similar → Validate results with actual BigQuery
  - **Dependencies**: T001

- [ ] **T012** [P] Integration test: BigQuery setup workflow  
  - **File**: `/tests/integration/test_setup_workflow.py`
  - **Acceptance**: `make setup` → Tables created → Vector index active → Ready for use
  - **Dependencies**: T001

- [ ] **T013** [P] Integration test: Migration quality verification
  - **File**: `/tests/integration/test_migration_quality.py`
  - **Acceptance**: PCA migration → Semantic similarity preserved → Performance validated
  - **Dependencies**: T001

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Data Models

- [ ] **T014** [P] Implement Pydantic models for data validation
  - **File**: `/src/models/vector_models.py`
  - **Acceptance**: SourceMetadata, SourceEmbedding, GapMetric models with validation
  - **Dependencies**: T004-T013 (all tests failing)

### Schema Management Library

- [ ] **T015** [P] Implement BigQuery schema manager
  - **File**: `/src/janapada_memory/schema_manager.py`
  - **Acceptance**: create_tables(), validate_schema(), create_indexes() methods implemented
  - **Dependencies**: T014

- [ ] **T016** [P] Implement schema manager CLI
  - **File**: `/src/cli/schema_cli.py`
  - **Acceptance**: `schema-manager --help`, `schema-manager create`, `schema-manager validate`
  - **Dependencies**: T015

### Vector Store Library

- [ ] **T017** [P] Implement BigQuery vector store core
  - **File**: `/src/janapada_memory/bigquery_vector_store.py`
  - **Acceptance**: insert_embedding(), search_similar_vectors(), health_check() methods
  - **Dependencies**: T014, T015

- [ ] **T018** [P] Implement vector store CLI
  - **File**: `/src/cli/vector_cli.py`
  - **Acceptance**: `bigquery-vector --help`, `bigquery-vector search`, `bigquery-vector insert`
  - **Dependencies**: T017

### Migration Utilities

- [ ] **T019** [P] Implement PCA dimension reduction
  - **File**: `/scripts/reduce_dimensions.py`
  - **Acceptance**: 3072→768 PCA reduction with 99% variance preservation
  - **Dependencies**: T002

- [ ] **T020** [P] Implement Vertex AI export utility
  - **File**: `/scripts/export_vertex_vectors.py`
  - **Acceptance**: Export existing vectors to JSON with metadata
  - **Dependencies**: None

- [ ] **T021** [P] Implement BigQuery import utility
  - **File**: `/scripts/import_to_bigquery.py`
  - **Acceptance**: Batch import vectors with validation and error handling
  - **Dependencies**: T017, T019

### API Endpoints (FastAPI)

- [ ] **T022** Implement vector store embeddings endpoints
  - **File**: `/src/api/vector_endpoints.py`
  - **Acceptance**: POST/GET /vector-store/embeddings with request validation
  - **Dependencies**: T017

- [ ] **T023** Implement vector search endpoint
  - **File**: `/src/api/vector_endpoints.py` 
  - **Acceptance**: POST /vector-store/search with similarity threshold
  - **Dependencies**: T017, T022

- [ ] **T024** Implement schema management endpoints
  - **File**: `/src/api/schema_endpoints.py`
  - **Acceptance**: POST /schema/tables, /schema/indexes, /schema/validate
  - **Dependencies**: T015

## Phase 3.4: Integration & Optimization

- [ ] **T025** Integrate authentication with BigQuery ADC
  - **File**: `/src/common/bigquery_auth.py`
  - **Acceptance**: Multi-environment auth (local, Cloud Run, Colab) working
  - **Dependencies**: T017

- [ ] **T026** Implement error handling and logging
  - **File**: `/src/common/error_handlers.py`
  - **Acceptance**: Structured logging, BigQuery-specific error handling
  - **Dependencies**: T017, T022-T024

- [ ] **T027** Implement query optimization and caching
  - **File**: `/src/janapada_memory/query_optimizer.py`
  - **Acceptance**: Partition pruning, result caching, <200ms latency
  - **Dependencies**: T017

- [ ] **T028** Create Makefile automation
  - **File**: `/Makefile`
  - **Acceptance**: `make setup`, `make migrate`, `make test`, `make run` targets working
  - **Dependencies**: T015, T021, All core components

## Phase 3.5: Migration & Validation

- [ ] **T029** Implement migration orchestration script
  - **File**: `/scripts/migrate_to_bigquery.py`
  - **Acceptance**: End-to-end migration: export → reduce → import → validate
  - **Dependencies**: T020, T019, T021

- [ ] **T030** [P] Create migration quality verification
  - **File**: `/scripts/verify_migration_quality.py`
  - **Acceptance**: Compare search results before/after migration, >80% top-3 overlap
  - **Dependencies**: T029

- [ ] **T031** [P] Create performance benchmarking
  - **File**: `/scripts/benchmark_performance.py`
  - **Acceptance**: Latency benchmarks, 95th percentile <200ms
  - **Dependencies**: T017

- [ ] **T032** [P] Create demo workflow script
  - **File**: `/demo.py`
  - **Acceptance**: Complete demo: setup → insert → search → validate
  - **Dependencies**: T017, T028

## Phase 3.6: Polish & Documentation

- [ ] **T033** [P] Unit tests for vector operations
  - **File**: `/tests/unit/test_vector_operations.py`
  - **Acceptance**: Test embedding validation, similarity calculations
  - **Dependencies**: T017

- [ ] **T034** [P] Unit tests for schema operations  
  - **File**: `/tests/unit/test_schema_operations.py`
  - **Acceptance**: Test table creation, validation logic
  - **Dependencies**: T015

- [ ] **T035** [P] Create library documentation
  - **File**: `/docs/llms.txt`
  - **Acceptance**: API documentation for bigquery_vector_store and schema_manager
  - **Dependencies**: T015, T017

- [ ] **T036** [P] Performance optimization documentation
  - **File**: `/docs/performance_tuning.md`
  - **Acceptance**: BigQuery slot management, query optimization guide
  - **Dependencies**: T027

- [ ] **T037** Final quickstart validation
  - **File**: `/specs/001-bigquery-vector-backend/quickstart.md`
  - **Acceptance**: Execute complete quickstart guide end-to-end successfully
  - **Dependencies**: All previous tasks

## Dependencies

```
Setup (T001-T003) 
  ↓
Tests (T004-T013) [All must fail]
  ↓ 
Core Implementation (T014-T024)
  ↓
Integration (T025-T028)
  ↓
Migration (T029-T032)
  ↓
Polish (T033-T037)
```

### Critical Dependencies
- **T004-T013 MUST FAIL** before starting T014-T024
- T015 (schema_manager) blocks T017 (vector_store)
- T017 blocks T022-T024 (API endpoints) 
- T028 (Makefile) requires T015, T021, and core components
- T029 (migration) requires T020, T019, T021

## Parallel Execution Examples

### Phase 3.2: Launch Contract Tests Together
```bash
# All can run in parallel - different files, no dependencies
Task: "Contract test POST /vector-store/embeddings in tests/contract/test_vector_store_post_embeddings.py"
Task: "Contract test GET /vector-store/embeddings in tests/contract/test_vector_store_get_embeddings.py" 
Task: "Contract test POST /vector-store/search in tests/contract/test_vector_store_search.py"
Task: "Contract test POST /schema/tables in tests/contract/test_schema_manager_create_tables.py"
Task: "Integration test vector workflow in tests/integration/test_vector_workflow.py"
```

### Phase 3.3: Launch Model Creation Together
```bash
# Different files, can be parallel
Task: "Implement Pydantic models in src/models/vector_models.py"
Task: "Implement PCA dimension reduction in scripts/reduce_dimensions.py"
Task: "Implement Vertex AI export in scripts/export_vertex_vectors.py"
```

### Phase 3.6: Launch Documentation Together  
```bash
# Different files, independent work
Task: "Unit tests for vector operations in tests/unit/test_vector_operations.py"
Task: "Unit tests for schema operations in tests/unit/test_schema_operations.py"
Task: "Create library documentation in docs/llms.txt"
Task: "Performance optimization guide in docs/performance_tuning.md"
```

## Acceptance Criteria Summary

**Setup Complete**: Makefile targets work, dependencies installed, environment configured
**Tests Failing**: All contract and integration tests written and failing (TDD Red phase)
**Core Implementation**: BigQuery vector store and schema manager libraries functional
**API Working**: FastAPI endpoints respond correctly to contract specifications
**Migration Ready**: Can migrate from Vertex AI to BigQuery with quality verification
**Performance Target**: <200ms vector search latency, 99% variance preservation
**Production Ready**: Authentication, error handling, logging, optimization implemented

## Notes

- **[P] tasks** = Different files, can run in parallel
- **Sequential tasks** = Same file or dependency relationships  
- **TDD Critical**: All tests (T004-T013) must be written and failing before implementation
- **File Paths**: All paths are absolute from repository root
- **Validation**: Each task specifies concrete acceptance criteria for GitHub tracking
- **Migration Focus**: Maintains API compatibility while switching backends