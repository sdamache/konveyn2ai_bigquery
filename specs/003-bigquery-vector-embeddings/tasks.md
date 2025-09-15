# Tasks: BigQuery Vector Embeddings Generation

**Input**: Design documents from `/specs/003-bigquery-vector-embeddings/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → Tech stack: Python 3.11+, google-generativeai, google-cloud-bigquery
   → Libraries: pathlib, hashlib for caching
   → Structure: Single project (pipeline module)
2. Load design documents:
   → data-model.md: EmbeddingRecord, CacheEntry, ProcessingStats entities
   → contracts/: embedding_pipeline_api.yaml, bigquery_schema.sql
   → research.md: Gemini API decisions, caching strategy
   → quickstart.md: Integration test scenarios
3. Generate tasks by category following TDD principles
4. Apply parallelization rules for independent files
5. Number tasks T001-T016 following dependency order
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Phase 3.1: Setup and Prerequisites

- [ ] **T001** Create pipeline module structure at `pipeline/` directory with `__init__.py`
  - **Acceptance**: Directory `pipeline/` exists with empty `__init__.py` file
  - **Files**: `pipeline/__init__.py`

- [ ] **T002** Install and configure BigQuery vector dependencies in requirements.txt
  - **Acceptance**: `google-generativeai>=0.7.0` and `google-cloud-bigquery>=3.11.0` added to requirements.txt
  - **Files**: `requirements.txt`

- [ ] **T003** [P] Configure embedding generation cache directory structure
  - **Acceptance**: `.cache/embeddings/` directory created with `.gitignore` entry
  - **Files**: `.cache/embeddings/.gitkeep`, `.gitignore`

## Phase 3.2: Contract Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

- [ ] **T004** [P] BigQuery schema validation contract test in `tests/contract/test_bigquery_embedding_schema.py`
  - **Acceptance**: Test validates source_embeddings table schema has VECTOR<768> column, proper clustering, and all required fields
  - **Files**: `tests/contract/test_bigquery_embedding_schema.py`
  - **Must Fail**: Schema validation fails until table is properly created

- [ ] **T005** [P] Embedding API contract test in `tests/contract/test_embedding_pipeline_api.py`
  - **Acceptance**: Test validates EmbeddingPipeline class interface matches OpenAPI spec (generate_embeddings method signature)
  - **Files**: `tests/contract/test_embedding_pipeline_api.py`
  - **Must Fail**: API contract fails until pipeline implementation exists

- [ ] **T006** [P] Vector search functionality contract test in `tests/contract/test_vector_search_contract.py`
  - **Acceptance**: Test validates BigQuery VECTOR_SEARCH() and ML.APPROXIMATE_NEIGHBORS() work with 768-dimensional vectors
  - **Files**: `tests/contract/test_vector_search_contract.py`
  - **Must Fail**: Vector search fails until embeddings are stored

- [ ] **T007** [P] Cache persistence contract test in `tests/contract/test_embedding_cache_contract.py`
  - **Acceptance**: Test validates cache can store/retrieve embeddings by SHA256 content hash with proper JSON format
  - **Files**: `tests/contract/test_embedding_cache_contract.py`
  - **Must Fail**: Cache contract fails until EmbeddingCache class exists

## Phase 3.3: Integration Tests (Sequential - BigQuery dependencies)

- [ ] **T008** End-to-end embedding generation integration test in `tests/integration/test_embedding_generation_e2e.py`
  - **Acceptance**: Test generates embeddings for sample chunks, stores in BigQuery, validates idempotent behavior
  - **Files**: `tests/integration/test_embedding_generation_e2e.py`
  - **Dependencies**: Requires T004-T007 to fail first

- [ ] **T009** Cache efficiency integration test in `tests/integration/test_cache_efficiency.py`
  - **Acceptance**: Test validates >70% cache hit rate on duplicate content, measures API cost savings
  - **Files**: `tests/integration/test_cache_efficiency.py`
  - **Dependencies**: Requires T008

- [ ] **T010** Batch processing integration test in `tests/integration/test_batch_processing.py`
  - **Acceptance**: Test processes 100+ chunks in batches of 32, validates throughput >100 chunks/minute
  - **Files**: `tests/integration/test_batch_processing.py`
  - **Dependencies**: Requires T008

- [ ] **T011** Vector similarity search integration test in `tests/integration/test_vector_similarity_search.py`
  - **Acceptance**: Test performs similarity search using BigQuery VECTOR_SEARCH(), returns top-5 results with cosine similarity
  - **Files**: `tests/integration/test_vector_similarity_search.py`
  - **Dependencies**: Requires T008

## Phase 3.4: Core Implementation (ONLY after tests are failing)

- [ ] **T012** [P] EmbeddingCache class implementation in `pipeline/embedding.py`
  - **Acceptance**: Implements SHA256-based disk cache with get/set methods, JSON serialization, cache hit/miss tracking
  - **Files**: `pipeline/embedding.py` (EmbeddingCache class only)
  - **Dependencies**: Must make T007 pass

- [ ] **T013** [P] EmbeddingGenerator class implementation in `pipeline/embedding.py`
  - **Acceptance**: Implements Gemini API client with batching (default 32), exponential backoff, API statistics tracking
  - **Files**: `pipeline/embedding.py` (EmbeddingGenerator class only)
  - **Dependencies**: Must make T005 pass partially

- [ ] **T014** EmbeddingPipeline orchestration class in `pipeline/embedding.py`
  - **Acceptance**: Implements end-to-end pipeline with idempotent behavior, BigQuery integration, comprehensive logging
  - **Files**: `pipeline/embedding.py` (EmbeddingPipeline class)
  - **Dependencies**: Requires T012, T013; Must make T005, T008 pass

- [ ] **T015** BigQuery schema enhancements in `src/janapada_memory/schema_manager.py`
  - **Acceptance**: Updates source_embeddings table schema with VECTOR<768>, proper clustering, validation functions
  - **Files**: `src/janapada_memory/schema_manager.py`
  - **Dependencies**: Must make T004 pass

- [ ] **T016** CLI interface and make command implementation
  - **Acceptance**: `python -m pipeline.embedding --help` works, `make embeddings` command executes successfully
  - **Files**: `pipeline/embedding.py` (main function), `Makefile` (embeddings target)
  - **Dependencies**: Requires T014, T015; Must make all contract tests pass

## Phase 3.5: Polish and Validation

- [ ] **T017** [P] Unit tests for EmbeddingCache in `tests/unit/test_embedding_cache.py`
  - **Acceptance**: Tests cache key generation, file I/O, error handling, cache statistics
  - **Files**: `tests/unit/test_embedding_cache.py`
  - **Dependencies**: Requires T012

- [ ] **T018** [P] Unit tests for EmbeddingGenerator in `tests/unit/test_embedding_generator.py`
  - **Acceptance**: Tests API client, batching logic, exponential backoff, error handling
  - **Files**: `tests/unit/test_embedding_generator.py` 
  - **Dependencies**: Requires T013

- [ ] **T019** Performance validation test in `tests/performance/test_embedding_performance.py`
  - **Acceptance**: Validates 100-500 chunks/minute throughput, <200ms API latency, 70-90% cache hit rate
  - **Files**: `tests/performance/test_embedding_performance.py`
  - **Dependencies**: Requires T016

- [ ] **T020** [P] Update README.md documentation with embedding generation section
  - **Acceptance**: README includes embedding pipeline usage, performance metrics, troubleshooting guide
  - **Files**: `README.md`
  - **Dependencies**: Requires T016

## Dependencies and Execution Order

### Phase 3.2 Dependencies (Tests First - TDD):
- **T004-T007** must be written and must FAIL before any implementation
- These can run in parallel [P] since they test different contracts

### Phase 3.3 Dependencies (Integration Tests):
- **T008** requires T004-T007 to fail first (establishes contracts)
- **T009-T011** require T008 (sequential BigQuery setup)

### Phase 3.4 Dependencies (Implementation):
- **T012** and **T013** can run in parallel [P] (different classes)
- **T014** requires T012, T013 (uses both classes)
- **T015** can run in parallel with T012-T013 [P] (different file)
- **T016** requires T014, T015 (final integration)

### Phase 3.5 Dependencies (Polish):
- **T017** requires T012; **T018** requires T013 (can run in parallel [P])
- **T019** requires T016 (end-to-end performance testing)
- **T020** can run in parallel with T017-T019 [P] (documentation)

## Parallel Execution Examples

### Phase 3.2 (Contract Tests):
```bash
# Launch T004-T007 together:
Task: "BigQuery schema validation contract test in tests/contract/test_bigquery_embedding_schema.py"
Task: "Embedding API contract test in tests/contract/test_embedding_pipeline_api.py"  
Task: "Vector search functionality contract test in tests/contract/test_vector_search_contract.py"
Task: "Cache persistence contract test in tests/contract/test_embedding_cache_contract.py"
```

### Phase 3.4 (Core Implementation):
```bash
# Launch T012, T013, T015 together:
Task: "EmbeddingCache class implementation in pipeline/embedding.py"
Task: "EmbeddingGenerator class implementation in pipeline/embedding.py"
Task: "BigQuery schema enhancements in src/janapada_memory/schema_manager.py"
```

### Phase 3.5 (Polish):
```bash  
# Launch T017, T018, T020 together:
Task: "Unit tests for EmbeddingCache in tests/unit/test_embedding_cache.py"
Task: "Unit tests for EmbeddingGenerator in tests/unit/test_embedding_generator.py"
Task: "Update README.md documentation with embedding generation section"
```

## Validation Checklist
*GATE: All must be checked before considering tasks complete*

- [x] All contracts (T004-T007) have corresponding implementation tasks (T012-T016)
- [x] All entities (EmbeddingRecord, CacheEntry, ProcessingStats) covered in implementation
- [x] All tests (T004-T011, T017-T019) come before implementation (T012-T016)
- [x] Parallel tasks [P] are truly independent (different files, no shared state)
- [x] Each task specifies exact file path and acceptance criteria
- [x] BigQuery schema contract (T004) maps to schema implementation (T015)
- [x] API contract (T005) maps to pipeline implementation (T014, T016)
- [x] Cache contract (T007) maps to cache implementation (T012)
- [x] Vector search contract (T006) validated by integration test (T011)

## Notes
- **TDD Critical**: Contract tests T004-T007 MUST fail before implementation begins
- **File Conflicts**: T012-T014 modify same file sequentially (pipeline/embedding.py)
- **BigQuery Dependencies**: Integration tests T008-T011 require proper BigQuery setup
- **Performance Goals**: T019 validates 100-500 chunks/minute, 70-90% cache hit rate
- **Cost Optimization**: Focus on cache efficiency and API call reduction
- **Idempotency**: All tests must validate re-run behavior doesn't duplicate data