# Tasks: BigQuery Memory Adapter (M2 Integration)

**Input**: Design documents from `/specs/004-bigquery-memory-adapter/`
**Prerequisites**: plan.md, research.md, data-model.md, contracts/, quickstart.md completed

## Execution Flow (main)
```
0. Review latest spec, plan, tasks.md, CLAUDE guidance, and open PR notes
   → Building on M1/M2 artifacts, preserving vector_index contract
1. Load plan.md from feature directory
   → Tech stack: Python 3.11+, google-cloud-bigquery, pytest
   → Structure: Single project extending src/janapada-memory/
2. Load design documents:
   → data-model.md: VectorSearchConfig, VectorSearchResult, BigQueryConnection entities
   → contracts/: vector_index_contract.py, bigquery_integration_contract.py
   → research.md: BigQuery VECTOR_SEARCH patterns, ADC authentication
3. Generate tasks by category:
   → Setup: BigQuery client, configuration loading
   → Tests: Contract tests MUST FAIL first (RED phase)
   → Core: BigQueryVectorIndex implementation, fallback logic
   → Integration: Svami orchestrator compatibility
   → Polish: Performance validation, documentation
4. Apply task rules:
   → Contract tests [P], implementation sequential
   → Real BigQuery dependencies, mock only for unit tests with justification
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths and acceptance criteria

## Phase 3.1: Setup & Environment

- [x] T001 Create BigQuery configuration module in `src/janapada_memory/config/bigquery_config.py`
  **Acceptance**: Configuration loads from environment variables (GOOGLE_CLOUD_PROJECT, BIGQUERY_DATASET_ID) with defaults, validates required fields, handles missing credentials gracefully.

- [x] T002 [P] Update project requirements in `requirements.txt` to include `google-cloud-bigquery>=3.0.0`
  **Acceptance**: `pip install -r requirements.txt` succeeds, `import google.cloud.bigquery` works in Python shell.

- [x] T003 [P] Extend `.env.example` with BigQuery configuration variables
  **Acceptance**: File contains GOOGLE_CLOUD_PROJECT, BIGQUERY_DATASET_ID, BIGQUERY_TABLE_PREFIX with example values and documentation comments.

## Phase 3.2: Contract Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

- [x] T004 [P] Verify vector_index_contract.py tests FAIL in `tests/contract/test_bigquery_vector_index_contract.py`
  **Acceptance**: `pytest specs/004-bigquery-memory-adapter/contracts/vector_index_contract.py -v` shows ImportError or NotImplementedError for BigQueryVectorIndex class.

- [x] T005 [P] Verify bigquery_integration_contract.py tests FAIL in `tests/integration/test_bigquery_integration.py`
  **Acceptance**: `pytest specs/004-bigquery-memory-adapter/contracts/bigquery_integration_contract.py -v` shows table/connection failures or ImportError for implementation.

- [x] T006 [P] Create fallback behavior contract test in `tests/contract/test_fallback_contract.py`
  **Acceptance**: Test verifies similarity_search returns local results when BigQuery client raises NotFound/Forbidden exceptions. Test MUST FAIL initially.

- [x] T007 [P] Create performance contract test in `tests/contract/test_performance_contract.py`
  **Acceptance**: Test verifies similarity_search completes within 500ms timeout. Test MUST FAIL initially due to missing implementation.

## Phase 3.3: Core Implementation (ONLY after tests are failing)

- [ ] T008 [P] Create VectorSearchConfig data class in `src/janapada_memory/models/vector_search_config.py`
  **Acceptance**: Class validates project_id format, dataset_id accessibility, top_k bounds (1-1000), distance_type enum (COSINE, DOT_PRODUCT). Immutable dataclass with validation.

- [ ] T009 [P] Create VectorSearchResult data class in `src/janapada_memory/models/vector_search_result.py`
  **Acceptance**: Class contains chunk_id (str), distance (float), metadata (Dict), source (str) fields. Implements comparison for distance ordering.

- [ ] T010 Create BigQueryConnection manager in `src/janapada_memory/connections/bigquery_connection.py`
  **Acceptance**: Manages BigQuery client lifecycle, ADC authentication, connection health checks. Handles authentication failures gracefully with structured logging.

- [ ] T011 Create BigQueryAdapter SQL query builder in `src/janapada_memory/adapters/bigquery_adapter.py`
  **Acceptance**: Constructs parameterized VECTOR_SEARCH queries, handles vector encoding for BigQuery, parses result rows into VectorSearchResult objects.

- [ ] T012 Create LocalVectorIndex fallback in `src/janapada_memory/fallback/local_vector_index.py`
  **Acceptance**: Implements VectorIndex interface using existing vector utilities, provides approximate similarity search, maintains result ordering contract.

- [ ] T013 Create BigQueryVectorIndex main implementation in `src/janapada_memory/bigquery_vector_index.py`
  **Acceptance**: Implements VectorIndex interface, delegates to BigQueryAdapter for primary search, falls back to LocalVectorIndex on errors, logs fallback activation with correlation IDs.

## Phase 3.4: Integration & Error Handling

- [ ] T014 Implement similarity_search method with fallback logic in `src/janapada_memory/bigquery_vector_index.py`
  **Acceptance**: Method tries BigQuery first, catches NotFound/Forbidden/ConnectionError, falls back to local index, preserves interface contract, returns ordered results.

- [ ] T015 Add structured logging and observability in `src/janapada_memory/bigquery_vector_index.py`
  **Acceptance**: Logs BigQuery job IDs, query latency, fallback activation, correlation IDs. JSON format with error context and remediation links.

- [ ] T016 Implement add_vectors and remove_vector interface methods in `src/janapada_memory/bigquery_vector_index.py`
  **Acceptance**: Methods maintain interface contract, delegate to both BigQuery (if available) and local index, handle partial failures gracefully.

- [ ] T017 Add configuration validation and error handling in `src/janapada_memory/bigquery_vector_index.py`
  **Acceptance**: Validates BigQuery project/dataset access on initialization, provides clear error messages for auth/permission issues, falls back gracefully.

## Phase 3.5: Integration with Existing System

- [ ] T018 Update Janapada memory service to use BigQueryVectorIndex in `src/janapada_memory/__init__.py`
  **Acceptance**: Service can be instantiated with BigQuery backend, maintains existing API surface, backward compatible with current Svami integration.

- [ ] T019 Create Svami integration smoke test in `tests/integration/test_svami_bigquery_integration.py`
  **Acceptance**: End-to-end test from Svami orchestrator through Janapada to BigQuery and back, verifies question answering flow works with BigQuery backend.

- [ ] T020 Update make setup to validate BigQuery table schema in `Makefile`
  **Acceptance**: `make setup` verifies source_embeddings table exists, has correct VECTOR column, proper permissions. Idempotent operation.

## Phase 3.6: Performance & Polish

- [ ] T021 [P] Add unit tests for configuration validation in `tests/unit/test_bigquery_config.py`
  **Acceptance**: Tests cover invalid project IDs, missing datasets, authentication failures, environment variable precedence. Mock BigQuery client with justification (avoiding real auth in unit tests).

- [ ] T022 [P] Add unit tests for query construction in `tests/unit/test_bigquery_adapter.py`
  **Acceptance**: Tests verify SQL query syntax, parameter encoding, result parsing, vector dimension validation. Mock BigQuery responses with justification.

- [ ] T023 [P] Add unit tests for fallback logic in `tests/unit/test_fallback_behavior.py`
  **Acceptance**: Tests verify fallback activation triggers, error propagation, logging behavior, interface preservation. Mock BigQuery exceptions with justification.

- [ ] T024 Create performance benchmark in `tests/performance/test_bigquery_performance.py`
  **Acceptance**: Measures similarity_search latency against real BigQuery, validates <500ms target, compares BigQuery vs local performance, records baseline metrics.

- [ ] T025 [P] Update documentation in `README.md` and `CLAUDE.md`
  **Acceptance**: Documents BigQuery configuration, environment setup, troubleshooting guide, performance characteristics. Includes quickstart examples.

- [ ] T026 [P] Record contract stabilization notes in `docs/contract_stabilization_notes.md`
  **Acceptance**: Documents "green" test run snapshot, BigQuery schema requirements, authentication setup, known issues and workarounds.

## Dependencies

**Critical Path**:
- Setup (T001-T003) → Contract Tests (T004-T007) → Core Models (T008-T009) → Connections (T010) → Adapters (T011-T012) → Main Implementation (T013) → Integration (T014-T020) → Polish (T021-T026)

**Blocking Relationships**:
- T004-T007 block ALL implementation tasks (TDD requirement)
- T008-T009 block T013 (data models required)
- T010-T012 block T013 (dependencies required)
- T013 blocks T014-T017 (main class required)
- T018 blocks T019 (service integration required)

## Parallel Execution Examples

### Phase 3.2 (Contract Tests - All MUST FAIL)
```bash
# Launch contract tests together - all should fail initially
pytest specs/004-bigquery-memory-adapter/contracts/vector_index_contract.py -v &
pytest specs/004-bigquery-memory-adapter/contracts/bigquery_integration_contract.py -v &
pytest tests/contract/test_fallback_contract.py -v &
pytest tests/contract/test_performance_contract.py -v &
wait
```

### Phase 3.3 (Models - Independent Files)
```bash
# Create data models in parallel
Task: "Create VectorSearchConfig data class in src/janapada_memory/models/vector_search_config.py"
Task: "Create VectorSearchResult data class in src/janapada_memory/models/vector_search_result.py"
```

### Phase 3.6 (Unit Tests - Independent Files)
```bash
# Unit tests can run in parallel
pytest tests/unit/test_bigquery_config.py -v &
pytest tests/unit/test_bigquery_adapter.py -v &
pytest tests/unit/test_fallback_behavior.py -v &
wait
```

## Risk Mitigation Tasks

### INFORMATION_SCHEMA Authentication Scope
- **Risk**: BigQuery INFORMATION_SCHEMA queries may fail across different auth scopes
- **Mitigation**: T017 includes auth scope validation and graceful degradation

### SDK Version Compatibility
- **Risk**: Mixed google-cloud-bigquery return types across versions
- **Mitigation**: T002 pins specific version, T022 tests type coercion

### Environment Validation
- **Risk**: Brittle environment validation error messages
- **Mitigation**: T001 includes comprehensive error handling and user-friendly messages

## Test Plan Execution

### BigQuery Integration Mode
```bash
# Run integration tests with real BigQuery
export GOOGLE_CLOUD_PROJECT=konveyn2ai
export BIGQUERY_DATASET_ID=source_ingestion
pytest specs/004-bigquery-memory-adapter/contracts/bigquery_integration_contract.py -v
pytest tests/integration/test_bigquery_integration.py -v
```

### Unit Test Coverage
```bash
# Run unit tests with mocks (justified usage)
pytest tests/unit/test_bigquery_config.py tests/unit/test_bigquery_adapter.py tests/unit/test_fallback_behavior.py -v --cov=src.janapada_memory
```

### Contract Stabilization
```bash
# Record green run for stabilization
pytest specs/004-bigquery-memory-adapter/contracts/ -v > docs/contract_stabilization_notes.md
```

## Notes
- **Constitutional Compliance**: RED-GREEN-REFACTOR enforced, real BigQuery dependencies in integration tests
- **Interface Preservation**: Existing vector_index contract MUST remain unchanged
- **Performance Target**: <500ms similarity search latency validated in T024
- **Fallback Requirement**: Local in-memory index provides graceful degradation
- **Observability**: Structured logging with correlation IDs and BigQuery job references

## Validation Checklist
*GATE: Checked before task execution*

- [x] All contracts have corresponding tests (T004-T007)
- [x] All entities have model tasks (T008-T009)
- [x] All tests come before implementation (T004-T007 → T008+)
- [x] Parallel tasks truly independent ([P] marked appropriately)
- [x] Each task specifies exact file path and acceptance criteria
- [x] No task modifies same file as another [P] task