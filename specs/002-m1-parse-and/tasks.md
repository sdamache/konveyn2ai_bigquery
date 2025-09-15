# Tasks: M1 Multi-Source Ingestion for BigQuery

**Input**: Design documents from `/specs/002-m1-parse-and/`
**Prerequisites**: plan.md ✓, research.md ✓, data-model.md ✓, contracts/ ✓

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → SUCCESS - Python 3.11, BigQuery Storage Write API, 5 parser libraries
2. Load optional design documents:
   → data-model.md: source_metadata, source_metadata_errors, ingestion_log
   → contracts/: bigquery-ddl.sql, parser-interfaces.py
   → research.md: Storage Write API, semantic chunking, hybrid parsers
3. Generate tasks by category:
   → Setup: project init, BigQuery tables, dependencies
   → Tests: contract tests, integration tests (TDD enforced)
   → Core: 5 parser libraries, chunking utilities, BigQuery writer
   → Integration: Make targets, CLI, error handling
   → Polish: performance validation, documentation
4. Apply task rules:
   → Parser libraries = different files = [P]
   → Common utilities = shared dependencies = sequential
   → Tests before implementation (TDD)
5. Number tasks T001-T028
6. SUCCESS - tasks ready for execution
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
Single project structure:
- `src/` - Source code at repository root
- `tests/` - Tests at repository root
- Parser libraries in `src/ingest/{k8s,fastapi,cobol,irs,mumps}/`

## Phase 3.1: Setup
- [x] T001 Create project structure: `src/ingest/{k8s,fastapi,cobol,irs,mumps}/`, `src/common/`, `tests/{contract,integration,unit}/`
- [x] T002 Initialize Python project with dependencies: google-cloud-bigquery, pyyaml, kubernetes, libcst, ulid-py, python-cobol
- [x] T003 [P] Configure linting: ruff, mypy, black configuration files
- [x] T004 Create BigQuery dataset and tables from `specs/002-m1-parse-and/contracts/bigquery-ddl.sql`

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Contract Tests (BigQuery Schema Validation)
- [x] T005 [P] Contract test source_metadata table schema in `tests/contract/test_source_metadata_schema.py`
- [x] T006 [P] Contract test source_metadata_errors table schema in `tests/contract/test_source_errors_schema.py`
- [x] T007 [P] Contract test ingestion_log table schema in `tests/contract/test_ingestion_log_schema.py`

### Parser Contract Tests
- [x] T008 [P] Contract test Kubernetes parser interface in `tests/contract/test_k8s_parser_contract.py`
- [x] T009 [P] Contract test FastAPI parser interface in `tests/contract/test_fastapi_parser_contract.py`
- [x] T010 [P] Contract test COBOL parser interface in `tests/contract/test_cobol_parser_contract.py`
- [x] T011 [P] Contract test IRS parser interface in `tests/contract/test_irs_parser_contract.py`
- [x] T012 [P] Contract test MUMPS parser interface in `tests/contract/test_mumps_parser_contract.py`

### Integration Tests (End-to-End Scenarios)
- [x] T013 [P] Integration test K8s manifest ingestion to BigQuery in `tests/integration/test_k8s_ingestion.py`
- [x] T014 [P] Integration test FastAPI project ingestion in `tests/integration/test_fastapi_ingestion.py`
- [x] T015 [P] Integration test COBOL copybook ingestion in `tests/integration/test_cobol_ingestion.py`
- [x] T016 [P] Integration test IRS layout ingestion in `tests/integration/test_irs_ingestion.py`
- [x] T017 [P] Integration test MUMPS dictionary ingestion in `tests/integration/test_mumps_ingestion.py`
- [x] T018 Integration test idempotency (repeated ingestion) in `tests/integration/test_idempotency.py`

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Common Utilities (Dependencies for all parsers)
- [x] T019 Content chunking utility in `src/common/chunking.py` with source-aware strategies
- [x] T020 Deterministic ID generation in `src/common/ids.py` using SHA256 and semantic paths
- [x] T021 Content normalization utility in `src/common/normalize.py` for consistent hashing
- [x] T022 BigQuery Storage Write API client in `src/common/bq_writer.py` with batch processing

### Parser Libraries (Can run in parallel - different files)
- [x] T023 [P] Kubernetes parser in `src/ingest/k8s/parser.py` using kr8s + PyYAML
- [x] T024 [P] FastAPI parser in `src/ingest/fastapi/parser.py` using AST + OpenAPI introspection
- [x] T025 [P] COBOL parser in `src/ingest/cobol/parser.py` using python-cobol + regex
- [x] T026 [P] IRS parser in `src/ingest/irs/parser.py` using regex + struct module
- [x] T027 [P] MUMPS parser in `src/ingest/mumps/parser.py` using custom regex + pyGTM validation

## Phase 3.4: Integration

### CLI and Make Targets
- [ ] T028 Create Makefile with targets: setup, ingest_k8s, ingest_fastapi, ingest_cobol, ingest_irs, ingest_mumps
- [ ] T029 CLI entrypoints in `src/cli/` with argument parsing, dry-run support, output formatting
- [ ] T030 Error handling and logging infrastructure in `src/common/logging.py` with structured JSON logs

### BigQuery Integration
- [ ] T031 Connect parsers to BigQuery writer with batch processing and retry logic
- [ ] T032 Implement idempotent upsert logic with content hash comparison
- [ ] T033 Add ingestion run tracking to `ingestion_log` table

## Phase 3.5: Polish

### Performance and Validation
- [ ] T034 [P] Unit tests for chunking strategies in `tests/unit/test_chunking.py`
- [ ] T035 [P] Unit tests for ID generation in `tests/unit/test_ids.py`
- [ ] T036 [P] Unit tests for BigQuery writer in `tests/unit/test_bq_writer.py`
- [ ] T037 Performance validation: process 100+ files per source type within 5 minutes
- [ ] T038 Data quality validation: verify ≥100 rows per source in BigQuery
- [ ] T039 [P] Update project documentation in `README.md` with usage examples
- [ ] T040 Generate sample data sets for each source type for demo purposes

## Dependencies

### Critical Path
1. **Setup** (T001-T004) → **Tests** (T005-T018) → **Common Utilities** (T019-T022) → **Parser Libraries** (T023-T027) → **Integration** (T028-T033) → **Polish** (T034-T040)

### Blocking Dependencies
- T019-T022 (common utilities) block T023-T027 (parsers)
- T023-T027 (parsers) block T028-T033 (integration)
- T005-T018 (tests) must FAIL before starting T019-T027 (TDD enforcement)

### Independent Work
- Parser libraries T023-T027 can run in parallel (different files)
- Contract tests T005-T012 can run in parallel (different files)
- Integration tests T013-T017 can run in parallel (independent test scenarios)
- Unit tests T034-T036 can run in parallel (different modules)

## Parallel Execution Examples

### Phase 3.2: Contract Tests
```bash
# Launch T008-T012 together (parser contract tests):
Task: "Contract test Kubernetes parser interface in tests/contract/test_k8s_parser_contract.py"
Task: "Contract test FastAPI parser interface in tests/contract/test_fastapi_parser_contract.py"
Task: "Contract test COBOL parser interface in tests/contract/test_cobol_parser_contract.py"
Task: "Contract test IRS parser interface in tests/contract/test_irs_parser_contract.py"
Task: "Contract test MUMPS parser interface in tests/contract/test_mumps_parser_contract.py"
```

### Phase 3.2: Integration Tests
```bash
# Launch T013-T017 together (end-to-end ingestion tests):
Task: "Integration test K8s manifest ingestion to BigQuery in tests/integration/test_k8s_ingestion.py"
Task: "Integration test FastAPI project ingestion in tests/integration/test_fastapi_ingestion.py"
Task: "Integration test COBOL copybook ingestion in tests/integration/test_cobol_ingestion.py"
Task: "Integration test IRS layout ingestion in tests/integration/test_irs_ingestion.py"
Task: "Integration test MUMPS dictionary ingestion in tests/integration/test_mumps_ingestion.py"
```

### Phase 3.3: Parser Libraries
```bash
# Launch T023-T027 together (parser implementations):
Task: "Kubernetes parser in src/ingest/k8s/parser.py using kr8s + PyYAML"
Task: "FastAPI parser in src/ingest/fastapi/parser.py using AST + OpenAPI introspection"
Task: "COBOL parser in src/ingest/cobol/parser.py using python-cobol + regex"
Task: "IRS parser in src/ingest/irs/parser.py using regex + struct module"
Task: "MUMPS parser in src/ingest/mumps/parser.py using custom regex + pyGTM validation"
```

## Notes
- **[P] tasks** = different files, no shared dependencies, can run in parallel
- **Sequential tasks** = shared files or blocking dependencies
- **TDD Enforcement**: All tests T005-T018 MUST fail before implementing T019-T027
- **Commit strategy**: Commit after each task completion
- **Error handling**: Continue-on-error for parsing, log all errors to `source_metadata_errors`

## Task Generation Rules Applied

1. **From Contracts**:
   - bigquery-ddl.sql → schema validation tests (T005-T007)
   - parser-interfaces.py → contract tests per parser (T008-T012)

2. **From Data Model**:
   - source_metadata entity → BigQuery writer (T022)
   - source_metadata_errors entity → error logging (T030)
   - ingestion_log entity → run tracking (T033)

3. **From User Stories (quickstart.md)**:
   - Each "make ingest_*" command → integration test (T013-T017)
   - Idempotency requirement → dedicated test (T018)
   - Performance goals → validation task (T037-T038)

4. **From Research Decisions**:
   - Storage Write API → BigQuery client implementation (T022)
   - Semantic chunking → chunking utility (T019)
   - Hybrid parsers → parser library tasks (T023-T027)

## Validation Checklist ✓

- [x] All contracts have corresponding tests (T005-T012)
- [x] All entities have implementation tasks (T019-T027, T030-T033)
- [x] All tests come before implementation (T005-T018 before T019-T027)
- [x] Parallel tasks are truly independent (different files, no shared state)
- [x] Each task specifies exact file path
- [x] No [P] task modifies same file as another [P] task
- [x] TDD order enforced: Tests fail → Implementation → Tests pass
- [x] Constitutional compliance: Library-first, real BigQuery (no mocks), structured logging

**Total Tasks**: 40 tasks ordered by dependency with 15 parallel execution opportunities