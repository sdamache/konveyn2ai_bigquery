# Implementation Plan: M1 - Parse and Ingest Multi-Source Artifacts

**Branch**: `002-m1-parse-and` | **Date**: 2025-09-14 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-m1-parse-and/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → SUCCESS - spec loaded with clear requirements
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Project Type: Single (data ingestion pipeline)
   → Structure Decision: DEFAULT (Option 1) - focused on data processing
3. Evaluate Constitution Check section below
   → Document simplicity approach for ingestion libraries
   → Update Progress Tracking: Initial Constitution Check
4. Execute Phase 0 → research.md
   → Research BigQuery Storage Write API, chunking patterns, parser libraries
5. Execute Phase 1 → contracts, data-model.md, quickstart.md, CLAUDE.md
6. Re-evaluate Constitution Check section
   → Validate library-first approach maintained
   → Update Progress Tracking: Post-Design Constitution Check
7. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
8. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Primary requirement: Ingest knowledge artifacts from 5 source families (Kubernetes, FastAPI, COBOL, IRS, MUMPS) into unified BigQuery schema with ≥100 rows per source, idempotent writes, and rich metadata for downstream AI processing and gap analysis.

Technical approach: Library-first architecture with independent parsers, common chunking utilities, BigQuery Storage Write API, and deterministic content hashing for idempotency.

## Technical Context
**Language/Version**: Python 3.11 (uv for packaging)
**Primary Dependencies**: google-cloud-bigquery, pyyaml, kubernetes, libcst, ulid-py
**Storage**: BigQuery (Storage Write API preferred, load jobs fallback)
**Testing**: pytest with real BigQuery temp datasets (no mocks for core functionality)
**Target Platform**: Linux server / containerized workloads
**Project Type**: single - data processing pipeline
**Performance Goals**: Process 1000+ artifacts/minute, <16KB content per chunk
**Constraints**: <200ms p95 per chunk, idempotent upserts, deterministic IDs
**Scale/Scope**: 5 source types, ≥100 rows per type, shared schema compliance

User guidance incorporated: "please follow the plan accordingly instead of jumping to implementation" - ensuring systematic approach through proper research and design phases before any coding begins.

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Simplicity**:
- Projects: 1 (ingestion pipeline with 5 parser libraries)
- Using framework directly? YES (BigQuery client, pyyaml, kubernetes client)
- Single data model? YES (source_metadata unified schema)
- Avoiding patterns? YES (direct parsers, no unnecessary abstractions)

**Architecture**:
- EVERY feature as library? YES - 5 parser libraries + common utilities
- Libraries listed: k8s_parser, fastapi_parser, cobol_parser, irs_parser, mumps_parser, common_chunking, bq_writer
- CLI per library: YES (ingest_k8s, ingest_fastapi, etc. with --help/--version/--dry-run)
- Library docs: llms.txt format planned? YES

**Testing (NON-NEGOTIABLE)**:
- RED-GREEN-Refactor cycle enforced? YES (tests written first, must fail)
- Git commits show tests before implementation? YES (strict TDD)
- Order: Contract→Integration→E2E→Unit strictly followed? YES
- Real dependencies used? YES (actual BigQuery, no mocks for core functionality)
- Integration tests for: new libraries, contract changes, shared schemas? YES
- FORBIDDEN: Implementation before test, skipping RED phase? ENFORCED

**Observability**:
- Structured logging included? YES (JSON logs with correlation IDs)
- Frontend logs → backend? N/A (server-side only)
- Error context sufficient? YES (source file, line numbers, error classification)

**Versioning**:
- Version number assigned? YES (1.0.0 for M1 milestone)
- BUILD increments on every change? YES
- Breaking changes handled? YES (schema versioning, migration tests)

## Project Structure

### Documentation (this feature)
```
specs/002-m1-parse-and/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Option 1: Single project (SELECTED)
src/
├── models/              # BigQuery table schemas, validation models
├── services/           # Parser services for each source type
│   ├── k8s/
│   ├── fastapi/
│   ├── cobol/
│   ├── irs/
│   └── mumps/
├── cli/                # Make targets and CLI entrypoints
└── lib/                # Common utilities (chunking, IDs, BQ writer)

tests/
├── contract/           # BigQuery schema validation tests
├── integration/        # End-to-end ingestion tests
└── unit/              # Parser-specific unit tests
```

**Structure Decision**: DEFAULT (Option 1) - focused data processing pipeline, no web/mobile components

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - BigQuery Storage Write API vs Load Jobs performance comparison
   - Optimal chunking strategies for each source type (token limits, overlap)
   - Parser library choices (AST vs regex for COBOL/IRS)
   - Deterministic ID generation patterns for content hashing

2. **Generate and dispatch research agents**:
   ```
   Task: "Research BigQuery Storage Write API best practices for high-throughput ingestion"
   Task: "Find optimal chunking strategies for code and structured data artifacts"
   Task: "Research Python AST parsing vs regex for COBOL copybook analysis"
   Task: "Find Kubernetes client patterns for manifest extraction and processing"
   Task: "Research FastAPI OpenAPI spec parsing with Pydantic model binding"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all technical decisions documented

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - source_metadata table schema with source-specific fields
   - ingestion_log entity for run tracking
   - artifact_hash model for idempotency
   - Validation rules from functional requirements

2. **Generate API contracts** from functional requirements:
   - BigQuery table DDL schemas
   - Parser interface contracts (input → normalized output)
   - CLI interface specifications
   - Output schemas to `/contracts/`

3. **Generate contract tests** from contracts:
   - BigQuery schema validation tests
   - Parser output format tests
   - CLI interface tests (must fail initially)

4. **Extract test scenarios** from user stories:
   - Each acceptance scenario → integration test
   - Quickstart validation = end-to-end test workflow

5. **Update agent file incrementally** (O(1) operation):
   - Run `/scripts/bash/update-agent-context.sh claude`
   - Add BigQuery, parsing libraries, testing patterns
   - Preserve existing hackathon context
   - Keep under 150 lines for token efficiency

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, CLAUDE.md

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- Each BigQuery schema → DDL creation task [P]
- Each parser contract → parser library task [P]
- Each acceptance scenario → integration test task
- Common utilities (chunking, IDs, BQ writer) → shared library tasks

**Ordering Strategy**:
- TDD order: Contract tests → Implementation tests → Implementation
- Dependency order: Common utilities → Parsers → CLI → Integration tests
- Mark [P] for parallel execution (independent parser libraries)

**Estimated Output**: 20-25 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (execute tasks.md following constitutional principles)
**Phase 5**: Validation (run tests, execute quickstart.md, validate ≥100 rows per source)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

No constitutional violations identified. Design maintains simplicity with:
- Single project structure
- Library-first architecture (5 parser libraries + utilities)
- Direct framework usage (no unnecessary abstractions)
- Real dependency testing (actual BigQuery)

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [x] Phase 3: Tasks generated (/tasks command)
- [x] Phase 3.1: Setup complete (T001-T004)
- [x] Phase 3.2: Integration tests complete (T005-T018)
- [x] Phase 3.3: Common utilities complete (T019-T022)
- [x] Phase 3.3: Parser libraries complete (T023-T027)
- [ ] Phase 3.4: Integration (T028-T033)
- [ ] Phase 3.5: Polish (T034-T040)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (none required)

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*