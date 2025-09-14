# Implementation Plan: BigQuery Vector Backend

**Branch**: `001-bigquery-vector-backend` | **Date**: 2025-09-14 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-bigquery-vector-backend/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
4. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
5. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, or `GEMINI.md` for Gemini CLI).
6. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
7. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
8. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Replace KonveyN2AI's Vertex AI vector store with BigQuery VECTOR tables, implementing a three-table schema (source_metadata, source_embeddings, gap_metrics) with 768-dimensional embeddings. Maintain stable vector_index interface while enabling BigQuery Studio and Colab compatibility.

## Technical Context
**Language/Version**: Python 3.11  
**Primary Dependencies**: google-cloud-bigquery, google-cloud-aiplatform, streamlit, numpy, pandas  
**Storage**: BigQuery (semantic_gap_detector dataset with 3 tables)  
**Testing**: pytest (following existing test patterns)  
**Target Platform**: Google Cloud Platform + local development  
**Project Type**: single (backend vector processing system)  
**Performance Goals**: <200ms vector search latency, support for 150 approximate neighbors  
**Constraints**: 768-dimensional vectors, BigQuery slot limits, maintain existing API compatibility  
**Scale/Scope**: Migration from existing 3072-dim Vertex AI system, ~1000s of vectors initially

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Simplicity**:
- Projects: 1 (vector backend only)
- Using framework directly? Yes (BigQuery Python client directly)
- Single data model? Yes (three tables with clear relationships)
- Avoiding patterns? Yes (no unnecessary abstractions)

**Architecture**:
- EVERY feature as library? Yes (bigquery_vector_store library)
- Libraries listed: bigquery_vector_store (vector operations), schema_manager (table management)
- CLI per library: bigquery-vector --help, schema-manager --help
- Library docs: llms.txt format planned? Yes

**Testing (NON-NEGOTIABLE)**:
- RED-GREEN-Refactor cycle enforced? Yes
- Git commits show tests before implementation? Yes
- Order: Contract→Integration→E2E→Unit strictly followed? Yes
- Real dependencies used? Yes (actual BigQuery, not mocks)
- Integration tests for: new libraries, contract changes, shared schemas? Yes
- FORBIDDEN: Implementation before test, skipping RED phase

**Observability**:
- Structured logging included? Yes
- Frontend logs → backend? N/A (backend only)
- Error context sufficient? Yes

**Versioning**:
- Version number assigned? 1.0.0
- BUILD increments on every change? Yes
- Breaking changes handled? Yes (parallel backend support during migration)

## Project Structure

### Documentation (this feature)
```
specs/001-bigquery-vector-backend/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/
```

**Structure Decision**: Option 1 (single project) - This is a backend vector processing system

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - BigQuery VECTOR_SEARCH function capabilities and limitations
   - Migration strategy from 3072-dim to 768-dim embeddings
   - BigQuery authentication patterns for multi-environment deployment
   - Performance characteristics vs Vertex AI

2. **Generate and dispatch research agents**:
   ```
   Task: "Research BigQuery VECTOR_SEARCH for semantic similarity search"
   Task: "Find best practices for BigQuery authentication in Python applications"
   Task: "Research vector dimension migration strategies and re-embedding approaches"
   Task: "Analyze BigQuery slot management and query optimization for vector operations"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all technical decisions documented

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - source_metadata (chunk_id, source, artifact_type, text_content, kind, api_path, record_name)
   - source_embeddings (chunk_id, embedding[768], created_at)  
   - gap_metrics (analysis_id, chunk_id, metric_type, metric_value, metadata)

2. **Generate API contracts** from functional requirements:
   - vector_store.insert_embedding(chunk_id, text, embedding)
   - vector_store.search_similar(query_embedding, limit, threshold)
   - schema_manager.create_tables()
   - schema_manager.validate_schema()
   - Output schemas to `/contracts/`

3. **Generate contract tests** from contracts:
   - Test BigQuery table creation and schema validation
   - Test vector insertion and retrieval operations
   - Test similarity search with various parameters
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:
   - End-to-end: Insert vectors → Search similar → Validate results
   - Migration: Export from Vertex → Import to BigQuery → Verify consistency
   - Setup: `make setup` → Tables created → Ready for use

5. **Update agent file incrementally**:
   - Update CLAUDE.md with BigQuery vector backend context
   - Add new dependencies and commands
   - Document migration procedures

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, CLAUDE.md

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `/templates/tasks-template.md` as base
- Each contract → contract test task [P]
- Each table schema → model creation task [P]
- Vector operations → service implementation tasks
- Migration utilities → data handling tasks
- Makefile targets → setup automation tasks

**Ordering Strategy**:
- TDD order: Schema tests → Table creation → Vector operation tests → Implementation
- Dependency order: Models → Services → CLI → Integration
- Mark [P] for parallel execution (independent components)

**Estimated Output**: 18-22 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*No constitutional violations identified*

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (none identified)

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*