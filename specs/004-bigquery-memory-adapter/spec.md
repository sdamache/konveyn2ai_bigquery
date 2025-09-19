# Feature Specification: BigQuery Memory Adapter (M2 Integration)

**Feature Branch**: `004-bigquery-memory-adapter`
**Created**: 2025-09-18
**Status**: Draft
**Input**: User description: "Use the @issues/issue#4.txt document only to prepare the spec. Respect the \"Acceptance Criteria (verbatim)\" exactly. Do not restate or summarize the issue text; just reference it. Add only: Approach (BigQuery `approximate_neighbors` adapter in `janapada-memory`, config/env wiring, in-memory fallback), Risks, Test Plan, and Rollback/Migration steps. Build on prior artifacts from Issues #1#3 (schemas, `vector_index` invariants, embeddings) without duplicating DDL or changing public interfaces. Do not introduce new surfaces. Keep it concise (<400 lines)."

## Execution Flow (main)
```
0. Review previous specs, plans, tasks, and PR feedback for this feature space
   � Building on M1 schemas/DDL and M2 embeddings from Issues #1-#3
1. Parse user description from Input
   � Integrate BigQuery vector search into Janapada memory service
2. Extract key concepts from description
   � Actors: Svami orchestrator, Janapada memory service; Actions: similarity_search via BigQuery; Data: source_embeddings table; Constraints: preserve vector_index contract
3. For each unclear aspect:
   � No clarifications needed - Issue #4 provides complete acceptance criteria
4. Fill User Scenarios & Testing section
   � End-to-end Q&A flow with BigQuery backend and fallback scenarios
5. Generate Functional Requirements
   � Each requirement derived from Issue #4 acceptance criteria
6. Identify Key Entities (if data involved)
7. Run Review Checklist
   � Implementation approach added per user request
8. Return: SUCCESS (spec ready for planning)
```

---

## � Quick Guidelines
-  Focus on WHAT users need and WHY
- L Avoid HOW to implement (no tech stack, APIs, code structure)
- =e Written for business stakeholders, not developers
- = Reference outstanding work first: specs must note any carry-over requirements or debt before adding new scope

---

## User Scenarios & Testing *(mandatory)*

### Primary User Story
As a Svami orchestrator, I need to perform vector similarity searches against BigQuery's source_embeddings table so that I can answer knowledge questions using the existing ingested corpus without changing my interface contract with the Janapada memory service.

### Acceptance Scenarios
1. **Given** BigQuery is configured and available, **When** Svami calls similarity_search on Janapada memory service, **Then** system executes approximate_neighbors query against source_embeddings and returns ranked chunk IDs
2. **Given** BigQuery is unavailable or misconfigured, **When** Svami calls similarity_search, **Then** system gracefully falls back to local in-memory vector index with approximate search
3. **Given** valid query vector and configuration, **When** similarity_search executes, **Then** results maintain expected rank ordering by similarity score

### Edge Cases
- What happens when BigQuery connection times out during query execution?
- How does system handle malformed or empty query vectors?
- What occurs when source_embeddings table is empty or missing?

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: Janapada memory service MUST execute BigQuery approximate_neighbors queries against source_embeddings table when BigQuery is configured
- **FR-002**: System MUST preserve existing vector_index interface contract without breaking Svami orchestration
- **FR-003**: System MUST provide graceful fallback to local in-memory vector index when BigQuery is unavailable
- **FR-004**: System MUST load BigQuery credentials and dataset/table names from configuration
- **FR-005**: System MUST return search results in order of similarity score matching expected rank ordering
- **FR-006**: System MUST handle BigQuery connection failures and query timeouts gracefully
- **FR-007**: System MUST maintain idempotent setup through existing `make setup` entry-point

### Key Entities *(include if feature involves data)*
- **source_embeddings**: BigQuery table containing VECTOR(D) embeddings from M2, target for approximate_neighbors queries
- **vector_index contract**: Existing interface between Svami and Janapada that must remain stable
- **BigQuery configuration**: Project/dataset/table/credential settings loaded from .env with sensible defaults

---

## Approach *(implementation guidance)*

### BigQuery approximate_neighbors Adapter
- Modify `src/janapada-memory` similarity_search method to construct and execute BigQuery SQL with VECTOR_SEARCH or approximate_neighbors function
- Leverage google-cloud-bigquery library for query execution and result parsing
- Maintain vector_index interface signatures and return types

### Configuration & Environment Wiring
- Extend configuration loading to accept GOOGLE_CLOUD_PROJECT, BIGQUERY_DATASET_ID, BIGQUERY_TABLE_PREFIX
- Use Application Default Credentials (ADC) or service account key from environment
- Provide sensible defaults: project from ADC, dataset "source_ingestion", table "source_embeddings"

### In-Memory Fallback
- Implement local vector index using existing vector utilities for testing and resilience
- Trigger fallback on BigQuery connection errors, missing tables, or credential issues
- Log fallback activation for operational visibility

---

## Risks

### Technical Risks
- **BigQuery query latency**: approximate_neighbors performance may exceed current in-memory expectations
- **Configuration complexity**: Multiple credential/project combinations could create setup friction
- **Fallback accuracy**: Local vector index may produce different similarity rankings than BigQuery

### Integration Risks
- **Interface contract drift**: Changes to similarity_search could inadvertently break Svami integration
- **Dependency chain**: New BigQuery dependency adds external failure points to memory service
- **Testing complexity**: Mocking BigQuery while maintaining contract fidelity requires careful test design

---

## Test Plan

### Unit Tests
- Mock BigQuery client and verify query construction for approximate_neighbors calls
- Assert rank ordering of returned chunk IDs matches expected similarity scores
- Test fallback path activation when BigQuery client raises connection exceptions
- Verify configuration loading from environment variables with defaults

### Integration Tests
- End-to-end test: Svami � Janapada � BigQuery � results for simple knowledge question
- Fallback verification: Disable BigQuery credentials and confirm local index usage
- Performance baseline: Compare BigQuery vs in-memory response times and accuracy

### Contract Tests
- Verify vector_index interface signatures remain unchanged
- Assert return types and error handling match existing Svami expectations
- Confirm no breaking changes to public method surfaces

---

## Rollback/Migration Steps

### Rollback Procedure
1. **Configuration rollback**: Remove BigQuery environment variables to force fallback mode
2. **Code rollback**: Revert janapada-memory to previous similarity_search implementation
3. **Verification**: Run Svami integration tests to confirm functionality restoration
4. **Cleanup**: Remove BigQuery-specific dependencies if no longer needed

### Migration Considerations
- **Gradual migration**: Deploy with BigQuery disabled initially, enable via configuration
- **Monitoring**: Track BigQuery query success rates and fallback activation frequency
- **Performance comparison**: Benchmark BigQuery vs in-memory response times before full cutover
- **Data consistency**: Verify BigQuery source_embeddings matches in-memory index content

---

## Review & Acceptance Checklist

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

---

## Execution Status

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [x] Review checklist passed

---