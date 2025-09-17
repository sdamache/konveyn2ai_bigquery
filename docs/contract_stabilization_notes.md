# Contract Stabilization Notes (Hackathon Sprint)

This document captures the end-to-end context from the API remediation guide, the contract run issues we fixed, and the outstanding items to address after the hackathon demo.

## 1. Environment & Modes

- Production defaults remain `GOOGLE_CLOUD_PROJECT=konveyn2ai` and `BIGQUERY_DATASET_ID=semantic_gap_detector`.
- Contract suites run hermetically with `SCHEMA_MANAGER_MODE=stub` and `VECTOR_STORE_MODE=stub`.
- Flip both env vars to `bigquery` (plus credentials) only when you need to exercise real datasets or integration tests.
- FastAPI now resets its in-memory stubs between pytest cases to keep each contract isolated.

## 2. Frequently Touched Files

| Area | Key Files | Purpose |
| --- | --- | --- |
| Schema manager API | `src/api/schema_endpoints.py`, `src/janapada_memory/schema_manager.py`, `src/janapada_memory/schema_manager_stub.py` | Table/index CRUD, validation |
| Vector store API | `src/api/vector_endpoints.py`, `src/janapada_memory/bigquery_vector_store.py`, `src/janapada_memory/vector_store_stub.py` | Embedding CRUD, search |
| Models / requests | `src/api/models.py` | Pydantic request/response schemas |
| Tests / fixtures | `tests/contract/*.py` | Contract expectations per endpoint |
| Shared constants | `src/janapada_memory/__init__.py`, `.env.example` | Default env values |

## 3. Pain Points That Drove the Sprint

1. **Schema endpoints**
   - `POST /schema/tables`: needed to surface 400/409 based on stub errors instead of blindly returning 201.
   - `POST /schema/indexes`: invalid tables/columns/options now respond with 400; indexes on stub tables succeed in 201.
   - `GET /schema/tables`: stub mirrors contract-required fields (`num_rows`, `num_bytes`, etc.).
   - `POST /schema/validate`: stub populates `tables`, `indexes`, `recommendations`, and optional `data_quality` block when `validate_data=True`.
2. **Vector endpoints**
   - Manual request parsing prevents FastAPI 422s and lets the API respond with contract-friendly 400/415 codes.
   - Responses now echo `created_at`, keep metadata as dictionaries, and strip optional fields when `None`.
   - Search responses remain deterministic in the stub while leaving room for BigQuery alignment later (metadata JSON parsing, new columns, etc.).
3. **Swagger alignment**
   - Pydantic models already matched the contract; the runtime code now conforms to them.

## 4. Stub vs BigQuery Behavior

- Both managers are still env-driven. Contracts use stubs; integration tests should flip to `bigquery` explicitly.
- Vector stub seeds two deterministic records (`func_calculate_similarity_001`, `func_to_delete_001`) and resets after each test to provide isolated runs.
- Keep coding against interface methods (`create_tables`, `search_similar_vectors`, etc.) so the real implementations can be swapped in without drift.

## 5. Work Plan We Executed

1. Schema endpoints: upfront validation, structured error payloads, and `data_quality` block in validation responses.
2. Vector endpoints: manual JSON parsing, NaN-friendly httpx shim, consistent metadata handling, trailing-slash 404 guard.
3. BigQuery parity remains future scope—queries and JSON parsing still need updates once we leave stub mode.

## 6. Testing Strategy

```bash
source venv/bin/activate
export TMPDIR=$(pwd)/.tmp && mkdir -p "$TMPDIR"
pytest tests/contract/test_schema_manager_create_tables.py \
       tests/contract/test_schema_manager_create_indexes.py \
       tests/contract/test_schema_manager_validate.py \
       tests/contract/test_vector_store_post_embeddings.py \
       tests/contract/test_vector_store_get_embeddings.py \
       tests/contract/test_vector_store_get_by_id.py \
       tests/contract/test_vector_store_search.py
```

- BigQuery integration loop is still available:

```bash
export SCHEMA_MANAGER_MODE=bigquery
export VECTOR_STORE_MODE=bigquery
pytest tests/integration/test_setup_workflow.py \
       tests/integration/test_vector_workflow.py
```

Keep stub mode as the default so contract cycles stay hermetic.

## 7. Pending TODOs (Non-Exhaustive)

- [ ] Swap `datetime.utcnow()` for `datetime.now(timezone.utc)` to remove deprecation warnings and unify timestamps.
- [ ] Enrich batch responses with per-entry error details (contract doesn’t assert it yet, but it is on the backlog).
- [ ] Mirror env-toggle notes in `CLAUDE.md` / other developer guides.
- [ ] Remove `tests/contract/contract_test_output.txt` and any remaining debug artifacts.
- [ ] Align BigQuery implementations (queries, metadata JSON parsing) with the stub behavior post-hackathon.

## 8. Commentary for the Next Developer

- The heavy schema refactor already landed; these changes bring the API up to contract parity without touching BigQuery.
- Treat the stubs as source-of-truth for contract tests. Real BigQuery integration can follow the same interface after the hackathon.
- Stage future changes schema-first, then vector endpoints, rerunning contracts between each cluster.
- After contract green, revisit BigQuery execution mode to ensure real runs match stub payloads (especially metadata dicts and timestamp formats).

## 9. Quick Reference Snippets

- **Schema manager factory**

  ```python
  if os.getenv("SCHEMA_MANAGER_MODE", "stub").lower() != "bigquery":
      return InMemorySchemaManager()
  return SchemaManager()
  ```

- **Embedding validation**

  ```python
  if len(request.embedding) != 768:
      raise ValueError("embedding must contain exactly 768 dimensions")
  ```

- **Vector search response shape**

  ```python
  return VectorSearchResponse(
      results=[SimilarityResult(**result) for result in results],
      query_embedding=request.query_embedding,
      total_results=len(results),
      search_time_ms=6,
  )
  ```

## 10. Commit Strategy (Sprint Summary)

1. Runtime contract fixes (`src/api/...`, `src/janapada_memory/...`, `tests`) in one “Contract alignment” commit.
2. Infrastructure patch (`src/api/main.py` NaN handling + stub resets) can ride with the commit above or stand alone.
3. Documentation update (`docs/contract_stabilization_notes.md`, README link) as a clean docs commit.

This keeps reviewers focused on logic vs. docs, and leaves space for future BigQuery-specific adjustments.

## 11. Final Notes

- All seven contract suites now pass (`93 passed`).
- Warnings are known (utcnow deprecation, FastAPI event warning) and can be addressed post-hackathon.
- Stub implementations remain “truth” for tests; BigQuery alignment is the next milestone once there’s time.
- Run integration tests with BigQuery env toggles before any demo to double-check real backend behavior.

## 12. BigQuery Integration Regression (Sept 17, 2025)

The latest integration run with `SCHEMA_MANAGER_MODE=bigquery` and `VECTOR_STORE_MODE=bigquery`:

```
============================= test session starts ==============================
collected 20 items

tests/integration/test_setup_workflow.py ..FFFF.F.FF..
tests/integration/test_vector_workflow.py FFFFFFF

================== 14 failed, 6 passed, 13 warnings in 48.57s ==================
```

Key failures surfaced by this run:

- `SchemaManager.list_tables()` expects `TableListItem.num_rows/num_bytes`, which are absent on the lightweight `TableListItem` returned by `BigQueryConnection.list_tables()`.
- Partitioning introspection uses `table.time_partitioning.type_.name`; the SDK now returns a string for `type_` in certain cases.
- Real BigQuery insertions return `datetime` objects in result payloads, and the current JSON serialization path does not coerce them, leading to `TypeError: Object of type datetime is not JSON serializable` in the vector workflow.
- Environment validation differs when BigQuery credentials are missing or misconfigured (`test_environment_variable_validation`).
- Vector workflow tests (`tests/integration/test_vector_workflow.py`) all require follow-up once the schema manager and vector store handle the real dataset responses consistently.

Use this output as the starting point for the next BigQuery alignment issue—each failure maps to a concrete gap between the stub contract behavior and the live GCP APIs.
