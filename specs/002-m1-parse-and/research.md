# Research Findings: M1 Multi-Source Ingestion

## BigQuery Ingestion Strategy

**Decision:** BigQuery Storage Write API
**Rationale:** Storage Write API provides superior performance (1 MBps+ throughput per connection), real-time availability (~90 minutes), and exactly-once semantics crucial for data integrity. For hackathon context, the ~$0.05/GB cost is negligible compared to operational simplicity and performance benefits.
**Alternatives considered:** Load Jobs (free but batch-oriented, shared slot limitations), Legacy Streaming API (deprecated, more expensive)
**Implementation notes:** Use default stream for better scalability, implement exponential backoff for quota errors, design schema with nullable fields for future expansion.

## Text Chunking Strategies

**Decision:** Source-aware semantic chunking with hierarchical preservation
**Rationale:** Each source type has natural semantic boundaries that must be preserved for effective embedding and retrieval. Variable chunk sizes optimize for content completeness while respecting token limits.
**Alternatives considered:** Fixed-size chunking (breaks semantic boundaries), very large chunks (exceed embedding limits)
**Implementation notes:**
- **Kubernetes:** 800-1200 tokens, 15% overlap, split at resource boundaries
- **FastAPI:** 600-900 tokens (code) / 1000-1500 tokens (specs), 20% overlap, function/class boundaries
- **COBOL:** 800-1200 tokens, 25% overlap, preserve 01-level hierarchies
- **IRS:** 900-1400 tokens, 15% overlap, record-type boundaries
- **MUMPS:** 700-1000 tokens, 30% overlap, file-entry level with cross-references

## Parser Library Architecture

**Decision:** Hybrid approach prioritizing performance
**Rationale:** Regex-based parsing is 36-100x faster than pure Python parsers (109ms vs 11,518ms), critical for processing 100+ files within hackathon timeline. Fallback to specialized libraries for complex edge cases.
**Alternatives considered:** Full AST parsing (too slow), pyparsing (100x slower), ANTLR4 (complex setup)
**Implementation notes:**
- **COBOL:** python-cobol + regex preprocessing for PIC/OCCURS/REDEFINES
- **IRS:** Pure regex with struct module for fixed-width positioning
- **MUMPS:** Custom regex parser + pyGTM validation
- **Error handling:** Fast failure detection, fallback parsers, continue-on-error processing

## Kubernetes Resource Extraction

**Decision:** kr8s + PyYAML hybrid approach
**Rationale:** kr8s provides kubectl-like simplicity (3-4 lines vs 15+ for common operations) with good async support. PyYAML with CLoader offers 6-18x performance improvement for bulk processing. Supports both live clusters and offline manifest processing for hackathon demos.
**Alternatives considered:** Official client (too verbose), pure YAML (no live cluster support), kubectl subprocess (poor error handling)
**Implementation notes:**
- Use CLoader for 6x YAML parsing speedup
- Async operations with kr8s.asyncio for concurrent extraction
- Metadata extraction: name, namespace, kind, labels, annotations, resource_version
- Fallback to manifest processing when clusters inaccessible

## FastAPI Analysis Strategy

**Decision:** Runtime OpenAPI extraction + AST source mapping
**Rationale:** FastAPI's built-in `/openapi.json` generation captures complete schemas, dynamic routes, and middleware effects automatically. AST parsing only needed for source code location mapping (file paths, line numbers).
**Alternatives considered:** Pure AST parsing (misses dynamic routes), static OpenAPI files (may be outdated), LibCST (3x slower)
**Implementation notes:**
- Extract from `app.openapi()` for complete schemas and model definitions
- Use AST for source location mapping (function names to file:line)
- Direct Pydantic model introspection for field types and validation rules
- Handle both running applications and source-only analysis

## Deterministic ID Generation

**Decision:** Content-hash based composite keys
**Rationale:** Ensures idempotent writes and change detection while maintaining semantic identifiers for debugging and lineage tracking.
**Implementation pattern:**
- **artifact_id:** `{source_type}://{semantic_path}` (e.g., `k8s://default/deployment/webapp`)
- **content_hash:** `sha256(normalized_content)`
- **Primary key:** `(source_type, artifact_id, content_hash)`

## Performance & Scalability Targets

**Decision:** Batch-optimized processing with parallelization
**Processing targets:** 1000+ artifacts/minute, <200ms p95 per chunk, <16KB content per row
**Scalability approach:** Parallel parser execution, batch BigQuery writes, content hashing for incremental updates
**Error resilience:** Continue-on-error processing, structured error logging to `source_metadata_errors` table

## Technology Stack Confirmation

**Language:** Python 3.11 (confirmed available)
**Core dependencies:** google-cloud-bigquery, pyyaml, kubernetes, libcst, ulid-py, python-cobol
**Testing:** pytest with real BigQuery temp datasets (no mocks for core functionality)
**Observability:** Structured JSON logging with correlation IDs, error classification, processing metrics

This research foundation enables systematic implementation of the M1 ingestion pipeline with proven performance characteristics and clear technical debt management.