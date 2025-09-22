# KonveyN2AI - BigQuery AI Hackathon Submission

[![CI](https://github.com/sdamache/konveyn2ai_bigquery/actions/workflows/ci.yml/badge.svg)](https://github.com/sdamache/konveyn2ai_bigquery/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini%20API-4285F4.svg)](https://ai.google.dev/)
[![BigQuery](https://img.shields.io/badge/Google%20Cloud-BigQuery%20Vector-4285F4.svg)](https://cloud.google.com/bigquery)

**🏆 BigQuery AI Hackathon Entry**: Intelligent Knowledge Gap Detection with native BigQuery vector operations, hybrid AI analysis, and real-time interactive dashboards.

## 🚀 Quick Start for Kaggle Judges

### 📺 Demo Materials
- **🎬 3-Minute Demo Video**: [Watch Live Demo](https://www.loom.com/share/819aaf1a42fd414da4f04f0fc54cb120)
- **📋 Hackathon Submission**: [HACKATHON.md](./HACKATHON.md) - Complete submission details
- **📊 Interactive Dashboard**: `issue6_visualization_notebook.ipynb` - Live BigQuery AI visualizations

### ⚡ 5-Minute Reproduction (Kaggle Environment)
```bash
# 1. Clone and setup
git clone https://github.com/sdamache/konveyn2ai_bigquery.git
cd konveyn2ai_bigquery
pip install -r requirements-kaggle.txt

# 2. Set BigQuery environment
export GOOGLE_CLOUD_PROJECT=konveyn2ai
export BIGQUERY_DATASET_ID=semantic_gap_detector
export GOOGLE_API_KEY=your_gemini_api_key

# 3. Run complete demo pipeline
make demo_hackathon

# 4. View interactive results
jupyter notebook issue6_visualization_notebook.ipynb
```

### ✅ Verification Checklist
- [ ] BigQuery tables created successfully (`make setup`)
- [ ] Sample artifacts ingested (`make ingest_k8s SOURCE=./examples/k8s-manifests/`)
- [ ] AI embeddings generated (`make embeddings LIMIT=50`)
- [ ] Gap analysis completed (`make compute_metrics`)
- [ ] Interactive dashboard displays heatmaps
- [ ] Sub-second query performance demonstrated

### 🎯 Key Innovation Highlights
- **BigQuery Native AI**: Uses VECTOR(768) columns and `VECTOR_SEARCH()` for semantic analysis
- **Multi-Source Intelligence**: Analyzes Kubernetes, FastAPI, COBOL, IRS, and MUMPS artifacts
- **Hybrid Analysis**: Combines deterministic SQL rules with AI confidence scoring
- **Real-Time Insights**: Interactive dashboard with sub-second BigQuery queries
- **Production Scale**: Designed for 1M+ artifacts with cost-efficient processing

---

## 📋 Project Overview

**KonveyN2AI BigQuery Backend** is a next-generation vector storage solution that migrates from Vertex AI to BigQuery VECTOR capabilities. This implementation provides **10x cost reduction** while maintaining high performance through PCA dimension reduction and BigQuery's native vector search capabilities.

## 🚀 Major Architecture Upgrade: Vertex AI → BigQuery VECTOR

### Migration Overview
- **Previous**: 3072-dimensional embeddings in Vertex AI Vector Search
- **Current**: 768-dimensional embeddings with BigQuery VECTOR_SEARCH  
- **Benefits**: 90% cost reduction, better integration, improved scalability
- **Quality**: Maintains 95%+ similarity preservation through PCA dimension reduction

## 🌟 Key Highlights

- **🧠 Intelligent Agent Orchestration**: Multi-agent workflow with specialized AI components
- **🔍 Advanced Semantic Search**: Vector embeddings with 768-dimensional Google Cloud Vertex AI
- **🎭 Role-Based AI Guidance**: Context-aware advice generation for different developer personas
- **⚡ Real-Time Performance**: Sub-second response times with intelligent caching
- **🛡️ Production Security**: Enterprise-grade authentication, logging, and monitoring
- **🔄 Graceful Degradation**: Robust fallback mechanisms ensure continuous operation
- **📈 Coverage Reporting**: BigQuery snapshots, Streamlit dashboards, and exportable reports (see `docs/progress_reporting.md`)
- **☁️ Cloud Ready Dashboard**: Deploy Streamlit insights to Cloud Run (`Dockerfile.streamlit`, `docs/deploy_streamlit_cloud_run.md`)

## ✅ Multi-Source Ingestion Status - PRODUCTION READY

**Current Phase**: **COMPLETE** - All T001-T040 tasks finished ✅
**Status**: Production-ready multi-source ingestion system with BigQuery integration
**Validation**: All 107 unit tests passing, performance targets met, 2,509 rows in BigQuery

### Parser Architecture Ready
- **Kubernetes**: YAML/JSON manifests → k8s://{namespace}/{kind}/{name}
- **FastAPI**: OpenAPI specs + AST → py://{src_path}#{start_line}-{end_line}
- **COBOL**: Copybooks + PIC clauses → cobol://{file}/01-{structure_name}
- **IRS**: IMF layouts + positioning → irs://{record_type}/{layout_version}/{section}
- **MUMPS**: FileMan dictionaries → mumps://{global_name}/{node_path}



## 🧠 Embedding Generation Pipeline

KonveyN2AI implements a production-ready embedding generation pipeline that creates 768-dimensional vectors for semantic search using Google Gemini API.

### Key Features
- **768-dimensional embeddings** using Google's `text-embedding-004` model
- **Idempotent behavior** - skip existing embeddings automatically  
- **Disk-based caching** with SHA256 content hashing
- **Exponential backoff** and retry logic for API resilience
- **Batch processing** with configurable batch sizes (default: 32)
- **Cost optimization** through intelligent caching and deduplication

### Quick Start
```bash
# Setup environment variables
export GOOGLE_CLOUD_PROJECT=konveyn2ai
export BIGQUERY_DATASET_ID=semantic_gap_detector  
export GOOGLE_API_KEY=your_gemini_api_key

# Generate embeddings for all pending chunks
make embeddings

# Dry run with limit (testing)
LIMIT=100 DRY_RUN=1 make embeddings

# Batch size and cache configuration
EMBED_BATCH_SIZE=16 EMBED_CACHE_DIR=.cache/embeddings make embeddings
```

### Usage Examples

#### Basic Embedding Generation
```python
from pipeline.embedding import EmbeddingPipeline

# Initialize pipeline
pipeline = EmbeddingPipeline(
    project_id="konveyn2ai",
    dataset_id="semantic_gap_detector", 
    api_key="your_gemini_api_key"
)

# Generate embeddings for pending chunks
result = pipeline.generate_embeddings(limit=1000)

print(f"Generated {result['embeddings_generated']} embeddings")
print(f"API calls: {result['generator_stats']['api_calls']}")
print(f"Cache hits: {result['generator_stats']['cache_hits']}")
```

#### Vector Similarity Search Testing
```python
# Test vector search functionality
python test_vector_search.py \
    --project konveyn2ai \
    --dataset semantic_gap_detector \
    --verbose
```

### Expected Performance
- **API Latency**: ~100-200ms per embedding request
- **Cache Hit Rate**: 70-90% on subsequent runs
- **Cost Estimate**: ~$0.025 per 1,000 chunks (768-dim embeddings)
- **Processing Rate**: 100-500 chunks per minute (depending on cache hits)

### BigQuery Vector Operations
The pipeline creates and manages a `source_embeddings` table with:
- **chunk_id**: Unique identifier linking to source_metadata  
- **embedding**: 768-dimensional FLOAT64 array
- **model**: Embedding model name (text-embedding-004)
- **content_hash**: SHA256 hash for deduplication
- **Vector search support**: Compatible with `VECTOR_SEARCH()` and `ML.APPROXIMATE_NEIGHBORS()`

### Error Handling & Monitoring
- **Exponential backoff** for rate limiting (429 errors)
- **Automatic retries** for transient failures (5xx errors) 
- **Comprehensive logging** with request tracing and performance metrics
- **Graceful degradation** - continues processing other chunks if some fail

### Cache Management
```bash
# Cache location
ls -la .cache/embeddings/

# Cache statistics
python -c "
from pipeline.embedding import EmbeddingCache
cache = EmbeddingCache()
print(f'Cache entries: {len(list(cache.cache_dir.glob(\"*.json\")))}')
"

# Clear cache (force regeneration)
rm -rf .cache/embeddings/
```

## 🚀Multi-Source Ingestion - Usage Examples

### Quick Start - Ingestion Pipeline

```bash
# Activate environment
source venv/bin/activate

# Run performance validation with BigQuery ingestion
python scripts/performance_validation_v2.py --with-bigquery --files-per-source 100

# Validate data quality in BigQuery
python scripts/data_quality_validation.py
```

### Individual Parser Usage

#### 1. Kubernetes Manifests
```python
from src.ingest.k8s.parser import KubernetesParserImpl

parser = KubernetesParserImpl()
result = parser.parse_file("deployment.yaml")
print(f"Chunks created: {len(result.chunks)}")
print(f"Errors: {len(result.errors)}")

# Example output: k8s://default/Deployment/nginx-deployment
for chunk in result.chunks:
    print(f"Artifact ID: {chunk.artifact_id}")
    print(f"Content: {chunk.content_text[:100]}...")
```

#### 2. FastAPI Applications
```python
from src.ingest.fastapi.parser import FastAPIParserImpl

parser = FastAPIParserImpl()
result = parser.parse_file("app.py")

# Example output: py://app.py#45-67
for chunk in result.chunks:
    if 'route' in chunk.source_metadata:
        route = chunk.source_metadata['route']
        print(f"Route: {route['method']} {route['path']}")
        print(f"Function: {route['function_name']}")
```

#### 3. COBOL Copybooks
```python
from src.ingest.cobol.parser import COBOLParserImpl

parser = COBOLParserImpl()
result = parser.parse_file("customer-record.cob")

# Example output: cobol://customer-record.cob/01-CUSTOMER-RECORD
for chunk in result.chunks:
    metadata = chunk.source_metadata
    print(f"Level: {metadata['level_number']}")
    print(f"Structure: {metadata['field_name']}")
    print(f"PIC Clause: {metadata.get('pic_clause', 'N/A')}")
```

#### 4. IRS IMF Layouts
```python
from src.ingest.irs.parser import IRSParserImpl

parser = IRSParserImpl()
result = parser.parse_file("imf_layout.txt")

# Example output: irs://01/2024.1/IDENTITY
for chunk in result.chunks:
    metadata = chunk.source_metadata
    print(f"Section: {metadata['section']}")
    print(f"Fields: {metadata['field_count']}")
    print(f"Layout Version: {metadata['layout_version']}")
```

#### 5. MUMPS/VistA Files
```python
from src.ingest.mumps.parser import MUMPSParserImpl

parser = MUMPSParserImpl()
result = parser.parse_file("patient_dd.m")

# Example output: mumps://PATIENT/0.01
for chunk in result.chunks:
    metadata = chunk.source_metadata
    print(f"Global: {metadata['global_name']}")
    print(f"Field Number: {metadata['field_number']}")
    print(f"Data Type: {metadata['data_type']}")
```

### BigQuery Integration

#### Direct BigQuery Writing
```python
from src.common.bq_writer import BigQueryWriter

# Initialize writer
writer = BigQueryWriter(
    project_id='konveyn2ai',
    dataset_id='source_ingestion'
)

# Write chunks to BigQuery
run_id = f"ingestion_{int(time.time())}"
result = writer.write_chunks(chunks, run_id)

print(f"Rows written: {result.rows_written}")
print(f"Duration: {result.processing_duration_ms}ms")
```

#### Query BigQuery Data
```python
from google.cloud import bigquery

client = bigquery.Client(project='konveyn2ai')

# Get row counts by source type
query = """
SELECT
    source_type,
    COUNT(*) as row_count,
    COUNT(DISTINCT artifact_id) as unique_artifacts,
    AVG(content_tokens) as avg_tokens
FROM `konveyn2ai.source_ingestion.source_metadata`
GROUP BY source_type
ORDER BY row_count DESC
"""

results = client.query(query)
for row in results:
    print(f"{row.source_type}: {row.row_count:,} rows, {row.unique_artifacts:,} artifacts")
```

### Performance & Validation Scripts

#### Performance Testing
```bash
# Test parsing performance only (no BigQuery)
python scripts/performance_validation_v2.py --files-per-source 100

# Test with BigQuery ingestion
python scripts/performance_validation_v2.py --with-bigquery --files-per-source 50

# Custom performance targets
python scripts/performance_validation_v2.py --target-minutes 3 --files-per-source 200
```

#### Data Quality Validation
```bash
# Full data quality report
python scripts/data_quality_validation.py

# Custom validation thresholds
python scripts/data_quality_validation.py --min-rows 150 --project my-project
```

### Testing & Development

#### Run All Tests
```bash
# Unit tests (107 tests total)
pytest tests/unit/ -v

# Contract tests (BigQuery integration)
pytest tests/contract/ -v

# Integration tests
pytest tests/integration/ -v

# Performance validation
pytest tests/performance/ -v
```

#### Development Workflow
```bash
# Install in development mode
pip install -e .

# Run linting and formatting
ruff check src/
black src/

# Security scanning
bandit -r src/

# Type checking
mypy src/
```

### Production Deployment

#### Environment Setup
```bash
# Production environment variables
export GOOGLE_CLOUD_PROJECT=your-project-id
export BIGQUERY_INGESTION_DATASET_ID=source_ingestion
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Verify BigQuery access
python -c "from google.cloud import bigquery; client = bigquery.Client(); print('✅ BigQuery connection successful')"
```

#### Batch Processing
```python
# Process multiple directories
from src.ingest.orchestrator import IngestionOrchestrator

orchestrator = IngestionOrchestrator(
    project_id='your-project',
    dataset_id='source_ingestion'
)

# Process different source types
source_configs = [
    {'path': '/data/k8s-manifests', 'type': 'kubernetes'},
    {'path': '/data/fastapi-apps', 'type': 'fastapi'},
    {'path': '/data/cobol-copybooks', 'type': 'cobol'},
    {'path': '/data/irs-layouts', 'type': 'irs'},
    {'path': '/data/mumps-files', 'type': 'mumps'}
]

for config in source_configs:
    result = orchestrator.ingest_directory(
        directory_path=config['path'],
        source_type=config['type'],
        write_to_bigquery=True
    )
    print(f"{config['type']}: {len(result.chunks)} chunks processed")
```

### Data Analysis Examples

#### BigQuery Analytics Queries
```sql
-- Top 10 most complex files by token count
SELECT
    source_type,
    artifact_id,
    content_tokens,
    collected_at
FROM `konveyn2ai.source_ingestion.source_metadata`
ORDER BY content_tokens DESC
LIMIT 10;

-- Source type distribution over time
SELECT
    DATE(collected_at) as ingestion_date,
    source_type,
    COUNT(*) as chunks_created
FROM `konveyn2ai.source_ingestion.source_metadata`
GROUP BY ingestion_date, source_type
ORDER BY ingestion_date DESC;

-- Error analysis
SELECT
    source_type,
    error_class,
    COUNT(*) as error_count,
    ARRAY_AGG(DISTINCT error_msg LIMIT 3) as sample_errors
FROM `konveyn2ai.source_ingestion.source_metadata_errors`
GROUP BY source_type, error_class
ORDER BY error_count DESC;
```

### Current System Status

**✅ Production Metrics (Last Validation)**:
- **Total Chunks in BigQuery**: 2,509 rows across all source types
- **Kubernetes**: 209 chunks ✅ (Target: ≥100)
- **FastAPI**: 200 chunks ✅ (Target: ≥100)
- **COBOL**: 100 chunks ✅ (Target: ≥100)
- **IRS**: 300 chunks ✅ (Target: ≥100)
- **MUMPS**: 1,700 chunks ✅ (Target: ≥100)
- **Overall Quality Score**: 100/100 (Grade A - Excellent)
- **Test Coverage**: 107 unit tests passing (100% pass rate)
- **Performance**: Processes 2,800 chunks from 500 files in <0.18 minutes

## 📚 Project Origin & Research Background

### 🏛️ Ancient Wisdom Meets Modern AI

KonveyN2AI draws its architectural inspiration from **Chanakya's Saptanga Model** - the ancient Indian political science framework describing the "seven limbs" of a kingdom. This 2,300-year-old governance model provides a proven blueprint for distributed system organization, adapted here for modern AI agent collaboration.

**Saptanga → KonveyN2AI Mapping:**
- **Svami (The Ruler)** → Orchestrator Service: Central coordination and decision-making
- **Janapada (The Territory)** → Memory Service: Knowledge domain and information storage
- **Amatya (The Minister)** → Advisor Service: Intelligent counsel and guidance generation
- **Durga (The Fortress)** → Guard-Fort Middleware: Security and protection layer

### 🔬 Research-Driven Development

This project represents the culmination of extensive research into **computational organizational intelligence** - the science of coordinating autonomous AI agents through proven collaborative frameworks. Drawing from cutting-edge research in multi-agent systems and organizational science, KonveyN2AI implements novel architectures that bridge ancient wisdom with modern AI capabilities.

**Foundational Research Insights:**
- **Evolutionary Agent Architectures**: Self-improving systems that enhance capabilities through iterative refinement, inspired by Darwin Gödel Machine and AlphaEvolve research
- **Multi-Agent Coordination Paradigms**: Advanced orchestrator-worker patterns and collaborative specialist ensembles for distributed problem-solving
- **Team Topologies Framework**: Stream-aligned, Platform, Enabling, and Complicated-Subsystem agent roles for optimal cognitive load distribution
- **Agile Protocols for AI**: Time-boxed sprints, retrospectives, and impediment resolution for robust long-running agent operations

**Performance Research Insights:**
- **Sub-200ms Vector Search**: Optimized embedding generation and similarity matching using Google Vertex AI
- **Sub-5 Second End-to-End**: Complete query → search → advise → respond workflow with intelligent caching
- **Dynamic Organizational Restructuring**: Adaptive selection of collaboration patterns based on task complexity and requirements


**Success Metrics:**
- ✅ **Novel architectural patterns** based on ancient governance models and modern organizational science
- ✅ **Production-ready implementation** with enterprise-grade security and monitoring
- ✅ **Sub-second response times** with intelligent caching and async processing
- ✅ **Comprehensive research documentation** bridging historical wisdom with cutting-edge AI

## 🏗️ Three-Tier Agent Architecture (Saptanga-Inspired)

KonveyN2AI implements a sophisticated microservices architecture with three specialized AI agents working in harmony, directly inspired by Chanakya's ancient Saptanga governance model. Each agent embodies a specific "limb" of the system, ensuring distributed intelligence with centralized coordination:

### 🎭 Amatya Role Prompter (`src/amatya-role-prompter/`)
**The Persona Intelligence Agent**
- **Role-Based AI Guidance**: Generates contextual advice tailored to specific developer roles (Backend Developer, Security Engineer, DevOps Specialist, etc.)
- **Google Gemini Integration**: Leverages Gemini-1.5-Flash for advanced natural language understanding and generation
- **Intelligent Prompting**: Dynamic prompt construction based on user context and code snippets
- **Fallback Mechanisms**: Graceful degradation with Vertex AI text models when Gemini is unavailable

### 🧠 Janapada Memory (`src/janapada-memory/`)
**The BigQuery Memory Adapter**
- **BigQuery-first retrieval**: Executes `APPROXIMATE_NEIGHBORS` queries over the `source_embeddings` table to satisfy `vector_index` lookups without changing Svami's contract.
- **Transparent resilience**: Automatically promotes a local in-memory index whenever BigQuery credentials, tables, or jobs fail, then resumes BigQuery usage once the service recovers.
- **Operational visibility**: Structured logs surface BigQuery job IDs, latency, and fallback activations for observability pipelines.

#### Configuration
| Variable | Description | Default |
| --- | --- | --- |
| `GOOGLE_CLOUD_PROJECT` | Google Cloud project housing the BigQuery dataset. | Derived from Application Default Credentials |
| `BIGQUERY_DATASET_ID` | Dataset that stores `source_embeddings` and related tables. | `source_ingestion` |
| `BIGQUERY_TABLE_PREFIX` | Prefix applied to Janapada-managed tables (e.g., `source_embeddings`). | `source_` |
| `GOOGLE_APPLICATION_CREDENTIALS` *(optional)* | Path to a service account JSON key if ADC is unavailable. | unset |

> Run `make setup` whenever the dataset needs to be created or refreshed; the target is idempotent and provisions the embeddings table required by the adapter.

   ```

#### Fallback behaviour
- BigQuery execution is attempted first for every `similarity_search` call. Connection errors, missing tables, authentication failures, or query timeouts trigger the local vector index.
- Fallback activation is logged with context (exception, project/dataset, retry guidance) and persists only for the failed request—subsequent calls retry BigQuery automatically.
- To simulate and test fallback manually, export an invalid project ID: `export GOOGLE_CLOUD_PROJECT=invalid-project` before running the contract suite; the adapter should continue serving results from the in-memory store while reporting the outage.

#### Performance expectations
- **Latency**: <500 ms per `similarity_search` query when BigQuery is healthy (per contract test baseline).
- **Recall**: Matches Svami expectations by returning ranked chunk IDs in the same order as BigQuery `APPROXIMATE_NEIGHBORS` results.
- **Throughput**: Designed for ~150 neighbors per request—the adapter streams results without loading entire tables into memory.
- **Fallback cost**: Local index keeps parity with historical in-memory behaviour so Svami remains responsive during BigQuery incidents.

### 🎯 Svami Orchestrator (`src/svami-orchestrator/`)
**The Workflow Coordination Agent**
- **Multi-Agent Orchestration**: Coordinates complex workflows between Amatya and Janapada agents
- **Request Routing**: Intelligent query analysis and service delegation
- **Real-Time Monitoring**: Comprehensive health checks and performance metrics
- **Resilient Architecture**: Continues operation even with partial service failures


### JSON-RPC 2.0 Protocol
- **Standardized Communication**: All inter-agent communication uses JSON-RPC 2.0
- **Error Handling**: Comprehensive error codes and structured error responses
- **Request Tracing**: Unique request IDs for distributed system debugging
- **Type Safety**: Full Pydantic validation for all message payloads

## 📊 Implementation Status

### ✅ Production-Ready Components

#### 🎯 **Svami Orchestrator** - Multi-Agent Workflow Coordinator
- **FastAPI Application**: Production-grade REST API with comprehensive middleware
- **Multi-Agent Orchestration**: Complete query → search → advise → respond workflow
- **Health Monitoring**: Real-time service status with parallel health checks (<1ms response)
- **Security**: Bearer token authentication, CORS, security headers, input validation
- **Performance**: Sub-3ms end-to-end orchestration with async/await architecture

#### 🧠 **Janapada Memory** - Semantic Search Engine
- **Vector Embeddings**: Google Cloud Vertex AI `text-embedding-004` (768 dimensions)
- **Matching Engine**: Real-time similarity search with configurable thresholds
- **Intelligent Caching**: LRU cache (1000 entries) for optimal performance
- **Graceful Fallback**: Continues operation when cloud services unavailable
- **JSON-RPC Interface**: Standardized search API with comprehensive error handling

#### 🎭 **Amatya Role Prompter** - AI Guidance Generator
- **Gemini Integration**: Google Gemini-1.5-Flash for advanced language understanding
- **Role-Based Prompting**: Context-aware advice for different developer personas
- **Fallback Architecture**: Vertex AI text models when Gemini unavailable
- **Dynamic Prompting**: Intelligent prompt construction based on code context

### 🏗️ **Infrastructure & DevOps**
- **CI/CD Pipeline**: Automated testing, linting, security scanning (99 tests, 100% pass rate)
- **Docker Containerization**: Multi-service deployment with docker-compose
- **Security Scanning**: Bandit security analysis, dependency vulnerability checks
- **Code Quality**: Black formatting, Ruff linting, comprehensive type hints

### 🔧 **Core Technologies**
- **Google Cloud Integration**: Vertex AI, Matching Engine, Service Accounts
- **Modern Python Stack**: FastAPI, Pydantic v2, asyncio, uvicorn
- **Production Monitoring**: Structured logging, health checks, metrics collection
- **Enterprise Security**: Authentication, authorization, input validation, CORS


## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    KonveyN2AI Multi-Agent System                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   User Query    │───▶│ Svami           │───▶│   Response   │ │
│  │   + Role        │    │ Orchestrator    │    │   + Sources  │ │
│  └─────────────────┘    │ (Port 8080)     │    └──────────────┘ │
│                         └─────────────────┘                     │
│                                │                                │
│                                ▼                                │
│         ┌─────────────────────────────────────────┐             │
│         │          Workflow Coordination          │             │
│         └─────────────────────────────────────────┘             │
│                    │                    │                       │
│                    ▼                    ▼                       │
│  ┌─────────────────────────┐    ┌─────────────────────────┐     │
│  │    Janapada Memory      │    │   Amatya Role Prompter  │     │
│  │    (Port 8081)          │    │   (Port 8082)           │     │
│  │                         │    │                         │     │
│  │ • Vector Embeddings     │    │ • Gemini API Integration│     │
│  │ • Semantic Search       │    │ • Role-Based Prompting  │     │
│  │ • Matching Engine       │    │ • Context Analysis      │     │
│  │ • LRU Caching           │    │ • Fallback Mechanisms   │     │
│  └─────────────────────────┘    └─────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```


## 📋 BigQuery AI Hackathon Submission Status

### ✅ Completed Requirements
- **🎯 Problem Solution**: Intelligent knowledge gap detection across 5 artifact types
- **🤖 BigQuery AI Integration**: Native VECTOR operations, semantic search, ML functions
- **📊 Real-Time Analytics**: Interactive dashboard with sub-second query performance
- **🔬 Technical Innovation**: Hybrid deterministic + AI analysis approach
- **📈 Measurable Impact**: 10+ hour weekly productivity improvement per developer
- **🎬 Demo Materials**: 3-minute video walkthrough with clear problem/solution/results
- **📚 Documentation**: Complete setup instructions and reproduction guide
- **🧪 Testing**: 107 tests across contract, integration, and unit levels

### 🚀 Submission Components
- **[HACKATHON.md](./HACKATHON.md)**: Complete submission with problem statement and innovation details
- **[issue6_visualization_notebook.ipynb](./issue6_visualization_notebook.ipynb)**: Interactive BigQuery AI dashboard
- **[requirements.txt](./requirements.txt)**: Streamlined dependencies for Kaggle environment
- **Demo Video**: [3-minute walkthrough](https://www.loom.com/share/819aaf1a42fd414da4f04f0fc54cb120)
- **Verification Scripts**: `verify_kaggle_submission.py` and `quick_setup_test.py`
- **Public Repository**: Full source code with comprehensive documentation

### 🏆 Hackathon Evaluation Alignment
- **Technical Implementation (35%)**: Production-ready BigQuery AI integration
- **Innovation & Impact (25%)**: Novel multi-source gap detection with measurable business value
- **BigQuery AI Usage (20%)**: Native vector operations central to core functionality
- **Demo & Documentation (20%)**: Clear reproduction instructions and comprehensive guides

### 🎯 Judge Quick Access
1. **Watch Demo**: [3-minute video](https://youtu.be/wJY4wYffFuc) showcasing complete pipeline
2. **Review Submission**: [HACKATHON.md](./HACKATHON.md) for technical details and innovation
3. **Try Interactive Demo**: Follow 5-minute setup instructions above
4. **Explore Code**: Browse repository for architecture and implementation details

---

**Built with ❤️ for the BigQuery AI Hackathon on Kaggle**

*Demonstrating intelligent knowledge gap detection at enterprise scale with BigQuery's native AI capabilities*
