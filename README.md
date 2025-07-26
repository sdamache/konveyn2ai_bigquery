# KonveyN2AI - Agentic AI Application

[![CI](https://github.com/neeharve/KonveyN2AI/actions/workflows/ci.yml/badge.svg)](https://github.com/neeharve/KonveyN2AI/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

KonveyN2AI is an Agentic AI Application built for a hackathon featuring Google Gemini API integration. The project implements a three-component architecture for AI-powered task management and automation.

## 🏗️ Architecture

The project follows a three-tier architecture:

### 1. Amatya Role Prompter (`src/amatya-role-prompter/`)
- Handles role-based prompting and user interaction
- Manages persona and context switching for different AI roles

### 2. Janapada Memory (`src/janapada-memory/`)
- ✅ **IMPLEMENTED**: FastAPI service with JSON-RPC endpoint (Subtask 6.1)
- ✅ **IMPLEMENTED**: Vertex AI embeddings integration with graceful fallback (Subtask 6.2)
- 🔄 **IN PROGRESS**: Vertex AI Matching Engine client (Subtask 6.3)
- 📋 **PLANNED**: FAISS fallback mechanism (Subtask 6.4)
- **Features**: Memory management, vector embeddings, knowledge retrieval
- **Integration**: Google Cloud AI Platform with 768-dimensional embeddings (text-embedding-004)
- **Status**: Demo-ready with comprehensive real-world testing completed

### 3. Svami Orchestrator (`src/svami-orchestrator/`)
- Coordinates between components
- Manages workflow orchestration and task execution
- Handles multi-agent coordination

## 🗃️ Enhanced Data Models

The project includes a comprehensive set of Pydantic data models that enable seamless communication between all three components:

### Core Communication Models
- **Snippet**: Code snippet representation with file path and content
- **SearchRequest**: Basic search queries with configurable result limits
- **AdviceRequest**: Role-based advice generation requests
- **QueryRequest**: User queries with optional role context
- **AnswerResponse**: Structured responses with sources and tracing

### Advanced Vector Search Models
- **SearchQuery**: Enhanced search with embedding support, multiple search types (VECTOR, KEYWORD, HYBRID, SEMANTIC), filters, and score thresholds
- **SearchResult**: Comprehensive search results with relevance scoring, metadata, and document references
- **DocumentChunk**: Document processing units with embedding storage and indexing support
- **Document**: Full document lifecycle management with status tracking and metadata
- **SearchType**: Enumeration supporting multiple search strategies

### Usage Example
```python
from src.common.models import SearchQuery, SearchType, DocumentChunk

# Create an advanced search query
query = SearchQuery(
    text="authentication middleware implementation",
    search_type=SearchType.HYBRID,
    top_k=10,
    filters={"language": "python"},
    min_score=0.7
)

# Process document chunks
chunk = DocumentChunk(
    content="def authenticate_request(token: str) -> bool:",
    document_id="doc-123",
    chunk_index=0,
    metadata={"language": "python", "function": "authenticate_request"}
)
```

### Integration Points
- **JSON-RPC Protocol**: All models integrate seamlessly with the JSON-RPC 2.0 communication layer
- **FastAPI Compatibility**: Full Pydantic v2 support for automatic API documentation and validation
- **Three-Tier Architecture**: Models designed specifically for Amatya ↔ Janapada ↔ Svami communication
- **Vector Search Ready**: Built-in support for 3072-dimensional embeddings and similarity search

## 📊 Project Status

### ✅ Completed Tasks
- **Task 1**: Project Setup and Configuration - Complete
- **Task 4**: Agent Manifest Generator - Complete
- **Task 5**: Enhanced Data Models and Schemas - Complete
  - ✅ Base data model structure with Pydantic v2
  - ✅ Request models (SearchRequest, AdviceRequest, QueryRequest)
  - ✅ Response models (AnswerResponse, SearchResult, DocumentChunk)
  - ✅ Advanced vector search models with embedding support
  - ✅ Comprehensive test coverage (28 tests, 100% pass rate)
  - ✅ JSON-RPC protocol integration
  - ✅ FastAPI compatibility and documentation

### 🚧 In Progress
- **Task 2**: Google Cloud Project Configuration
- **Task 3**: Vector Index Setup

### 📋 Upcoming Tasks
- **Task 6**: Janapada Memory Service Implementation
- **Task 7**: Amatya Advisor Service Implementation
- **Task 8**: Svami Orchestrator Service Implementation
- **Task 9**: Repository Ingestion Pipeline

### 🎯 Dependencies Ready
Task 5 completion has unblocked all service implementation tasks (6, 7, 8, 9) by providing:
- Complete data model foundation
- Vector search capabilities
- Service communication protocols
- Document processing structures

## 📋 Submission Checklist

- [ ] All code in `src/` runs without errors
- [ ] `ARCHITECTURE.md` contains a clear diagram sketch and explanation
- [ ] `EXPLANATION.md` covers planning, tool use, memory, and limitations
- [ ] `DEMO.md` links to a 3–5 min video with timestamped highlights


## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- Google Cloud account with AI Platform enabled
- API keys for required services

### Environment Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/neeharve/KonveyN2AI.git
   cd KonveyN2AI
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   **Option A: Using pip (recommended for quick setup)**
   ```bash
   pip install -r requirements.txt
   # Install package in development mode for proper imports
   pip install -e .
   ```

   **Option B: Using Poetry (recommended for development)**
   ```bash
   # Install Poetry if not already installed
   curl -sSL https://install.python-poetry.org | python3 -

   # Install dependencies
   poetry install

   # Activate Poetry shell
   poetry shell
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

5. **Verify installation**
   ```bash
   python -m pytest tests/test_project_setup.py -v
   ```

### Required API Keys

Add these to your `.env` file:
- `ANTHROPIC_API_KEY` - For Claude model access via Augment Agent
- `GOOGLE_API_KEY` - For Google Gemini models
- `PERPLEXITY_API_KEY` - For research capabilities (optional)

### Component-Specific Dependencies

Each component has its own requirements file for modular development:

- `src/amatya-role-prompter/requirements.txt` - Role management and prompting
- `src/janapada-memory/requirements.txt` - Memory and vector embeddings
- `src/svami-orchestrator/requirements.txt` - Workflow orchestration
- `src/common/requirements.txt` - Shared utilities

To install dependencies for a specific component:
```bash
pip install -r src/[component-name]/requirements.txt
```

### Google Cloud Configuration

The project uses Google Cloud AI Platform with:
- Project ID: `konveyn2ai`
- Location: `us-central1`
- Vector index: 3072 dimensions with cosine similarity
- Approximate neighbors count: 150


## 📂 Folder Layout

![Folder Layout Diagram](images/folder-githb.png)



## 🏅 Judging Criteria

- **Technical Excellence **
  This criterion evaluates the robustness, functionality, and overall quality of the technical implementation. Judges will assess the code's efficiency, the absence of critical bugs, and the successful execution of the project's core features.

- **Solution Architecture & Documentation **
  This focuses on the clarity, maintainability, and thoughtful design of the project's architecture. This includes assessing the organization and readability of the codebase, as well as the comprehensiveness and conciseness of documentation (e.g., GitHub README, inline comments) that enables others to understand and potentially reproduce or extend the solution.

- **Innovative Gemini Integration **
  This criterion specifically assesses how effectively and creatively the Google Gemini API has been incorporated into the solution. Judges will look for novel applications, efficient use of Gemini's capabilities, and the impact it has on the project's functionality or user experience. You are welcome to use additional Google products.

- **Societal Impact & Novelty **
  This evaluates the project's potential to address a meaningful problem, contribute positively to society, or offer a genuinely innovative and unique solution. Judges will consider the originality of the idea, its potential real‑world applicability, and its ability to solve a challenge in a new or impactful way.
