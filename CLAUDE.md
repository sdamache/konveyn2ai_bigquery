
# CLAUDE.md

***⭐ CRITICAL COMMUNICATION PROTOCOL ⭐***
**EVERY OUTPUT MUST START WITH 2-3 LINES EXPLAINING:**
1. **What I'm doing** (the specific action/fix being performed)
2. **Why I'm doing it** (the problem being solved or improvement made)  
3. **How it connects** (how this fits into the larger task/goal)

This explanation helps you understand, guide, and steer better. Keep it concise but clear.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KonveyN2AI_BigQuery is an Agentic AI Application built for the [BigQuery AI Hackathon](https://www.kaggle.com/competitions/bigquery-ai-hackathon) featuring BigQuery data analysis with Google Gemini API integration. The project implements a three-tier architecture where BigQuery serves as the system of record for data storage, vector search, and deterministic gap analysis, with AI orchestration providing bounded, additive intelligence.

## Core Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Development Commands
```bash
# Run the vector index setup (Google Cloud AI Platform)
python vector_index.py

# Activate virtual environment before any Python work
source venv/bin/activate

# M1 Ingestion Commands (NEW)
make setup                    # Create BigQuery tables (idempotent)
make ingest_k8s SOURCE=./k8s-manifests/
make ingest_fastapi SOURCE=./fastapi-project/
make ingest_cobol SOURCE=./cobol-copybooks/
make ingest_irs SOURCE=./irs-layouts/
make ingest_mumps SOURCE=./fileman-dicts/
```

## Architecture Components

The project follows a three-tier architecture:

### Tier 1: Presentation Layer
- **BigQuery Studio/Colab notebook** + static HTML heatmap viewer
- **Optional**: Vercel chat interface wired to Svami
- **Amatya Role Prompter** (`src/amatya-role-prompter/`) - Handles role-based prompting and user interaction

### Tier 2: Application/Orchestration Layer
- **Query Orchestration**: Lightweight Python layer (CLI/Make targets, optional FastAPI)
- **AI Orchestration**: Svami (`src/svami-orchestrator/`) for bounded AI tasks only
- Handles ingestion, schema creation, vector writes, SQL vector search, gap-rule queries
- **Janapada Memory** (`src/janapada-memory/`) - Integrates with BigQuery for vector embeddings

### Tier 3: Data Layer (BigQuery as System of Record)
- **source_metadata**: Normalized chunks from ingested artifacts
- **source_embeddings**: VECTOR(D) embeddings using Gemini
- **gap_metrics**: Results from deterministic SQL gap rules
- Executes vector search (DOT_PRODUCT/NORM or ANN) and joins

## Required Dependencies

From `requirements.txt`:
- `google-cloud-aiplatform` - Google Cloud AI Platform integration
- `google-generativeai` - Google Gemini API
- `google-cloud-bigquery` - BigQuery data operations (M1 NEW)
- `pyyaml` - YAML parsing for Kubernetes (M1 NEW)
- `kubernetes` - Kubernetes API client (M1 NEW)
- `libcst` - Python AST parsing for FastAPI (M1 NEW)
- `ulid-py` - Deterministic ID generation (M1 NEW)
- `streamlit` - Web interface framework
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `python-dotenv` - Environment variable management

## Google Cloud Configuration

The project uses Google Cloud AI Platform with:
- Project ID: `konveyn2ai`
- Location: `us-central1`
- Vector index: 3072 dimensions with cosine similarity
- Approximate neighbors count: 150

## Environment Variables

Required API keys and configuration (see `.env.example`):
- `GOOGLE_API_KEY` - For Google Gemini models via google console
- `GOOGLE_CLOUD_PROJECT_ID` - BigQuery project identifier
- `BIGQUERY_DATASET_ID` - Dataset for source_metadata, source_embeddings, gap_metrics tables
- Google Cloud credentials (ADC or service account key)
- Additional provider keys as needed

## Key Files

- `vector_index.py` - Google Cloud AI Platform vector index setup
- `requirements.txt` - Python dependencies
- `ARCHITECTURE.md` - Architectural documentation template
- `EXPLANATION.md` - Technical implementation details
- `DEMO.md` - Demo video and presentation materials


## Development Workflow

1. **Setup**: Create virtual environment and install dependencies
2. **Configuration**: Set up Google Cloud credentials and API keys
3. **Vector Index**: Run `vector_index.py` to initialize the AI Platform index
4. **Component Development**: Work within the three main src/ directories
5. **Integration**: Ensure components communicate through the orchestrator

## Data Flow & Orchestration

### BigQuery ↔ Embeddings Pipeline
1. **Ingest**: Artifacts (Kubernetes YAML/JSON, FastAPI OpenAPI & AST, COBOL copybooks, IRS IMF layouts, MUMPS/VistA) → normalized chunks to `source_metadata`
2. **Embed**: Text chunks via Gemini embeddings (app tier) → vectors to `source_embeddings` VECTOR(D)
3. **Retrieve**: SQL vector search (DOT_PRODUCT/NORM or ANN) joining embeddings → metadata
4. **Analyze**: Deterministic rules in SQL → results to `gap_metrics`
5. **Present**: Export CSV/MD + render static HTML heatmap; notebook displays same

### Orchestration Split
- **Query Orchestration** (deterministic, data-centric): App tier + BigQuery owns ingestion, schema creation, vector writes, SQL vector search, gap-rule queries, metrics materialization, exports. Goal: reproducible, auditable pipeline.
- **AI Orchestration** (bounded, optional): Svami handles narrow tasks only: doc summaries, "model-as-judge" consistency checks, answer snippets citing BigQuery rows. Never source of truth.

## Testing & Demo Requirements

This is a BigQuery AI Hackathon project requiring:
- Functional implementation (no mocks for core features, BigQuery is core functionality)
- 3-5 minute demo video
- Clear architecture documentation
- Working integration with BigQuery data analysis and Google Gemini API
- Demo-ready: One notebook run creates tables → ingests → embeds → searches → computes gaps → exports viewer

## File Structure
```
KonveyN2AI_BigQuery/
├── src/
│   ├── amatya-role-prompter/    # Role & prompt management
│   ├── janapada-memory/         # Memory & persistence
│   └── svami-orchestrator/      # Workflow orchestration
├── vector_index.py              # AI Platform setup
├── requirements.txt             # Dependencies
├── .env.example                 # Environment template
├── ARCHITECTURE.md              # Architecture docs
├── EXPLANATION.md               # Technical details
├── DEMO.md                      # Demo materials
└── GEMINI.md                    # Task Master integration
```

## **⭐ CRITICAL PROJECT MANAGEMENT PROTOCOLS ⭐**

### **🔥 MANDATORY FILE EDITING POLICY 🔥**
- **ALWAYS** prefer editing existing files over creating new ones
- **NEVER** create new files unless absolutely necessary for core functionality
- If debugging takes excessive time (>3 iterations), move old file to `.legacy/` or `.redundant/` folder
- Mark legacy files clearly and ensure they're ignored in `.gitignore`
- **Example:** Instead of creating `test_new.py`, enhance existing `test_file.py`

### **🧪 TESTING & DEMO REQUIREMENTS**
- **⚠️ HACKATHON RULE:** Create REAL functionality, NOT mocks
- Focus on **functional tests** that verify actual behavior
- **AVOID** circular testing dependencies (test-for-test-for-test)
- Mock only external dependencies (APIs, databases)
- **Demo-ready code** required - everything must work for live presentation
- Test both happy path AND error scenarios

### **🔍 SYSTEMATIC APPROACH PROTOCOL**
1. **ANALYZE:** Use Task agents to research issues online before coding
2. **SYNTHESIZE:** Understand root cause, don't patch symptoms  
3. **EXECUTE:** Make focused changes in existing files
4. **VERIFY:** Test locally before any git operations
5. **UPDATE MEMORY:** Document lessons learned after each task

### **🚀 GIT WORKFLOW STANDARDS**
- **TEST LOCALLY FIRST** - Never push untested code
- Create feature branches: `feat/task-<id>-<description>`
- Example: `feat/task-1.2-jwt-authentication` or `feat/task-3.1-api-endpoints`
- **Commit format:** `feat: implement <feature> (task <id>)`
- Only push to remote after local verification succeeds
- Create PRs with task context and testing evidence

### **🎯 FOCUS & EFFICIENCY RULES**
- **NO REDUNDANT FILES:** We created `test_limited_creation.py` AND `test_github_setup.py` for same function - AVOID THIS
- **CONSOLIDATE:** Analyze existing files, understand their purpose, enhance them
- **QUICK SYNTHESIS:** Don't overthink - understand, fix, move on
- **USE SUB-AGENTS:** For research and online investigation before coding
- **MEMORY UPDATES:** After every major task completion, update project memory

### **📊 PROGRESS TRACKING**
- **TaskMaster Status Sync:** Always update task status in TaskMaster first
- **GitHub Integration:** Sync TaskMaster → GitHub Issues automatically
- **Memory Documentation:** Record what works, what doesn't, lessons learned
- **Systematic Reviews:** After each task, document approach and results

### **⚡ EXECUTION SPEED OPTIMIZATION**
- **Research First:** Use agents to understand problems before coding
- **Focused Changes:** Edit existing files, don't create new infrastructure
- **Quick Iterations:** If stuck >30min, get help or try different approach
- **Working Code Priority:** Hackathon = demo-ready functionality required

---

**🔴 CRITICAL REMINDERS:**
- ⭐ **EDIT EXISTING FILES FIRST** - only create new files if absolutely necessary
- ⭐ **REAL FUNCTIONALITY** - no mocks for core demo features  
- ⭐ **TEST LOCALLY FIRST** - verify before git push
- ⭐ **UPDATE MEMORY** - document learnings after each task
- ⭐ **SYSTEMATIC APPROACH** - analyze → synthesize → execute → verify

---

## **📋 LIVING MEMORY SYSTEM**

### **Memory Management Protocol**
- **Task Memory**: Each task gets a living memory file in `.memory/tasks/task-[id]-[name].md`
- **Decision Memory**: Architecture and development decisions in `.memory/decisions/`
- **Session Memory**: Daily development sessions in `.memory/sessions/`
- **Access**: Use `ls -la .memory/` to view all memory files
- **Update Rule**: ALWAYS update task memory after significant progress (>30min work)

### **Memory File Template**
```markdown
# Task [ID]: [Title]

## Current Status
- **Started**: [Date]
- **Status**: [in-progress/completed/blocked]
- **Branch**: [feature-branch-name]

## Progress Log
- [Date] [Time]: [What was accomplished]
- [Date] [Time]: [Next step taken]

## Technical Decisions
- [Decision]: [Reasoning]

## Lessons Learned
- [What worked]
- [What didn't work]
- [Key insights]

## Next Steps
- [ ] [Specific action item]
```

### **Usage Instructions**
- **Before starting task**: Create memory file from template
- **During work**: Update progress log in real-time
- **After major steps**: Document decisions and lessons learned
- **Task completion**: Final summary and handoff notes


## Important Notes

- Focus on functional implementation over mocks for demo readiness
- BigQuery data analysis with Google Gemini API integration is core judging criterion
- BigQuery is core functionality, not external dependency - use real data for hackathon demo
- Maintain clear documentation for architectural decisions
- Query orchestration (deterministic) vs AI orchestration (bounded) - keep AI additive, not foundational

## Evaluation Priorities

**Technical Implementation (35% total):**
- Clean, efficient code that runs easily (20%) - prioritize readability and performance
- BigQuery AI as core function throughout solution (15%) - not just peripheral usage

**Innovation & Impact (25%):** Novel approach addressing significant problem with measurable impact (revenue, engagement, time saved)

**Demo & Documentation (20%):** Clear problem-solution relationship + architectural diagram + BigQuery AI explanation

**Assets (20%):** Public blog/video + GitHub repository with clear solution demonstration
