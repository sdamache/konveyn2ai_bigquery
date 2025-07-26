# KonveyN2AI - Agentic AI Application

KonveyN2AI is an Agentic AI Application built for a hackathon featuring Google Gemini API integration. The project implements a three-component architecture for AI-powered task management and automation.

## 🏗️ Architecture

The project follows a three-tier architecture:

### 1. Amatya Role Prompter (`src/amatya-role-prompter/`)
- Handles role-based prompting and user interaction
- Manages persona and context switching for different AI roles

### 2. Janapada Memory (`src/janapada-memory/`)
- Implements memory management and persistence
- Handles vector embeddings and knowledge retrieval
- Integrates with Google Cloud AI Platform for 3072-dimensional embeddings

### 3. Svami Orchestrator (`src/svami-orchestrator/`)
- Coordinates between components
- Manages workflow orchestration and task execution
- Handles multi-agent coordination

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
   ```bash
   pip install -r requirements.txt
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


