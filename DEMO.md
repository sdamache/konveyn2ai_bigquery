# KonveyN2AI - Live Demo Guide

## 🎬 Demo Video

📺 **Hosted Public Video Link:**
**[KonveyN2AI Multi-Agent System Demo](https://www.loom.com/share/819aaf1a42fd414da4f04f0fc54cb120?sid=037503c0-dc7f-48ca-a3ef-2b44420ac304)**

*3-minute demonstration of intelligent multi-agent workflow for code understanding and developer guidance*

### 📋 Demo Timestamps

- **00:00–00:30** — Problem introduction & system overview
- **00:30–01:30** — Multi-agent workflow demonstration
- **01:30–02:30** — Semantic search & AI guidance generation
- **02:30–03:00** — Performance metrics & edge case handling

---

## 🚀 Live Demo Script

### Problem Statement (00:00-00:30)

**"How can developers quickly understand complex codebases and get expert-level implementation guidance?"**

KonveyN2AI solves this through:
- **Intelligent Code Search**: Semantic understanding beyond keyword matching
- **Role-Based AI Guidance**: Tailored advice for different developer personas
- **Multi-Agent Orchestration**: Specialized AI agents working in harmony

### System Overview
```
User Query → Svami Orchestrator → Janapada Memory + Amatya Role Prompter → Intelligent Response
```

## 🎯 Demo Scenario: Authentication Middleware Implementation

### Step 1: Query Input (00:30-00:45)

**User Query**: *"How do I implement secure authentication middleware for a FastAPI application?"*
**Role Context**: *"Backend Developer"*

```bash
curl -X POST http://localhost:8080/answer \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer demo-token" \
  -d '{
    "question": "How do I implement secure authentication middleware for FastAPI?",
    "role": "backend_developer"
  }'
```

### Step 2: Multi-Agent Workflow (00:45-01:30)

#### 🎯 Svami Orchestrator - Workflow Coordination
```
[2025-07-26 10:30:15] INFO: Received query from backend_developer
[2025-07-26 10:30:15] INFO: Generated request ID: req_auth_12345
[2025-07-26 10:30:15] INFO: Initiating multi-agent workflow
```

#### 🧠 Janapada Memory - Semantic Search
```
[2025-07-26 10:30:15] INFO: Converting query to 768-dimensional embedding
[2025-07-26 10:30:15] INFO: Searching vector index for similar code snippets
[2025-07-26 10:30:15] INFO: Found 5 relevant snippets with similarity > 0.7
```

**Search Results**:
```json
{
  "snippets": [
    {
      "file_path": "src/guard_fort/middleware.py",
      "content": "def authenticate_request(token: str) -> bool:\n    \"\"\"Validate Bearer token authentication\"\"\"\n    if not token or not token.startswith('Bearer '):\n        return False\n    return validate_jwt_token(token[7:])",
      "similarity_score": 0.89
    }
  ]
}
```

#### 🎭 Amatya Role Prompter - AI Guidance Generation
```
[2025-07-26 10:30:16] INFO: Analyzing code snippets for backend_developer role
[2025-07-26 10:30:16] INFO: Constructing specialized prompt for Gemini API
[2025-07-26 10:30:16] INFO: Generating tailored implementation guidance
```

### Step 3: Intelligent Response Generation (01:30-02:30)

#### Role-Based Analysis
The Amatya agent analyzes the code snippets specifically for a "Backend Developer" persona:

**Prompt Construction**:
```
As a Backend Developer working with FastAPI, analyze these authentication middleware examples and provide implementation guidance focusing on:
- Security best practices
- Performance considerations
- Error handling patterns
- Integration with existing FastAPI applications

Code Examples:
[Relevant snippets from Janapada search]
```

#### Gemini API Integration
```
[2025-07-26 10:30:16] INFO: Calling Gemini-1.5-Flash model
[2025-07-26 10:30:16] INFO: Response generated in 187ms
[2025-07-26 10:30:16] INFO: Confidence score: 0.92
```

### Step 4: Orchestrated Final Response (02:30-03:00)

#### Complete Response
```json
{
  "answer": "As a Backend Developer, here's how to implement secure authentication middleware for FastAPI:\n\n## 1. Basic Middleware Structure\n```python\nfrom fastapi import FastAPI, Request, HTTPException\nfrom fastapi.security import HTTPBearer, HTTPAuthorizationCredentials\n\nclass AuthenticationMiddleware:\n    def __init__(self, app: FastAPI):\n        self.app = app\n        self.security = HTTPBearer()\n    \n    async def __call__(self, request: Request, call_next):\n        # Extract and validate token\n        auth: HTTPAuthorizationCredentials = await self.security(request)\n        if not self.validate_token(auth.credentials):\n            raise HTTPException(status_code=401, detail=\"Invalid token\")\n        \n        response = await call_next(request)\n        return response\n```\n\n## 2. Security Best Practices\n- **Token Validation**: Always validate JWT tokens with proper signature verification\n- **Rate Limiting**: Implement rate limiting to prevent brute force attacks\n- **HTTPS Only**: Ensure all authentication happens over HTTPS\n- **Token Expiration**: Use short-lived tokens with refresh mechanisms",

  "sources": [
    "src/guard_fort/middleware.py",
    "src/guard_fort/auth.py",
    "src/common/security.py"
  ],

  "request_id": "req_auth_12345",
  "confidence_score": 0.92,
  "processing_time_ms": 387
}
```

## 📊 Performance Demonstration

### Real-Time Metrics
```
Total Processing Time: 387ms
├── Semantic Search: 120ms (31%)
├── AI Generation: 187ms (48%)
├── Orchestration: 80ms (21%)
└── Network Overhead: <1ms

Memory Usage: 45MB (efficient caching)
Cache Hit Rate: 85% (embedding cache)
Confidence Score: 0.92 (high quality)
```

### Edge Case Handling

#### Scenario: Service Degradation
```bash
# Simulate Gemini API unavailability
curl -X POST http://localhost:8080/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "authentication patterns", "role": "developer"}'
```

**Graceful Fallback Response**:
```json
{
  "answer": "Based on the code snippets found, here are authentication patterns... [Generated using Vertex AI fallback]",
  "sources": ["src/guard_fort/middleware.py"],
  "request_id": "req_fallback_789",
  "confidence_score": 0.78,
  "note": "Response generated using fallback model due to primary service unavailability"
}
```

## 🎭 Agentic Behavior Highlights

### 1. Intelligent Planning
- **Query Analysis**: Understands intent beyond keywords
- **Role Context**: Adapts response style to user persona
- **Workflow Orchestration**: Coordinates multiple AI agents seamlessly

### 2. Memory & Context Usage
- **Semantic Memory**: 768-dimensional vector embeddings for code understanding
- **Context Preservation**: Maintains request context across agent interactions
- **Intelligent Caching**: LRU cache for performance optimization

### 3. Tool Integration
- **Google Gemini API**: Advanced language understanding and generation
- **Vertex AI Embeddings**: Semantic code search capabilities
- **Matching Engine**: Production-scale vector similarity search

### 4. Adaptive Behavior
- **Fallback Mechanisms**: Graceful degradation when services unavailable
- **Error Recovery**: Intelligent retry logic and alternative approaches
- **Performance Optimization**: Dynamic caching and connection pooling

---

## 🚀 Try It Yourself

### Option 1: Live Demo Website (Recommended)
🌐 **[Try KonveyN2AI Live Demo](https://konveyn2ai-website.vercel.app/)**

Experience the multi-agent system directly through our web interface:
- No setup required - just open the link and start asking questions
- Test different developer roles and see real-time responses
- Explore the authentication middleware example from the demo

### Option 2: Local Development Setup
```bash
# Start the system
docker-compose up -d

# Test the multi-agent workflow
curl -X POST http://localhost:8080/answer \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer demo-token" \
  -d '{
    "question": "Your question here",
    "role": "backend_developer"
  }'

# Check system health
curl http://localhost:8080/health/detailed
```

### Available Developer Roles
- `backend_developer` - Server-side implementation focus
- `frontend_developer` - UI/UX and client-side focus
- `security_engineer` - Security and compliance focus
- `devops_engineer` - Infrastructure and deployment focus
- `data_scientist` - Analytics and ML focus

**Experience the future of intelligent code assistance with KonveyN2AI!**