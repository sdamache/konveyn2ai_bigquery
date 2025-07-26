# KonveyN2AI - System Architecture

## 🏗️ Multi-Agent Architecture Overview

KonveyN2AI implements a sophisticated three-tier microservices architecture where specialized AI agents collaborate to provide intelligent code understanding and developer guidance. Each agent has distinct responsibilities and communicates through standardized JSON-RPC 2.0 protocols.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          KonveyN2AI Multi-Agent System                      │
│                                                                             │
│  ┌─────────────────┐                                      ┌──────────────┐  │
│  │   User Query    │                                      │   Response   │  │
│  │   + Role        │                                      │   + Sources  │  │
│  │   + Context     │                                      │   + Metadata │  │
│  └─────────────────┘                                      └──────────────┘  │
│           │                                                        ▲        │
│           ▼                                                        │        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    🎯 Svami Orchestrator (Port 8080)                   │ │
│  │                                                                        │ │
│  │  • FastAPI Application with Guard-Fort Middleware                      │ │
│  │  • Multi-Agent Workflow Coordination                                   │ │
│  │  • Request Routing & Response Aggregation                              │ │
│  │  • Health Monitoring & Performance Metrics                             │ │
│  │  • Authentication & Security Headers                                   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│           │                                                        ▲        │
│           ▼                                                        │        │
│  ┌─────────────────────────────┐    ┌─────────────────────────────┐         │
│  │  🧠 Janapada Memory Agent   │    │  🎭 Amatya Role Prompter    │         │
│  │      (Port 8081)            │    │      (Port 8082)            │         │
│  │                             │    │                             │         │
│  │ ┌─────────────────────────┐ │    │ ┌─────────────────────────┐ │         │
│  │ │   Vector Embeddings     │ │    │ │   Gemini API Client     │ │         │
│  │ │   • text-embedding-004  │ │    │ │   • gemini-1.5-flash    │ │         │
│  │ │   • 768 dimensions      │ │    │ │   • Context analysis    │ │         │
│  │ │   • LRU caching         │ │    │ │   • Role-based prompts  │ │         │
│  │ └─────────────────────────┘ │    │ └─────────────────────────┘ │         │
│  │                             │    │                             │         │
│  │ ┌─────────────────────────┐ │    │ ┌─────────────────────────┐ │         │
│  │ │   Semantic Search       │ │    │ │   Fallback Mechanisms   │ │         │
│  │ │   • Matching Engine     │ │    │ │   • Vertex AI models    │ │         │
│  │ │   • Similarity scoring  │ │    │ │   • Error handling      │ │         │
│  │ │   • Result ranking      │ │    │ │   • Graceful degradation│ │         │
│  │ └─────────────────────────┘ │    │ └─────────────────────────┘ │         │
│  └─────────────────────────────┘    └─────────────────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🎯 Agent Responsibilities

### 1. Svami Orchestrator - The Workflow Coordinator

**Primary Role**: Entry point and multi-agent workflow orchestration

**Core Capabilities**:
- **Request Processing**: Receives user queries with role context
- **Workflow Orchestration**: Coordinates query → search → advise → respond pipeline
- **Service Integration**: Manages communication with Janapada and Amatya agents
- **Response Aggregation**: Combines search results with AI-generated advice
- **Health Monitoring**: Real-time status checks of all system components

**Technical Implementation**:
- **Framework**: FastAPI with async/await architecture
- **Middleware**: Guard-Fort for authentication, logging, and security
- **Communication**: JSON-RPC 2.0 client for inter-agent communication
- **Performance**: Sub-3ms orchestration with parallel health checks
- **Security**: Bearer token authentication, CORS, security headers

### 2. Janapada Memory - The Semantic Search Engine

**Primary Role**: Intelligent code search and memory management

**Core Capabilities**:
- **Vector Embeddings**: Converts text to 768-dimensional vectors using Vertex AI
- **Semantic Search**: Finds relevant code snippets using similarity matching
- **Intelligent Caching**: LRU cache with 1000-entry capacity for performance
- **Matching Engine**: Real-time vector search with configurable thresholds
- **Graceful Fallback**: Continues operation when cloud services unavailable

**Technical Implementation**:
- **Embedding Model**: Google Cloud Vertex AI `text-embedding-004`
- **Vector Store**: Vertex AI Matching Engine for production-scale search
- **Caching Strategy**: LRU cache for frequently accessed embeddings
- **API Interface**: JSON-RPC 2.0 with comprehensive error handling
- **Performance**: ~120ms vector search with similarity scoring

### 3. Amatya Role Prompter - The AI Guidance Generator

**Primary Role**: Context-aware advice generation for different developer roles

**Core Capabilities**:
- **Role-Based Prompting**: Tailored advice for Backend Developers, Security Engineers, etc.
- **Context Analysis**: Intelligent analysis of code snippets and user intent
- **Gemini Integration**: Advanced language understanding with Gemini-1.5-Flash
- **Dynamic Prompting**: Constructs specialized prompts based on context
- **Fallback Architecture**: Vertex AI text models when Gemini unavailable

**Technical Implementation**:
- **Primary Model**: Google Gemini-1.5-Flash for advanced reasoning
- **Fallback Models**: Vertex AI text generation models
- **Prompt Engineering**: Dynamic prompt construction with role context
- **API Integration**: Google Generative AI SDK with proper error handling
- **Performance**: ~200ms response generation with streaming support

## 🔄 Multi-Agent Workflow

### Complete Query Processing Pipeline

```
1. Query Reception (Svami Orchestrator)
   ├── Validate input parameters
   ├── Generate unique request ID
   ├── Extract role context
   └── Initiate workflow

2. Semantic Search (Janapada Memory)
   ├── Convert query to vector embedding
   ├── Search vector index for similar code
   ├── Rank results by similarity score
   └── Return top-K relevant snippets

3. AI Guidance Generation (Amatya Role Prompter)
   ├── Analyze code snippets in role context
   ├── Construct specialized prompt
   ├── Generate response via Gemini API
   └── Return tailored advice

4. Response Orchestration (Svami Orchestrator)
   ├── Combine search results with AI advice
   ├── Add source attribution
   ├── Calculate confidence scores
   └── Return comprehensive response
```

### Request Flow Example

```json
// 1. User Query Input
{
  "question": "How do I implement secure authentication middleware?",
  "role": "backend_developer"
}

// 2. Janapada Search Request
{
  "jsonrpc": "2.0",
  "method": "search",
  "params": {"query": "authentication middleware", "k": 5},
  "id": "req_12345"
}

// 3. Janapada Search Response
{
  "jsonrpc": "2.0",
  "result": {
    "snippets": [
      {
        "file_path": "src/guard_fort/middleware.py",
        "content": "def authenticate_request(token: str) -> bool:",
        "similarity_score": 0.89
      }
    ]
  },
  "id": "req_12345"
}

// 4. Amatya Advice Request
{
  "jsonrpc": "2.0",
  "method": "advise",
  "params": {
    "role": "backend_developer",
    "chunks": [{"file_path": "...", "content": "..."}]
  },
  "id": "req_12345"
}

// 5. Final Orchestrated Response
{
  "answer": "As a Backend Developer, here's how to implement...",
  "sources": ["src/guard_fort/middleware.py"],
  "request_id": "req_12345",
  "confidence_score": 0.92
}
```

## 🛠️ Technical Infrastructure

### Communication Protocol

**JSON-RPC 2.0 Standard**
- **Standardized Messaging**: All inter-agent communication uses JSON-RPC 2.0
- **Error Handling**: Comprehensive error codes and structured responses
- **Request Tracing**: Unique request IDs for distributed system debugging
- **Type Safety**: Full Pydantic validation for all message payloads

### Data Models

**Core Communication Models**:
```python
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    role: str = Field(default="developer")

class AnswerResponse(BaseModel):
    answer: str = Field(..., description="AI-generated response")
    sources: List[str] = Field(default_factory=list)
    request_id: str = Field(..., description="Request tracing ID")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)

class Snippet(BaseModel):
    file_path: str = Field(..., description="Source file path")
    content: str = Field(..., description="Code snippet content")
    similarity_score: Optional[float] = Field(None, ge=0.0, le=1.0)
```

### Security Architecture

**Multi-Layer Security**:
- **Authentication**: Bearer token and API key support
- **Authorization**: Role-based access control
- **Input Validation**: Comprehensive Pydantic model validation
- **Security Headers**: CORS, CSP, XSS protection, HSTS
- **Error Sanitization**: No internal details exposed in responses

### Performance Optimization

**Caching Strategy**:
- **Embedding Cache**: LRU cache for vector embeddings (1000 entries)
- **Response Cache**: Intelligent caching of AI-generated responses
- **Connection Pooling**: Efficient HTTP client connection management

**Async Architecture**:
- **Non-Blocking I/O**: Full async/await implementation
- **Parallel Processing**: Concurrent health checks and service calls
- **Resource Management**: Proper cleanup and connection lifecycle

## 🚀 Deployment Architecture

### Container Orchestration

```yaml
# docker-compose.yml structure
services:
  svami-orchestrator:
    ports: ["8080:8080"]
    depends_on: [janapada-memory, amatya-role-prompter]

  janapada-memory:
    ports: ["8081:8081"]
    environment:
      - GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json

  amatya-role-prompter:
    ports: ["8082:8082"]
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
```

### Service Discovery

**Agent Manifest Endpoints**:
- `GET /.well-known/agent.json` - Service capability advertisement
- `GET /health` - Basic health status
- `GET /health/detailed` - Comprehensive component status

### Monitoring & Observability

**Health Monitoring**:
```python
# Health check response structure
{
  "status": "healthy|degraded|unhealthy",
  "components": {
    "embedding_model": "ready|unavailable",
    "matching_engine": "ready|unavailable",
    "vector_search": true|false
  },
  "performance": {
    "response_time_ms": 1.2,
    "cache_hit_rate": 0.85
  }
}
```

**Structured Logging**:
- **JSON Format**: Machine-readable logs with request correlation
- **Request Tracing**: Unique IDs propagated through entire workflow
- **Performance Metrics**: Response times, cache hit rates, error counts
- **Security Events**: Authentication failures, rate limiting, suspicious activity

## 🔧 Google Cloud Integration

### Vertex AI Services

**Text Embedding Model**:
- **Model**: `text-embedding-004`
- **Dimensions**: 768
- **Performance**: ~130ms per embedding generation
- **Caching**: LRU cache for frequently accessed embeddings

**Matching Engine**:
- **Index Type**: Approximate Nearest Neighbors (ANN)
- **Similarity Metric**: Cosine similarity
- **Performance**: ~120ms for vector search queries
- **Scalability**: Handles millions of vectors with sub-second response

**Gemini API Integration**:
- **Model**: `gemini-1.5-flash`
- **Use Case**: Advanced reasoning and context analysis
- **Performance**: ~200ms response generation
- **Fallback**: Vertex AI text models for reliability

### Authentication & Security

**Service Account Configuration**:
```json
{
  "type": "service_account",
  "project_id": "konveyn2ai",
  "private_key_id": "...",
  "client_email": "konveyn2ai-vertex-ai@konveyn2ai.iam.gserviceaccount.com"
}
```

**Required IAM Roles**:
- `roles/aiplatform.user` - Vertex AI access
- `roles/ml.developer` - Matching Engine operations
- `roles/serviceusage.serviceUsageConsumer` - API usage

## 📊 Performance Characteristics

### Response Time Metrics

| Component | Operation | Average Time | 95th Percentile |
|-----------|-----------|--------------|-----------------|
| Svami Orchestrator | Full workflow | 450ms | 800ms |
| Janapada Memory | Vector search | 120ms | 200ms |
| Amatya Role Prompter | AI generation | 200ms | 350ms |
| Health Checks | Parallel execution | 1ms | 3ms |

### Scalability Considerations

**Horizontal Scaling**:
- **Stateless Design**: All agents are stateless and horizontally scalable
- **Load Balancing**: Multiple instances can run behind load balancers
- **Database Independence**: No persistent storage requirements

**Vertical Scaling**:
- **Memory Usage**: Efficient with LRU caching and connection pooling
- **CPU Utilization**: Async architecture maximizes CPU efficiency
- **Network I/O**: Optimized with connection pooling and keep-alive

## 🛡️ Resilience & Reliability

### Fault Tolerance

**Graceful Degradation**:
- **Service Failures**: System continues with partial functionality
- **API Timeouts**: Configurable timeouts with retry mechanisms
- **Model Unavailability**: Fallback to alternative models

**Error Handling**:
- **Circuit Breaker Pattern**: Prevents cascade failures
- **Retry Logic**: Exponential backoff for transient failures
- **Error Propagation**: Structured error responses with context

### Disaster Recovery

**Backup Strategies**:
- **Configuration**: Environment-based configuration management
- **State Recovery**: Stateless design enables rapid recovery
- **Service Restoration**: Health checks enable automated recovery detection

---

**Architecture designed for production scalability, reliability, and maintainability**
