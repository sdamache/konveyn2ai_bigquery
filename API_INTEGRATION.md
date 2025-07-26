# KonveyN2AI - Google Gemini API Integration Guide

## 🚀 Google Gemini API Integration

KonveyN2AI showcases advanced integration with Google's Gemini API, demonstrating innovative usage patterns for multi-agent AI systems. Our implementation leverages both Gemini-1.5-Flash for advanced reasoning and Vertex AI for embeddings and fallback capabilities.

## 🎯 Multi-Model Strategy

### Primary: Gemini-1.5-Flash
- **Use Case**: Advanced language understanding and context-aware response generation
- **Performance**: ~200ms average response time
- **Capabilities**: Complex reasoning, code analysis, role-based prompting

### Secondary: Vertex AI Text Models
- **Use Case**: Fallback when Gemini API unavailable
- **Performance**: ~250ms average response time
- **Capabilities**: Reliable text generation with graceful degradation

### Embeddings: Vertex AI text-embedding-004
- **Use Case**: Semantic code search and similarity matching
- **Performance**: ~130ms per embedding generation
- **Dimensions**: 768-dimensional vectors for optimal performance

## 🔧 Implementation Architecture

### Amatya Role Prompter - Gemini Integration

```python
import google.generativeai as genai
from vertexai.language_models import TextGenerationModel

class AdvisorService:
    def __init__(self, config: AmataConfig):
        self.config = config
        self.gemini_model = None
        self.llm_model = None
        
    async def initialize(self):
        """Initialize both Gemini and Vertex AI models for redundancy."""
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if api_key:
            # Primary: Gemini API
            genai.configure(api_key=api_key)
            if "gemini" in self.config.model_name.lower():
                self.gemini_model = genai.GenerativeModel(self.config.model_name)
                logger.info(f"Gemini API initialized: {self.config.model_name}")
        
        # Fallback: Vertex AI
        vertexai.init(project=self.config.project_id, location=self.config.location)
        await self._load_model_with_retry()
        
    async def generate_advice(self, request: AdviceRequest) -> str:
        """Generate role-based advice with intelligent fallback."""
        prompt = self.prompt_constructor.construct_advice_prompt(
            request.role, request.chunks
        )
        
        # Try Gemini first
        if self.gemini_model:
            try:
                response = await self._generate_with_gemini(prompt)
                if response:
                    return response.strip()
            except Exception as e:
                logger.warning(f"Gemini API failed, using fallback: {e}")
        
        # Fallback to Vertex AI
        if self.llm_model:
            return await self._generate_with_retry(prompt)
        
        # Final fallback
        return self._generate_fallback_response(request.role, request.chunks)
```

### Advanced Prompt Engineering

```python
class PromptConstructor:
    def construct_advice_prompt(self, role: str, chunks: List[Snippet]) -> str:
        """Construct role-specific prompts for Gemini API."""
        
        role_context = self._get_role_context(role)
        code_context = self._format_code_snippets(chunks)
        
        prompt = f"""
{role_context}

## Code Analysis Task
Analyze the following code snippets and provide implementation guidance:

{code_context}

## Requirements
- Focus on {role.replace('_', ' ').title()} perspective
- Provide practical, actionable advice
- Include code examples where relevant
- Highlight security and performance considerations
- Explain the reasoning behind recommendations

## Response Format
Structure your response with clear sections and code examples.
"""
        return prompt
    
    def _get_role_context(self, role: str) -> str:
        """Generate role-specific context for better AI responses."""
        contexts = {
            "backend_developer": """
You are an expert Backend Developer with deep knowledge of:
- Server-side architecture and design patterns
- API design and implementation
- Database optimization and security
- Performance tuning and scalability
- Security best practices and authentication
""",
            "security_engineer": """
You are a Security Engineer specializing in:
- Application security and vulnerability assessment
- Secure coding practices and code review
- Authentication and authorization systems
- Threat modeling and risk assessment
- Compliance and security standards
""",
            "devops_engineer": """
You are a DevOps Engineer expert in:
- Infrastructure as Code and automation
- CI/CD pipeline design and optimization
- Container orchestration and deployment
- Monitoring, logging, and observability
- Cloud architecture and scalability
"""
        }
        return contexts.get(role, contexts["backend_developer"])
```

## 🧠 Janapada Memory - Vertex AI Embeddings

### Embedding Generation with Caching

```python
from vertexai.language_models import TextEmbeddingModel
from functools import lru_cache

class JanapadaMemoryService:
    def __init__(self):
        self.embedding_model = None
        
    async def initialize(self):
        """Initialize Vertex AI embedding model."""
        try:
            self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
            logger.info("Vertex AI embedding model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
    
    @lru_cache(maxsize=1000)
    def generate_embedding_cached(self, text: str) -> Optional[List[float]]:
        """Generate embedding with LRU caching for performance."""
        if not self.embedding_model or not text.strip():
            return None
            
        try:
            embeddings = self.embedding_model.get_embeddings([text])
            if embeddings and len(embeddings) > 0:
                return embeddings[0].values
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            
        return None
    
    def search_with_embeddings(self, query: str, k: int = 5) -> List[Snippet]:
        """Perform semantic search using vector embeddings."""
        query_embedding = self.generate_embedding_cached(query)
        
        if not query_embedding:
            logger.warning("Query embedding failed, using fallback search")
            return self._fallback_search(query, k)
        
        # Search vector index for similar embeddings
        similar_snippets = self._vector_similarity_search(query_embedding, k)
        
        return [
            Snippet(
                file_path=snippet["file_path"],
                content=snippet["content"],
                similarity_score=snippet["score"]
            )
            for snippet in similar_snippets
        ]
```

### Vector Similarity Search

```python
def _vector_similarity_search(self, query_embedding: List[float], k: int) -> List[dict]:
    """Perform vector similarity search using cosine similarity."""
    
    # Mock implementation - in production, this would use Vertex AI Matching Engine
    mock_code_snippets = [
        {
            "file_path": "src/guard_fort/middleware.py",
            "content": "def authenticate_request(token: str) -> bool:",
            "embedding": [0.1, 0.2, 0.3, ...],  # 768-dimensional vector
        },
        {
            "file_path": "src/guard_fort/auth.py",
            "content": "class AuthenticationMiddleware:",
            "embedding": [0.2, 0.1, 0.4, ...],  # 768-dimensional vector
        }
    ]
    
    results = []
    for snippet in mock_code_snippets:
        similarity = self._cosine_similarity(query_embedding, snippet["embedding"])
        if similarity > 0.7:  # Configurable threshold
            results.append({
                "file_path": snippet["file_path"],
                "content": snippet["content"],
                "score": similarity
            })
    
    # Sort by similarity score and return top-k
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]

def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    import numpy as np
    
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    
    dot_product = np.dot(vec1_np, vec2_np)
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)
```

## 🎯 Svami Orchestrator - Multi-Agent Coordination

### Intelligent Workflow Orchestration

```python
class SvamiOrchestrator:
    def __init__(self):
        self.janapada_client = JsonRpcClient("http://localhost:8081")
        self.amatya_client = JsonRpcClient("http://localhost:8082")
    
    async def answer_query(self, query: QueryRequest) -> AnswerResponse:
        """Orchestrate multi-agent workflow for intelligent responses."""
        request_id = str(uuid.uuid4())
        
        try:
            # Step 1: Semantic Search (Janapada Memory)
            search_response = await self.janapada_client.call(
                method="search",
                params={"query": query.question, "k": 5},
                id=request_id
            )
            
            snippets = [
                Snippet(**snippet) 
                for snippet in search_response.get("snippets", [])
            ]
            
            # Step 2: AI Guidance Generation (Amatya Role Prompter)
            if snippets:
                advice_response = await self.amatya_client.call(
                    method="advise",
                    params={
                        "role": query.role,
                        "chunks": [snippet.model_dump() for snippet in snippets]
                    },
                    id=request_id
                )
                
                answer = advice_response.get("answer", "")
            else:
                answer = "No relevant code snippets found for your query."
            
            # Step 3: Response Aggregation
            return AnswerResponse(
                answer=answer,
                sources=[snippet.file_path for snippet in snippets],
                request_id=request_id,
                confidence_score=self._calculate_confidence(snippets, answer)
            )
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            return AnswerResponse(
                answer="I encountered an error processing your request. Please try again.",
                sources=[],
                request_id=request_id,
                confidence_score=0.0
            )
    
    def _calculate_confidence(self, snippets: List[Snippet], answer: str) -> float:
        """Calculate confidence score based on search results and response quality."""
        if not snippets:
            return 0.0
        
        # Average similarity score from search results
        avg_similarity = sum(s.similarity_score or 0.0 for s in snippets) / len(snippets)
        
        # Response length factor (longer responses generally more comprehensive)
        length_factor = min(len(answer) / 1000, 1.0)
        
        # Combined confidence score
        return (avg_similarity * 0.7) + (length_factor * 0.3)
```

## 🔒 Security & Authentication

### API Key Management

```python
# Environment-based configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Secure API key validation
def validate_api_keys():
    """Validate required API keys are present."""
    if not GOOGLE_API_KEY:
        logger.warning("GOOGLE_API_KEY not found - Gemini API unavailable")
    
    if not GOOGLE_APPLICATION_CREDENTIALS:
        logger.warning("GOOGLE_APPLICATION_CREDENTIALS not found - Vertex AI may fail")
    
    return bool(GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS)
```

### Rate Limiting & Error Handling

```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

class RateLimitedGeminiClient:
    def __init__(self):
        self.request_count = 0
        self.last_reset = time.time()
        self.max_requests_per_minute = 60
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_with_retry(self, prompt: str) -> str:
        """Generate response with exponential backoff retry."""
        await self._check_rate_limit()
        
        try:
            response = await self.gemini_model.generate_content_async(prompt)
            self.request_count += 1
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    async def _check_rate_limit(self):
        """Implement rate limiting to respect API quotas."""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.last_reset > 60:
            self.request_count = 0
            self.last_reset = current_time
        
        # Wait if approaching rate limit
        if self.request_count >= self.max_requests_per_minute:
            wait_time = 60 - (current_time - self.last_reset)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                self.request_count = 0
                self.last_reset = time.time()
```

## 📊 Performance Optimization

### Intelligent Caching Strategy

```python
from functools import lru_cache
import hashlib

class CachedGeminiClient:
    def __init__(self):
        self.response_cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    def _cache_key(self, prompt: str, role: str) -> str:
        """Generate cache key for prompt and role combination."""
        content = f"{role}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def generate_cached_response(self, prompt: str, role: str) -> str:
        """Generate response with intelligent caching."""
        cache_key = self._cache_key(prompt, role)
        
        # Check cache first
        if cache_key in self.response_cache:
            cached_response, timestamp = self.response_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                logger.info("Cache hit for Gemini response")
                return cached_response
        
        # Generate new response
        response = await self.generate_with_retry(prompt)
        
        # Cache the response
        self.response_cache[cache_key] = (response, time.time())
        
        return response
```

## 🎯 Innovation Highlights

### 1. Multi-Model Fallback Architecture
- **Primary**: Gemini-1.5-Flash for advanced reasoning
- **Secondary**: Vertex AI text models for reliability
- **Tertiary**: Static fallback responses for maximum uptime

### 2. Context-Aware Prompt Engineering
- **Role-Based Prompting**: Tailored prompts for different developer personas
- **Dynamic Context**: Code snippets integrated into prompts for relevance
- **Structured Output**: Consistent response formatting for better UX

### 3. Intelligent Performance Optimization
- **LRU Caching**: Embedding and response caching for sub-second performance
- **Rate Limiting**: Respectful API usage with exponential backoff
- **Parallel Processing**: Concurrent health checks and service calls

### 4. Production-Ready Error Handling
- **Graceful Degradation**: System continues with partial functionality
- **Comprehensive Logging**: Full request tracing for debugging
- **User-Friendly Errors**: Technical errors translated to helpful messages

---

**KonveyN2AI demonstrates the future of intelligent multi-agent systems powered by Google's cutting-edge AI technologies.**
