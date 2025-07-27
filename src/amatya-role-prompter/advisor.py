"""
Core advisor service for LLM-powered advice generation.

This module implements the main business logic for generating role-specific
advice using Google Cloud Vertex AI text models.
"""

import logging
import asyncio
import os
from typing import Optional, List
import vertexai
from vertexai.generative_models import GenerativeModel

# Gemini API imports
try:
    from google import genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Gemini API not available - install google-genai")

# Import common modules
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from common.models import AdviceRequest, Snippet

# Handle both relative and absolute imports
try:
    from .config import AmataConfig
    from .prompts import PromptConstructor
except ImportError:
    from config import AmataConfig
    from prompts import PromptConstructor

logger = logging.getLogger(__name__)


class AdvisorService:
    """Service for generating role-specific advice using Gemini API with Vertex AI fallback."""

    def __init__(self, config: AmataConfig):
        """Initialize the advisor service."""
        self.config = config
        self.llm_model: Optional[GenerativeModel] = None
        self.gemini_client: Optional[genai.Client] = None
        self.prompt_constructor = PromptConstructor()
        self._initialized = False

        logger.info(
            f"AdvisorService initialized - Vertex AI: {config.model_name}, Gemini: {config.use_gemini}"
        )

    async def initialize(self):
        """Initialize Gemini API and Vertex AI with hybrid approach."""
        try:
            # Initialize Gemini API first (primary)
            await self._initialize_gemini()

            # Initialize Vertex AI (fallback)
            await self._initialize_vertex_ai()

            self._initialized = True
            logger.info("AdvisorService initialization complete")

        except Exception as e:
            logger.error(f"Failed to initialize AdvisorService: {e}")
            # For demo purposes, continue with fallback mode
            logger.warning("Continuing in fallback mode")
            self._initialized = True

    async def _initialize_gemini(self):
        """Initialize Gemini API client."""
        if not GEMINI_AVAILABLE:
            logger.warning("Gemini API not available - skipping Gemini initialization")
            return

        if self.config.use_gemini:
            try:
                # Initialize Gemini client with keyword arguments only
                self.gemini_client = genai.Client(api_key=self.config.gemini_api_key)
                logger.info("✅ Gemini client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                self.gemini_client = None
        else:
            logger.info("Gemini API disabled - no API key provided")

    async def _initialize_vertex_ai(self):
        """Initialize Vertex AI as fallback."""
        try:
            # Check for Google Cloud credentials
            credentials_available = self._check_credentials()

            if not credentials_available:
                logger.warning(
                    "Google Cloud credentials not available - Vertex AI fallback disabled"
                )
                return

            # Initialize Vertex AI
            vertexai.init(project=self.config.project_id, location=self.config.location)
            logger.info(f"Vertex AI initialized for project {self.config.project_id}")

            # Load the text generation model with retry logic
            await self._load_model_with_retry()

        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            logger.warning("Vertex AI fallback not available")

    async def cleanup(self):
        """Cleanup resources."""
        self._initialized = False
        self.llm_model = None
        self.gemini_client = None
        logger.info("AdvisorService cleanup complete")

    def _check_credentials(self) -> bool:
        """Check if Google Cloud credentials are available."""
        try:
            # Skip expensive credential checks in test environments
            if os.getenv("PYTEST_CURRENT_TEST") or "pytest" in sys.modules:
                logger.info("Test environment detected - skipping credential checks")
                return False

            # Check for service account key file
            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if credentials_path and os.path.exists(credentials_path):
                logger.info("Found service account credentials file")
                return True

            # Check for default credentials (gcloud auth)
            try:
                import google.auth

                credentials, project = google.auth.default()
                if credentials:
                    logger.info("Found default Google Cloud credentials")
                    return True
            except Exception as e:
                # Intentionally ignore credential check failures - this is expected
                # when running in environments without Google Cloud setup
                logger.debug(f"Default credentials check failed: {e}")  # nosec B110

            # Check for API key
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key and api_key != "your_google_api_key_here":
                logger.info("Found Google API key")
                return True

            logger.warning("No Google Cloud credentials found")
            return False

        except Exception as e:
            logger.warning(f"Error checking credentials: {e}")
            return False

    async def _load_model_with_retry(self, max_retries: int = 3):
        """Load the Vertex AI Gemini model with retry logic."""
        # Skip expensive model loading in test environments
        if os.getenv("PYTEST_CURRENT_TEST") or "pytest" in sys.modules:
            logger.info("Test environment detected - skipping model loading")
            return

        for attempt in range(max_retries):
            try:
                # Initialize Vertex AI Gemini model
                self.llm_model = GenerativeModel(self.config.model_name)
                logger.info(
                    f"✅ Loaded Vertex AI Gemini model: {self.config.model_name}"
                )
                return

            except Exception as e:
                logger.warning(
                    f"Vertex AI model loading attempt {attempt + 1} failed: {e}"
                )
                if attempt == max_retries - 1:
                    logger.error("Failed to load Vertex AI model after all retries")
                    raise
                await asyncio.sleep(2 * (attempt + 1))  # Exponential backoff

    async def is_healthy(self) -> bool:
        """Check if the service is healthy and ready."""
        return self._initialized and (
            self.gemini_client is not None or self.llm_model is not None
        )

    async def generate_advice(self, request: AdviceRequest) -> str:
        """
        Generate role-specific advice using hybrid Gemini-first, Vertex AI fallback approach.

        Args:
            request: AdviceRequest containing role and code chunks

        Returns:
            str: Generated advice in markdown format
        """
        if not self._initialized:
            raise RuntimeError("AdvisorService not initialized")

        try:
            # Construct the prompt
            prompt = self.prompt_constructor.construct_prompt(
                role=request.role, chunks=request.chunks
            )

            logger.info(
                f"Generating advice for role '{request.role}' with {len(request.chunks)} chunks"
            )

            # Try Gemini first (primary)
            if self.gemini_client is not None:
                try:
                    response = await self._generate_with_gemini(prompt)
                    if response:
                        logger.info("✅ Successfully generated advice using Gemini API")
                        return response.strip()
                    else:
                        logger.warning(
                            "Empty response from Gemini, trying Vertex AI fallback"
                        )
                except Exception as e:
                    logger.warning(f"Gemini API failed: {e}, trying Vertex AI fallback")

            # Fallback to Vertex AI
            if self.llm_model is not None:
                try:
                    response = await self._generate_with_vertex_ai(prompt)
                    if response and response.text:
                        logger.info(
                            "✅ Successfully generated advice using Vertex AI fallback"
                        )
                        return response.text.strip()
                    else:
                        logger.warning(
                            "Empty response from Vertex AI, using mock fallback"
                        )
                except Exception as e:
                    logger.warning(f"Vertex AI failed: {e}, using mock fallback")

            # Final fallback - enhanced mock response
            logger.info("Using enhanced mock response (no AI services available)")
            return self._generate_enhanced_mock_response(
                request.role, request.chunks, prompt
            )

        except Exception as e:
            logger.error(f"Error generating advice: {e}")
            # Return fallback response on error
            return self._generate_fallback_response(request.role, request.chunks)

    async def _generate_with_gemini(self, prompt: str) -> Optional[str]:
        """Generate response using Gemini API."""
        if not self.gemini_client:
            return None

        try:
            # Use the new google-genai SDK
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.gemini_client.models.generate_content(
                    model=self.config.gemini_model,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=self.config.temperature,
                        max_output_tokens=self.config.max_output_tokens,
                        top_k=self.config.top_k,
                        top_p=self.config.top_p,
                    ),
                ),
            )

            if response and hasattr(response, "text") and response.text:
                return response.text
            else:
                logger.warning("Empty or invalid response from Gemini API")
                return None

        except Exception as e:
            logger.error(f"Gemini API generation failed: {e}")
            return None

    async def _generate_with_vertex_ai(self, prompt: str):
        """Generate response using Vertex AI (fallback)."""
        return await self._generate_with_retry(prompt)

    async def _generate_with_retry(self, prompt: str, max_retries: int = 3):
        """Generate response using Vertex AI Gemini model with retry logic."""
        for attempt in range(max_retries):
            try:
                # Use Vertex AI Gemini model generate_content method
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.llm_model.generate_content(
                        prompt,
                        generation_config={
                            "temperature": self.config.temperature,
                            "max_output_tokens": self.config.max_output_tokens,
                            "top_k": self.config.top_k,
                            "top_p": self.config.top_p,
                        },
                    ),
                )
                return response

            except Exception as e:
                logger.warning(f"Vertex AI attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff

    def _generate_fallback_response(self, role: str, chunks: List[Snippet]) -> str:
        """
        Generate a fallback response when LLM fails.

        Args:
            role: User role
            chunks: Code snippets

        Returns:
            str: Fallback advice in markdown format
        """
        file_references = [chunk.file_path for chunk in chunks[:3]]

        fallback_advice = f"""# Onboarding Guide for {role.replace('_', ' ').title()}

I apologize, but I'm experiencing technical difficulties generating a detailed response. Here's a basic guide based on the available information:

## Getting Started

1. **Review Project Structure**
   - Start by examining `{file_references[0] if file_references else 'the main project files'}`
   - Understand the overall architecture and organization

2. **Core Functionality**
   - Focus on the implementation in `{file_references[1] if len(file_references) > 1 else 'core modules'}`
   - Review the main business logic and data flow

3. **Implementation Details**
   - Study the specific details in `{file_references[2] if len(file_references) > 2 else 'implementation files'}`
   - Pay attention to patterns and conventions used

## Next Steps

- Set up your development environment
- Run the existing tests to understand expected behavior
- Start with small changes to familiarize yourself with the codebase
- Reach out to the team for guidance on specific areas

Please try your query again later for more detailed, AI-generated guidance.
"""

        logger.info(f"Generated fallback response for role '{role}'")
        return fallback_advice

    def _generate_enhanced_mock_response(
        self, role: str, chunks: List[Snippet], prompt: str
    ) -> str:
        """
        Generate an enhanced mock response for demo purposes.

        Args:
            role: User role
            chunks: Code snippets
            prompt: Constructed prompt (for context)

        Returns:
            str: Enhanced mock advice in markdown format
        """
        file_references = [chunk.file_path for chunk in chunks]
        role_title = role.replace("_", " ").title()

        # Create a more sophisticated mock response based on the role
        if "backend" in role.lower():
            mock_advice = f"""# Backend Developer Onboarding Guide

Welcome to the KonveyN2AI project! As a **{role_title}**, you'll be working with our three-tier architecture.

## 🏗️ Architecture Overview

This project follows a microservices architecture with three main components:

1. **Amatya Role Prompter** - Role-based prompting and user interaction
2. **Janapada Memory** - Memory management and vector embeddings
3. **Svami Orchestrator** - Workflow orchestration and coordination

## 🚀 Getting Started

### 1. Environment Setup
- Review the configuration in `{file_references[0] if file_references else 'config files'}`
- Set up your Google Cloud credentials for Vertex AI integration
- Install dependencies: `pip install -r requirements.txt`

### 2. Core Backend Components
- **FastAPI Applications**: Each component runs as a separate FastAPI service
- **JSON-RPC Communication**: Services communicate using JSON-RPC 2.0 protocol
- **Pydantic Models**: Shared data models in `src/common/models.py`

### 3. Key Files to Review
{chr(10).join([f"- `{fp}` - Core implementation" for fp in file_references[:3]])}

## 🔧 Development Workflow

1. **Local Development**: Each service runs on different ports (8001, 8002, 8003)
2. **Testing**: Use pytest for unit tests, real functionality required
3. **API Documentation**: FastAPI auto-generates docs at `/docs` endpoint

## 📝 Next Steps

- Set up your development environment
- Review the JSON-RPC endpoint implementations
- Test the service communication patterns
- Explore the Google Cloud AI Platform integration

*This is a demo response - full AI-powered guidance available with proper Vertex AI setup.*
"""
        elif "security" in role.lower():
            mock_advice = f"""# Security Engineer Onboarding Guide

Welcome to KonveyN2AI security review! As a **{role_title}**, focus on these security aspects.

## 🔒 Security Architecture

### Authentication & Authorization
- **GuardFort Middleware**: Comprehensive security middleware in `src/guard_fort/`
- **API Key Management**: Secure handling of Google Cloud credentials
- **Request Tracing**: Full request lifecycle tracking

### Key Security Components
{chr(10).join([f"- `{fp}` - Security implementation" for fp in file_references[:3]])}

## 🛡️ Security Checklist

1. **Input Validation**: All Pydantic models provide validation
2. **API Security**: JSON-RPC endpoints with authentication
3. **Secrets Management**: Environment-based configuration
4. **Logging**: Structured logging for audit trails

## 🔍 Areas to Review

- GuardFort middleware configuration
- API endpoint security patterns
- Google Cloud IAM integration
- Request/response sanitization

*This is a demo response - full AI-powered security guidance available with proper Vertex AI setup.*
"""
        else:
            # Generic developer response
            mock_advice = f"""# {role_title} Onboarding Guide

Welcome to KonveyN2AI! Here's your personalized onboarding guide.

## 📋 Project Overview

KonveyN2AI is a hackathon project featuring Google Gemini API integration with a three-tier architecture.

## 🎯 Your Role Focus

As a **{role_title}**, you'll be working with:

### Key Files
{chr(10).join([f"- `{fp}` - Implementation details" for fp in file_references[:3]])}

### Development Environment
- Python 3.10+ with virtual environment
- Google Cloud AI Platform integration
- FastAPI for service architecture

## 🚀 Getting Started

1. **Setup**: Review environment configuration
2. **Architecture**: Understand the three-tier design
3. **Integration**: Learn the JSON-RPC communication patterns
4. **Testing**: Implement real functionality (no mocks for core features)

## 📚 Resources

- Project documentation in `/docs`
- API documentation at service `/docs` endpoints
- Common models in `src/common/`

*This is a demo response - full AI-powered guidance available with proper Vertex AI setup.*
"""

        logger.info(f"Generated enhanced mock response for role '{role}'")
        return mock_advice
