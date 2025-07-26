"""
Core advisor service for LLM-powered advice generation.

This module implements the main business logic for generating role-specific
advice using Google Cloud Vertex AI text models.
"""

import logging
import asyncio
from typing import Optional, List
import vertexai
from vertexai.language_models import TextGenerationModel

# Import common modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from common.models import AdviceRequest, Snippet
from .config import AmataConfig
from .prompts import PromptConstructor

logger = logging.getLogger(__name__)


class AdvisorService:
    """Service for generating role-specific advice using Vertex AI."""
    
    def __init__(self, config: AmataConfig):
        """Initialize the advisor service."""
        self.config = config
        self.llm_model: Optional[TextGenerationModel] = None
        self.prompt_constructor = PromptConstructor()
        self._initialized = False
        
        logger.info(f"AdvisorService initialized with model: {config.model_name}")
    
    async def initialize(self):
        """Initialize Vertex AI and load the model."""
        try:
            # Initialize Vertex AI
            vertexai.init(
                project=self.config.project_id,
                location=self.config.location
            )
            logger.info(f"Vertex AI initialized for project {self.config.project_id}")
            
            # Load the text generation model
            self.llm_model = TextGenerationModel.from_pretrained(self.config.model_name)
            logger.info(f"Loaded model: {self.config.model_name}")
            
            self._initialized = True
            logger.info("AdvisorService initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize AdvisorService: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources."""
        self._initialized = False
        self.llm_model = None
        logger.info("AdvisorService cleanup complete")
    
    async def is_healthy(self) -> bool:
        """Check if the service is healthy and ready."""
        return self._initialized and self.llm_model is not None
    
    async def generate_advice(self, request: AdviceRequest) -> str:
        """
        Generate role-specific advice based on the request.
        
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
                role=request.role,
                chunks=request.chunks
            )
            
            logger.info(f"Generating advice for role '{request.role}' with {len(request.chunks)} chunks")
            
            # Generate response using Vertex AI
            response = await self._generate_with_retry(prompt)
            
            if response and response.text:
                logger.info("Successfully generated advice")
                return response.text.strip()
            else:
                logger.warning("Empty response from LLM, using fallback")
                return self._generate_fallback_response(request.role, request.chunks)
                
        except Exception as e:
            logger.error(f"Error generating advice: {e}")
            # Return fallback response on error
            return self._generate_fallback_response(request.role, request.chunks)
    
    async def _generate_with_retry(self, prompt: str, max_retries: int = 3):
        """Generate response with retry logic."""
        for attempt in range(max_retries):
            try:
                # Run the synchronous predict method in a thread pool
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.llm_model.predict(
                        prompt,
                        **self.config.get_vertex_ai_config()
                    )
                )
                return response
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
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
