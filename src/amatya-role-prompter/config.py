"""
Configuration management for Amatya Role Prompter service.

Handles environment variables, Google Cloud settings, and service configuration
following KonveyN2AI standards.
"""

import os
import logging
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class AmataConfig:
    """Configuration for Amatya Role Prompter service."""

    # Google Cloud Configuration
    project_id: str
    location: str

    # Vertex AI Configuration
    model_name: str
    temperature: float
    max_output_tokens: int
    top_k: int
    top_p: float

    # Gemini API Configuration
    gemini_api_key: str
    gemini_model: str
    use_gemini: bool

    # Service Configuration
    service_name: str
    port: int
    log_level: str

    # Environment
    environment: str
    debug: bool

    def __init__(self):
        """Initialize configuration from environment variables."""
        # Google Cloud Configuration
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "konveyn2ai")
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        # Vertex AI Configuration
        self.model_name = os.getenv("VERTEX_AI_MODEL", "text-bison-001")
        self.temperature = float(os.getenv("VERTEX_AI_TEMPERATURE", "0.2"))
        self.max_output_tokens = int(os.getenv("VERTEX_AI_MAX_TOKENS", "300"))
        self.top_k = int(os.getenv("VERTEX_AI_TOP_K", "40"))
        self.top_p = float(os.getenv("VERTEX_AI_TOP_P", "0.8"))

        # Gemini API Configuration
        # Try GEMINI_API_KEY first, then fall back to GOOGLE_API_KEY for compatibility
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv(
            "GOOGLE_API_KEY", ""
        )
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")
        self.use_gemini = bool(
            self.gemini_api_key
            and self.gemini_api_key
            not in ["", "your_gemini_api_key_here", "your_google_api_key_here"]
        )

        # Service Configuration
        self.service_name = os.getenv("SERVICE_NAME", "amatya-role-prompter")
        self.port = int(os.getenv("PORT", "8001"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

        # Environment
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "true").lower() == "true"

        # Validate required configuration
        self._validate_config()

        logger.info(f"Initialized configuration for {self.service_name}")
        logger.info(f"Project: {self.project_id}, Location: {self.location}")
        logger.info(
            f"Vertex AI Model: {self.model_name}, Environment: {self.environment}"
        )
        logger.info(
            f"Gemini API: {'Enabled' if self.use_gemini else 'Disabled'} (Model: {self.gemini_model})"
        )

    def _validate_config(self):
        """Validate required configuration values."""
        required_fields = [
            ("project_id", self.project_id),
            ("location", self.location),
            ("model_name", self.model_name),
        ]

        missing_fields = []
        for field_name, field_value in required_fields:
            if not field_value:
                missing_fields.append(field_name)

        if missing_fields:
            raise ValueError(
                f"Missing required configuration: {', '.join(missing_fields)}"
            )

    def get_vertex_ai_config(self) -> dict:
        """Get Vertex AI model configuration."""
        return {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }

    def get_gemini_config(self) -> dict:
        """Get Gemini API model configuration."""
        return {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"
