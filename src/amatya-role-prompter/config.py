"""
Configuration management for Amatya Role Prompter service.

Handles environment variables, Google Cloud settings, and service configuration
following KonveyN2AI standards with production-ready security defaults.
"""

import logging
import os
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
    cors_origins: list[str]
    host: str

    # Retry Configuration
    max_retries: int
    retry_delay_base: float
    request_timeout: float

    # Environment
    environment: str
    debug: bool

    def __init__(self) -> None:
        """
        Initialize configuration from environment variables.

        Loads configuration with production-safe defaults for security settings
        including CORS origins, host binding, and retry parameters.

        Raises:
            ValueError: If required configuration fields are missing
        """
        # Google Cloud Configuration
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "konveyn2ai")
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        # Vertex AI Configuration
        self.model_name = os.getenv("VERTEX_AI_MODEL", "gemini-2.0-flash-001")
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

        # Environment (needed for other configurations)
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "true").lower() == "true"

        # Service Configuration
        self.service_name = os.getenv("SERVICE_NAME", "amatya-role-prompter")
        self.port = int(os.getenv("PORT", "8001"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

        # CORS Configuration - production-safe defaults
        cors_env = os.getenv("CORS_ORIGINS", "")
        if cors_env:
            self.cors_origins = [origin.strip() for origin in cors_env.split(",")]
        else:
            # Default to localhost for development, empty for production
            self.cors_origins = (
                ["http://localhost:3000", "http://localhost:8080"]
                if self.environment == "development"
                else []
            )

        # Host Configuration - environment-specific defaults
        if self.environment == "production":
            # Production: bind only to localhost for security
            default_host = "127.0.0.1"
        else:
            # Development: bind to all interfaces for convenience
            default_host = "0.0.0.0"  # nosec B104
        self.host = os.getenv("HOST", default_host)

        # Retry Configuration
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.retry_delay_base = float(os.getenv("RETRY_DELAY_BASE", "1.0"))
        self.request_timeout = float(os.getenv("REQUEST_TIMEOUT", "30.0"))

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

    def _validate_config(self) -> None:
        """
        Validate required configuration values.

        Raises:
            ValueError: If any required configuration fields are missing or invalid
        """
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

    def get_vertex_ai_config(self) -> dict[str, float | int]:
        """
        Get Vertex AI model configuration parameters.

        Returns:
            dict: Configuration parameters for Vertex AI model generation
                including temperature, max_output_tokens, top_k, and top_p
        """
        return {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }

    def get_retry_config(self) -> dict[str, float | int]:
        """
        Get retry configuration parameters.

        Returns:
            dict: Retry configuration including max_retries, retry_delay_base,
                and request_timeout for API calls
        """
        return {
            "max_retries": self.max_retries,
            "retry_delay_base": self.retry_delay_base,
            "request_timeout": self.request_timeout,
        }

    def get_security_config(self) -> dict[str, str | list[str]]:
        """
        Get security configuration parameters.

        Returns:
            dict: Security configuration including CORS origins and host binding
        """
        return {
            "cors_origins": self.cors_origins,
            "host": self.host,
            "environment": self.environment,
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
