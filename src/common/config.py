"""
Configuration management for KonveyN2AI.

This module handles loading and managing configuration from environment variables
and provides a centralized configuration interface for all components.
"""

import os
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for KonveyN2AI application."""

    # API Keys
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    PERPLEXITY_API_KEY: Optional[str] = os.getenv("PERPLEXITY_API_KEY")

    # Google Cloud Configuration
    GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "konveyn2ai")
    GOOGLE_CLOUD_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    # Application Configuration
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")

    # Vector Index Configuration
    VECTOR_DIMENSIONS: int = int(os.getenv("VECTOR_DIMENSIONS", "768"))
    SIMILARITY_METRIC: str = os.getenv("SIMILARITY_METRIC", "cosine")
    APPROXIMATE_NEIGHBORS_COUNT: int = int(
        os.getenv("APPROXIMATE_NEIGHBORS_COUNT", "150")
    )

    # Service URLs
    AMATYA_SERVICE_URL: str = os.getenv("AMATYA_SERVICE_URL", "http://localhost:8001")
    JANAPADA_SERVICE_URL: str = os.getenv(
        "JANAPADA_SERVICE_URL", "http://localhost:8002"
    )
    SVAMI_SERVICE_URL: str = os.getenv("SVAMI_SERVICE_URL", "http://localhost:8003")

    @classmethod
    def validate_required_keys(cls) -> None:
        """Validate that required API keys are present."""
        required_keys = ["ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
        missing_keys = []

        for key in required_keys:
            if not getattr(cls, key):
                missing_keys.append(key)

        if missing_keys:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_keys)}. "
                "Please check your .env file."
            )

    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development environment."""
        return cls.ENVIRONMENT == "development"

    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production environment."""
        return cls.ENVIRONMENT == "production"


# Global configuration instance
config = Config()
