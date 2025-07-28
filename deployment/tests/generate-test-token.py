#!/usr/bin/env python3
"""
Test Token Generator for KonveyN2AI Testing

Generates secure test tokens for use in CI/CD testing instead of static demo tokens.
This improves security by using dynamic tokens with proper format validation.
"""

import hashlib
import secrets
import time
from datetime import datetime, timedelta


def generate_test_token(prefix: str = "test", validity_hours: int = 24) -> str:
    """
    Generate a secure test token for KonveyN2AI testing.

    Args:
        prefix: Token prefix for identification
        validity_hours: Token validity period in hours

    Returns:
        str: Secure test token in JWT-like format
    """
    # Generate secure random components
    timestamp = int(time.time())
    random_bytes = secrets.token_bytes(16)

    # Create header (base64url-like encoding)
    header = f"{prefix}-{timestamp}"
    header_encoded = header.encode().hex()

    # Create payload with expiration
    expiry = timestamp + (validity_hours * 3600)
    payload = f"exp-{expiry}-rnd-{random_bytes.hex()}"
    payload_encoded = payload.encode().hex()

    # Create signature-like component
    signature_data = f"{header_encoded}.{payload_encoded}".encode()
    signature = hashlib.sha256(signature_data).hexdigest()[:32]

    # Combine in JWT-like format
    token = f"{header_encoded}.{payload_encoded}.{signature}"

    return token


def generate_api_key(prefix: str = "konveyn2ai", length: int = 32) -> str:
    """
    Generate a secure API key for testing.

    Args:
        prefix: API key prefix
        length: Total length of the API key

    Returns:
        str: Secure API key
    """
    # Generate random suffix
    suffix_length = length - len(prefix) - 1  # -1 for separator
    suffix = secrets.token_urlsafe(suffix_length)[:suffix_length]

    return f"{prefix}-{suffix}"


def get_test_credentials() -> dict:
    """
    Get test credentials for current testing session.

    Returns:
        dict: Test credentials including Bearer token and API key
    """
    return {
        "bearer_token": generate_test_token("konveyn2ai-test", 24),
        "api_key": generate_api_key("konveyn2ai-test", 32),
        "demo_tokens": [
            "konveyn2ai-token",  # Accepted by GuardFort
            "hackathon-demo",  # Accepted by GuardFort
            "test-token",  # Accepted by GuardFort
        ],
    }


def main():
    """Generate and display test credentials."""
    print("KonveyN2AI Test Token Generator")
    print("=" * 40)

    credentials = get_test_credentials()

    print(f"Generated Bearer Token: {credentials['bearer_token']}")
    print(f"Generated API Key: {credentials['api_key']}")
    print("\nAccepted Demo Tokens:")
    for token in credentials["demo_tokens"]:
        print(f"  - {token}")

    print(f"\nTokens valid until: {datetime.now() + timedelta(hours=24)}")
    print("\nUsage in tests:")
    print(f'  -H "Authorization: Bearer {credentials["bearer_token"]}"')
    print(f'  -H "Authorization: ApiKey {credentials["api_key"]}"')


if __name__ == "__main__":
    main()
