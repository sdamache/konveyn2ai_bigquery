#!/usr/bin/env python3
"""
Script to fix async_client fixture issues in integration tests.
Replaces async_client parameter usage with direct httpx.AsyncClient creation.
"""

import re


def fix_integration_tests():
    """Fix async_client usage in integration tests."""

    file_path = "tests/integration/test_service_interactions.py"

    with open(file_path) as f:
        content = f.read()

    # Pattern 1: Remove async_client from function parameters
    # Match function definitions that include async_client parameter
    def_pattern = (
        r"(async def test_[^(]+\(\s*self,\s*)([^)]*?)async_client,?\s*([^)]*?)\):"
    )

    def replace_function_def(match):
        prefix = match.group(1)
        before_params = match.group(2).strip()
        after_params = match.group(3).strip()

        # Clean up parameter list
        params = []
        if before_params and before_params != ",":
            params.append(before_params.rstrip(",").strip())
        if after_params and after_params != ",":
            params.append(after_params.lstrip(",").strip())

        # Filter out empty strings
        params = [p for p in params if p]

        if params:
            param_str = ", ".join(params)
            return f"{prefix}{param_str}):"
        else:
            return f"{prefix.rstrip(', ')}):"

    content = re.sub(def_pattern, replace_function_def, content)

    # Pattern 2: Replace await async_client.method() calls with async context manager
    # Find all await async_client.post() or await async_client.get() calls
    client_call_pattern = r"(\s+)(response = await async_client\.(post|get)\([^)]+\))"

    def replace_client_call(match):
        indent = match.group(1)
        call = match.group(2)
        method = match.group(3)

        # Extract the method call details
        call_match = re.search(
            r"response = await async_client\.(post|get)\(([^)]+)\)", call
        )
        if call_match:
            method = call_match.group(1)
            args = call_match.group(2)

            return f"""{indent}# Create async client directly in test
{indent}async with httpx.AsyncClient(timeout=30.0) as async_client:
{indent}    response = await async_client.{method}({args})"""

        return match.group(0)  # Return original if no match

    content = re.sub(client_call_pattern, replace_client_call, content)

    # Write the fixed content back
    with open(file_path, "w") as f:
        f.write(content)

    print("Fixed integration tests async_client usage")


if __name__ == "__main__":
    fix_integration_tests()
