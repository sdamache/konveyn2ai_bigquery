#!/usr/bin/env python3
"""
Credential Testing Script for KonveyN2AI Development
Tests Google Cloud authentication and service access
"""

import os
import sys

import google.auth
from google.cloud import aiplatform, artifactregistry_v1, run_v2

# Load environment variables if .env.dev exists
try:
    from dotenv import load_dotenv
    if os.path.exists('.env.dev'):
        load_dotenv('.env.dev')
        print("📝 Loaded environment variables from .env.dev file")
    elif os.path.exists('.env'):
        load_dotenv('.env')
        print("📝 Loaded environment variables from .env file")
    else:
        print("📝 No .env.dev or .env file found, using system environment variables")
except ImportError:
    print("⚠️  python-dotenv not installed, using system environment variables")

def test_basic_credentials():
    """Test basic Google Cloud authentication"""
    print("\n🔐 Testing Basic Credentials...")

    try:
        credentials, project = google.auth.default()
        print("✅ Default credentials found")
        print(f"   Auth type: {type(credentials).__name__}")
        print(f"   Project: {project}")

        # Check if using service account or user credentials
        if hasattr(credentials, 'service_account_email'):
            print(f"   Service Account: {credentials.service_account_email}")

        return True, project

    except Exception as e:
        print(f"❌ Basic credential test failed: {e}")
        return False, None

def test_vertex_ai_access():
    """Test Vertex AI permissions (Amatya, Janapada components)"""
    print("\n🤖 Testing Vertex AI Access...")

    try:
        # Initialize AI Platform
        aiplatform.init(project='konveyn2ai', location='us-central1')
        print("✅ AI Platform initialized successfully")

        # Test vector index access
        indexes = list(aiplatform.MatchingEngineIndex.list())
        print(f"✅ Vector indexes accessible: {len(indexes)} found")

        for idx in indexes:
            if 'konveyn2ai-code-index' in idx.display_name:
                print(f"   ✅ Main index found: {idx.display_name}")
                break

        return True

    except Exception as e:
        print(f"❌ Vertex AI access failed: {e}")
        print("   This is required for Amatya and Janapada components")
        return False

def test_cloud_run_access():
    """Test Cloud Run permissions (Svami component)"""
    print("\n🏃 Testing Cloud Run Access...")

    try:
        client = run_v2.ServicesClient()
        parent = "projects/konveyn2ai/locations/us-central1"
        services = list(client.list_services(parent=parent))
        print(f"✅ Cloud Run services accessible: {len(services)} found")

        return True

    except Exception as e:
        print(f"❌ Cloud Run access failed: {e}")
        print("   This is required for Svami Orchestrator component")
        return False

def test_artifact_registry_access():
    """Test Artifact Registry permissions"""
    print("\n📦 Testing Artifact Registry Access...")

    try:
        client = artifactregistry_v1.ArtifactRegistryClient()
        parent = "projects/konveyn2ai/locations/us-central1"
        repos = list(client.list_repositories(parent=parent))
        print(f"✅ Artifact Registry accessible: {len(repos)} repositories found")

        for repo in repos:
            if 'konveyn2ai-repo' in repo.name:
                print(f"   ✅ Main repository found: {repo.name.split('/')[-1]}")
                break

        return True

    except Exception as e:
        print(f"❌ Artifact Registry access failed: {e}")
        print("   This is required for container deployments")
        return False

def test_environment_variables():
    """Test required environment variables"""
    print("\n🌍 Testing Environment Variables...")

    required_vars = [
        'PROJECT_ID',
        'REGION',
        'VECTOR_INDEX_ID'
    ]

    optional_vars = [
        'GOOGLE_API_KEY',
        'GOOGLE_GEMINI_API_KEY',
        'GOOGLE_APPLICATION_CREDENTIALS'
    ]

    missing_required = []

    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")
            missing_required.append(var)

    for var in optional_vars:
        value = os.getenv(var)
        if value:
            if 'KEY' in var:
                print(f"✅ {var}: ***[REDACTED]***")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"⚠️  {var}: Not set (optional)")

    return len(missing_required) == 0

def main():
    """Main test runner"""
    print("🚀 KonveyN2AI Credential Test Suite")
    print("=" * 50)

    tests_passed = 0
    total_tests = 5

    # Test 1: Basic credentials
    cred_ok, project = test_basic_credentials()
    if cred_ok:
        tests_passed += 1

    # Test 2: Environment variables
    env_ok = test_environment_variables()
    if env_ok:
        tests_passed += 1

    # Test 3: Vertex AI access
    vertex_ok = test_vertex_ai_access()
    if vertex_ok:
        tests_passed += 1

    # Test 4: Cloud Run access
    run_ok = test_cloud_run_access()
    if run_ok:
        tests_passed += 1

    # Test 5: Artifact Registry access
    artifact_ok = test_artifact_registry_access()
    if artifact_ok:
        tests_passed += 1

    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 All tests passed! You're ready for development.")
        print("\n📋 Next Steps:")
        print("   1. Start implementing the three-component architecture")
        print("   2. Run individual component tests")
        print("   3. Test inter-service communication")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the setup:")
        print("\n🔧 Troubleshooting:")
        if not cred_ok:
            print("   - Run: gcloud auth application-default login")
        if not env_ok:
            print("   - Copy .env.example to .env and fill in values")
        if not vertex_ok:
            print("   - Check Vertex AI service account permissions")
        if not run_ok:
            print("   - Check Cloud Run service account permissions")
        if not artifact_ok:
            print("   - Check Artifact Registry access")
        return 1

if __name__ == "__main__":
    sys.exit(main())
