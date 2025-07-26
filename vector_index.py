#!/usr/bin/env python3
"""
Google Cloud AI Platform Vector Index Setup for KonveyN2AI
Creates and configures vector index for the three-component architecture
"""

import os
import sys
from google.cloud import aiplatform
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project configuration from CLAUDE.md
PROJECT_ID = "konveyn2ai"
LOCATION = "us-central1"
VECTOR_DIMENSIONS = 3072
APPROXIMATE_NEIGHBORS_COUNT = 150
DISTANCE_MEASURE_TYPE = "COSINE_DISTANCE"

def check_credentials():
    """Check if Google Cloud credentials are properly configured"""
    try:
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        # Test credentials by listing indexes
        list(aiplatform.MatchingEngineIndex.list())
        print("✅ Google Cloud credentials are properly configured")
        return True
    except Exception as e:
        print(f"❌ Google Cloud credentials issue: {str(e)}")
        print("\nTo fix this:")
        print("1. Install Google Cloud CLI: https://cloud.google.com/sdk/docs/install")
        print("2. Run: gcloud auth application-default login")
        print("3. Or set GOOGLE_APPLICATION_CREDENTIALS environment variable")
        return False

def list_existing_indexes():
    """List all existing vector indexes in the project"""
    try:
        indexes = list(aiplatform.MatchingEngineIndex.list())
        print(f"Existing vector indexes in {PROJECT_ID}:")
        
        if not indexes:
            print("  No existing indexes found")
            return []
        
        for index in indexes:
            print(f"  - {index.display_name} ({index.resource_name})")
            
        return indexes
        
    except Exception as e:
        print(f"Error listing indexes: {str(e)}")
        return []

def create_vector_index(index_display_name="konveyn2ai-code-index"):
    """Create vector index for the KonveyN2AI three-component architecture"""
    try:
        print(f"🔧 Creating vector index: {index_display_name}")
        
        # Create a 3072-dim index for code/doc embeddings, using cosine similarity
        index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=index_display_name,
            dimensions=VECTOR_DIMENSIONS,
            distance_measure_type=DISTANCE_MEASURE_TYPE,
            approximate_neighbors_count=APPROXIMATE_NEIGHBORS_COUNT,
            leaf_node_embedding_count=500,
            leaf_nodes_to_search_percent=10,
        )
        
        print(f"✅ Vector index created successfully:")
        print(f"   Name: {index.display_name}")
        print(f"   Resource Name: {index.resource_name}")
        print(f"   Dimensions: {VECTOR_DIMENSIONS}")
        print(f"   Distance Measure: {DISTANCE_MEASURE_TYPE}")
        
        return index
        
    except Exception as e:
        print(f"❌ Error creating vector index: {str(e)}")
        raise

def main():
    """Main function to set up Google Cloud AI Platform vector index"""
    print("🚀 KonveyN2AI - Google Cloud AI Platform Vector Index Setup")
    print("=" * 60)
    
    # Check credentials first
    if not check_credentials():
        sys.exit(1)
    
    # Initialize AI Platform
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    print(f"Initialized AI Platform for project: {PROJECT_ID} in {LOCATION}")
    
    # List existing indexes
    existing_indexes = list_existing_indexes()
    
    # Check if index already exists
    index_name = "konveyn2ai-code-index"
    existing_index = None
    
    for index in existing_indexes:
        if index.display_name == index_name:
            existing_index = index
            break
    
    if existing_index:
        print(f"\n✅ Vector index '{index_name}' already exists")
        print(f"   Resource Name: {existing_index.resource_name}")
    else:
        print(f"\n🔧 Creating new vector index: {index_name}")
        try:
            new_index = create_vector_index(index_name)
            print(f"✅ Vector index setup completed successfully! Index: {new_index.name}")
        except Exception as e:
            print(f"❌ Failed to create vector index: {str(e)}")
            sys.exit(1)
    
    print("\n📋 Next Steps for KonveyN2AI Architecture:")
    print("1. Configure .env file with GOOGLE_API_KEY")
    print("2. Implement embedding generation in janapada-memory/")
    print("3. Create vector search in amatya-role-prompter/")
    print("4. Set up orchestration in svami-orchestrator/")
    print("5. Test integration with sample data")

if __name__ == "__main__":
    main()

