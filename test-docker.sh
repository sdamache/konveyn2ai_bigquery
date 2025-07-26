#!/bin/bash

# Test script for Docker containerization setup
echo "🐳 Testing KonveyN2AI Docker Setup"
echo "=================================="

# Check Docker availability
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi

echo "✅ Docker is installed: $(docker --version)"

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "⚠️  Docker daemon is not running. Please start Docker Desktop."
    echo "📋 You can still review the Docker files created:"
    echo "   - Dockerfile.svami"
    echo "   - Dockerfile.janapada" 
    echo "   - Dockerfile.amatya"
    echo "   - docker-compose.yml"
    echo "   - .dockerignore"
    echo "   - README.docker.md"
    exit 1
fi

echo "✅ Docker daemon is running"

# Test building images
echo ""
echo "🔨 Testing Docker builds..."

# Build Svami image
echo "Building Svami Orchestrator..."
if docker build -f Dockerfile.svami -t konveyn2ai/svami:test . > build-svami.log 2>&1; then
    echo "✅ Svami image built successfully"
else
    echo "❌ Svami image build failed. Check build-svami.log"
fi

# Build Janapada image  
echo "Building Janapada Memory..."
if docker build -f Dockerfile.janapada -t konveyn2ai/janapada:test . > build-janapada.log 2>&1; then
    echo "✅ Janapada image built successfully"
else
    echo "❌ Janapada image build failed. Check build-janapada.log"
fi

# Build Amatya image
echo "Building Amatya Role Prompter..."
if docker build -f Dockerfile.amatya -t konveyn2ai/amatya:test . > build-amatya.log 2>&1; then
    echo "✅ Amatya image built successfully"
else
    echo "❌ Amatya image build failed. Check build-amatya.log"
fi

# Test Docker Compose validation
echo ""
echo "🐙 Testing Docker Compose..."
if docker-compose config > /dev/null 2>&1; then
    echo "✅ docker-compose.yml is valid"
else
    echo "❌ docker-compose.yml has syntax errors"
fi

# Show image sizes if builds succeeded
echo ""
echo "📊 Docker Images:"
docker images | grep konveyn2ai || echo "No konveyn2ai images found"

echo ""
echo "🎉 Docker setup testing complete!"
echo "📚 See README.docker.md for deployment instructions"