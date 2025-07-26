#!/usr/bin/env python3
"""
KonveyN2AI Environment Setup Script
Simple environment setup and dependency check for hackathon development
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} completed")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.8+ required")
        return False


def install_dependencies():
    """Install dependencies from requirements.txt"""
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False

    print("📦 Installing dependencies from requirements.txt...")
    cmd = f"{sys.executable} -m pip install -r requirements.txt"
    return run_command(cmd, "Installing dependencies")


def check_environment_variables():
    """Check for required environment variables"""
    required_vars = ["GOOGLE_API_KEY", "ANTHROPIC_API_KEY"]

    optional_vars = ["PERPLEXITY_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"]

    print("🔑 Checking environment variables...")

    missing_required = []
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var} is set")
        else:
            print(f"❌ {var} is missing")
            missing_required.append(var)

    for var in optional_vars:
        if os.getenv(var):
            print(f"✅ {var} is set (optional)")
        else:
            print(f"⚠️  {var} is not set (optional)")

    return len(missing_required) == 0


def create_env_example():
    """Create .env.example if it doesn't exist"""
    env_example_content = """# KonveyN2AI Environment Variables
# Copy this file to .env and fill in your actual values

# Required API Keys
GOOGLE_API_KEY=your_google_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional API Keys
PERPLEXITY_API_KEY=your_perplexity_api_key_here

# Google Cloud Configuration
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
GOOGLE_CLOUD_PROJECT=konveyn2ai
GOOGLE_CLOUD_LOCATION=us-central1

# Development Settings
DEBUG=true
LOG_LEVEL=INFO
"""

    if not Path(".env.example").exists():
        with open(".env.example", "w") as f:
            f.write(env_example_content)
        print("✅ Created .env.example file")
    else:
        print("✅ .env.example already exists")


def main():
    """Main setup function"""
    print("🚀 KonveyN2AI Environment Setup")
    print("=" * 40)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Create .env.example
    create_env_example()

    # Install dependencies
    if install_dependencies():
        print("✅ Dependencies installed successfully")
    else:
        print("❌ Failed to install dependencies")
        sys.exit(1)

    # Check environment variables
    if check_environment_variables():
        print("✅ All required environment variables are set")
    else:
        print("⚠️  Some required environment variables are missing")
        print("📝 Please check .env.example and set up your environment variables")

    print("\n🎉 Environment setup complete!")
    print("\n📋 Next steps:")
    print("1. Set up your API keys in environment variables")
    print("2. Configure Google Cloud credentials if needed")
    print("3. Run: python vector_index.py to set up AI Platform")
    print("4. Start development with the three-tier architecture")

    # Check if .memory folder exists
    if Path(".memory").exists():
        print("✅ Memory system is set up")
    else:
        print("⚠️  Memory system not found - run memory setup if needed")


if __name__ == "__main__":
    main()
