# KonveyN2AI Linting Workflow - Validation-Only CI

## Overview

KonveyN2AI uses a **"Fix Locally, Validate in CI"** approach for code quality and linting. This ensures:

- ✅ **Zero file modifications in CI** - CI only validates, never changes code
- ✅ **Perfect version synchronization** - Identical tool versions between local and CI
- ✅ **Fast feedback loop** - Issues caught locally before reaching CI
- ✅ **Consistent code quality** - Automated fixes applied locally with identical tools

## Architecture

### Local Development (Auto-Fix Mode)
- **Config**: `.pre-commit-config.yaml`
- **Behavior**: Automatically fixes issues when possible
- **Tools**: Black (format), Ruff (lint + fix), MyPy (type check), Bandit (security)
- **Trigger**: Pre-commit hooks on `git commit` or manual `pre-commit run --all-files`

### CI Pipeline (Validation-Only Mode)
- **Config**: `.pre-commit-config-ci.yaml`
- **Behavior**: Validates code quality, fails if issues found, **never modifies files**
- **Tools**: Black `--check --diff`, Ruff `--check`, MyPy, Bandit `--exit-zero`
- **Trigger**: GitHub Actions on push/PR

## Tool Versions (Synchronized)

All versions are pinned and synchronized across environments:

| Tool | Version | Requirements.txt | Local Config | CI Config |
|------|---------|------------------|--------------|-----------|
| Black | 25.1.0 | `black==25.1.0` | `rev: 25.1.0` | `rev: 25.1.0` |
| Ruff | 0.1.15 | `ruff==0.1.15` | `rev: v0.1.15` | `rev: v0.1.15` |
| MyPy | 1.8.0 | `mypy==1.8.0` | system | system |
| Bandit | 1.7.6 | `bandit==1.7.6` | system | system |
| Pre-commit | 3.6.0 | `pre-commit==3.6.0` | - | - |

## Developer Workflow

### 1. Setup (One-time)
```bash
# Install dependencies with pinned versions
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install
```

### 2. Daily Development
```bash
# Make code changes
vim src/my_file.py

# Commit (pre-commit hooks auto-fix issues)
git add .
git commit -m "feat: my changes"

# If pre-commit fixes issues, review and commit again
git add .
git commit -m "feat: my changes"
```

### 3. Manual Linting (Optional)
```bash
# Run all linting tools manually
pre-commit run --all-files

# Run specific tools
black .
ruff check --fix .
mypy src/
```

### 4. Push to CI
```bash
# Push clean, pre-validated code
git push origin my-branch
```

## CI Validation Process

### 1. Version Synchronization Check
```bash
python scripts/validate_tool_versions.py
```
- Validates that all tool versions match across configs
- Fails CI if versions are out of sync

### 2. Tool Version Verification
```bash
black --version  # Must match requirements.txt
ruff --version   # Must match requirements.txt
# etc.
```

### 3. Code Quality Validation
```bash
pre-commit run --config .pre-commit-config-ci.yaml --all-files
```
- **Black**: `--check --diff` (shows what would change, fails if formatting needed)
- **Ruff**: `--check` (validates linting, fails if issues found)
- **MyPy**: Type checking validation
- **Bandit**: Security scanning (warnings only)

## Configuration Files

### `.pre-commit-config.yaml` (Local Development)
```yaml
# Auto-fix mode for local development
- id: black
  args: []  # Default behavior: auto-fix

- id: ruff  
  args: [--fix, --exit-non-zero-on-fix]  # Auto-fix + exit on changes
```

### `.pre-commit-config-ci.yaml` (CI Validation)
```yaml
# Validation-only mode for CI
- id: black
  args: [--check, --diff]  # Check only, show diff

- id: ruff
  args: [--check]  # Check only, no --fix
```

## Troubleshooting

### CI Fails with "Files would be reformatted"
```bash
# Fix locally
black .
git add .
git commit -m "style: apply black formatting"
git push
```

### CI Fails with "Linting errors found"
```bash
# Fix locally
ruff check --fix .
git add .
git commit -m "style: fix linting issues"
git push
```

### Version Synchronization Errors
```bash
# Check current versions
python scripts/validate_tool_versions.py

# Update versions in all three files:
# - requirements.txt
# - .pre-commit-config.yaml  
# - .pre-commit-config-ci.yaml
```

### Pre-commit Hooks Not Running
```bash
# Reinstall hooks
pre-commit uninstall
pre-commit install

# Test hooks
pre-commit run --all-files
```

## Benefits

### For Developers
- ✅ **Immediate feedback** - Issues fixed during development
- ✅ **No CI surprises** - Local validation matches CI exactly
- ✅ **Automated fixes** - Tools fix most issues automatically
- ✅ **Consistent formatting** - Team uses identical tool versions

### For CI/CD
- ✅ **Fast validation** - No time spent on formatting/fixing
- ✅ **Immutable code** - CI never modifies files
- ✅ **Clear failures** - Obvious what needs to be fixed locally
- ✅ **Version consistency** - Guaranteed tool version synchronization

### For Code Quality
- ✅ **Consistent standards** - Identical tools across all environments
- ✅ **Automated enforcement** - Can't commit without passing checks
- ✅ **Zero drift** - Pinned versions prevent tool version conflicts
- ✅ **Security scanning** - Bandit runs on all code changes

## Maintenance

### Updating Tool Versions
1. Update version in `requirements.txt`
2. Update corresponding `rev:` in `.pre-commit-config.yaml`
3. Update corresponding `rev:` in `.pre-commit-config-ci.yaml`
4. Run `python scripts/validate_tool_versions.py` to verify
5. Test locally: `pre-commit run --all-files`
6. Commit and push to validate CI

### Adding New Tools
1. Add to `requirements.txt` with pinned version
2. Add to both pre-commit configs with appropriate args
3. Update version validation script
4. Update this documentation
