# Test Directory

This directory contains tests and validation tools for the race_monitor package.

## Test Files

### `test_race_monitor.py`
Basic unit tests that validate:
- Package structure and imports
- Dependencies availability  
- Configuration files existence
- Metadata consistency

**Usage:**
```bash
# Run all tests
pytest test/test_race_monitor.py -v

# Run with coverage
pytest test/test_race_monitor.py --cov=race_monitor --cov-report=html
```

### `validate_environment.py`
Environment validation script for development setup. Checks:
- Python version compatibility (3.10+)
- ROS2 installation and sourcing
- All Python dependencies
- EVO submodule initialization
- ROS2 package availability
- Build tools (colcon)
- Workspace structure

**Usage:**
```bash
# Validate current environment
python3 test/validate_environment.py

# Make it executable and run
chmod +x test/validate_environment.py
./test/validate_environment.py
```

## Running Tests

### Local Development
```bash
# Install test dependencies
pip install pytest pytest-cov mypy flake8

# Run all tests
pytest test/ -v

# Run with coverage
pytest test/ --cov=race_monitor --cov-report=html

# Lint code
flake8 race_monitor/ --max-line-length=127

# Type checking
mypy --ignore-missing-imports race_monitor/
```

### CI/CD
Tests are automatically run in GitHub Actions CI pipeline:
- Python tests on multiple versions (3.10, 3.11)
- ROS2 build and integration tests
- Package build validation

## Test Categories

- **Unit tests**: Basic functionality and imports
- **Integration tests**: ROS2 launch and node tests (in CI)
- **Environment validation**: Development setup verification

## Adding New Tests

1. Create test files following the `test_*.py` naming convention
2. Use appropriate test markers:
   - `@pytest.mark.unit` for unit tests
   - `@pytest.mark.integration` for integration tests
   - `@pytest.mark.slow` for slow-running tests

3. Follow the existing test structure and patterns

Example:
```python
import pytest

class TestNewFeature:
    @pytest.mark.unit
    def test_basic_functionality(self):
        # Test implementation
        pass
        
    @pytest.mark.integration
    def test_ros2_integration(self):
        # Integration test
        pass
```