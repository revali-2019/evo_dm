"""
Pytest configuration and shared fixtures for evodm tests.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Shared fixtures can be defined here
# Example:
# @pytest.fixture
# def sample_data():
#     return {"key": "value"}

