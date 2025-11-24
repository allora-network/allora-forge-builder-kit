"""
Pytest configuration for Allora Forge Builder Kit tests.

This file automatically sets up the ALLORA_API_KEY environment variable
by reading from the .allora_api_key file if it exists.
"""

import os
import pytest
from pathlib import Path


def pytest_configure(config):
    """
    Called before test run starts.
    Auto-loads ALLORA_API_KEY from .allora_api_key file.
    Also registers custom markers.
    """
    # Register custom markers
    config.addinivalue_line(
        "markers", 
        "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )
    
    # Look for .allora_api_key in multiple locations
    possible_paths = [
        Path(__file__).parent / ".allora_api_key",  # tests/.allora_api_key
        Path(__file__).parent.parent / ".allora_api_key",  # project root
        Path(__file__).parent.parent / "notebooks" / ".allora_api_key",  # notebooks/.allora_api_key (correct location!)
        Path.home() / ".allora_api_key",  # home directory
        Path(__file__).parent.parent.parent / "Open Alpha" / "Open Trader" / ".allora_api_key",  # existing location
    ]
    
    # If ALLORA_API_KEY is already set, don't override
    if os.environ.get("ALLORA_API_KEY"):
        print(f"\n✓ ALLORA_API_KEY already set from environment")
        return
    
    # Try to find and read the key file
    for key_path in possible_paths:
        if key_path.exists():
            try:
                api_key = key_path.read_text().strip()
                if api_key:
                    os.environ["ALLORA_API_KEY"] = api_key
                    print(f"\n✓ Loaded ALLORA_API_KEY from {key_path}")
                    return
            except Exception as e:
                print(f"\n⚠ Could not read {key_path}: {e}")
    
    print("\n⚠ ALLORA_API_KEY not found - Allora tests will be skipped")
