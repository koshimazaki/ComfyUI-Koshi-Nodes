"""Shared fixtures for Koshi <-> Deforum2026 cross-verification tests."""

import sys
import os
import pytest

# Koshi nodes path
KOSHI_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if KOSHI_ROOT not in sys.path:
    sys.path.insert(0, KOSHI_ROOT)

# Deforum2026 scheduling path
REPO_ROOT = os.path.dirname(KOSHI_ROOT)
DEFORUM_SCHEDULING = os.path.join(REPO_ROOT, "Deforum2026", "scheduling")
DEFORUM_ROOT = os.path.join(REPO_ROOT, "Deforum2026")


@pytest.fixture
def deforum_available():
    """Check if Deforum2026 repo is available."""
    return os.path.isdir(DEFORUM_SCHEDULING)


@pytest.fixture(autouse=True)
def add_deforum_path():
    """Add Deforum2026 paths for cross-verification."""
    paths_to_add = [DEFORUM_ROOT, DEFORUM_SCHEDULING]
    added = []
    for p in paths_to_add:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
            added.append(p)
    yield
    for p in added:
        if p in sys.path:
            sys.path.remove(p)
