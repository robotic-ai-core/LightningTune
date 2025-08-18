"""
Lightning BOHB Test Suite.

Comprehensive tests for hyperparameter optimization with pause/resume capabilities.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))