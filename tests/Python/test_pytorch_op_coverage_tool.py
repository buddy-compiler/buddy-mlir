# RUN: %PYTHON %s
"""Run dependency-free coverage-tool regressions through the Python lit suite."""

import sys
import unittest
from pathlib import Path

if __name__ == "__main__":
    scripts = (
        Path(__file__).resolve().parents[2] / "scripts/pytorch_op_coverage"
    )
    suite = unittest.defaultTestLoader.discover(
        str(scripts), pattern="test_coverage.py"
    )
    result = unittest.TextTestRunner().run(suite)
    sys.exit(not result.wasSuccessful())
