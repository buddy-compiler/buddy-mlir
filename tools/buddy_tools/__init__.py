"""Resource helper for bundled Buddy MLIR binaries and libraries."""

from importlib import resources
from pathlib import Path


def bin_dir() -> Path:
    """Absolute path to packaged Buddy executables."""
    return Path(resources.files(__name__) / "bin")


def lib_dir() -> Path:
    """Absolute path to packaged Buddy libraries."""
    return Path(resources.files(__name__) / "lib")


__all__ = ["bin_dir", "lib_dir"]
