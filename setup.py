from __future__ import annotations

import os
import shutil
from pathlib import Path

from setuptools import find_namespace_packages, find_packages, setup
from setuptools.dist import Distribution
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
from setuptools.command.build import build as _build
from setuptools.command.build_py import build_py as _build_py

ROOT = Path(__file__).parent.resolve()


def _resolve_build_dir() -> Path:
    """Resolve the CMake build directory that already contains compiled outputs."""
    build_dir = Path(os.environ.get("BUDDY_BUILD_DIR", "build"))
    if not build_dir.is_absolute():
        build_dir = ROOT / build_dir
    return build_dir.resolve()


CMAKE_BUILD = _resolve_build_dir()
PYTHON_PACKAGES_DIR = CMAKE_BUILD / "python_packages"
BIN_DIR = CMAKE_BUILD / "bin"
LIB_DIR = CMAKE_BUILD / "lib"

if not PYTHON_PACKAGES_DIR.exists():
    raise SystemExit(
        "buddy-mlir expects a populated CMake build tree. "
        "Please configure and build with BUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON "
        "before building the wheel (default build dir: ./build)."
    )

REL_PYTHON_PACKAGES_DIR = os.path.relpath(PYTHON_PACKAGES_DIR, ROOT)
if REL_PYTHON_PACKAGES_DIR.startswith(".."):
    raise SystemExit(
        f"BUDDY_BUILD_DIR must reside inside the project root ({ROOT}) so packaging "
        f"can use relative paths. Current: {PYTHON_PACKAGES_DIR}"
    )

# Stage python packages into the build tree so setuptools never sees absolute paths.
STAGING_ROOT = CMAKE_BUILD / "py-stage"
if STAGING_ROOT.exists():
    shutil.rmtree(STAGING_ROOT)
STAGING_ROOT.mkdir(parents=True, exist_ok=True)
STAGING_SRC = STAGING_ROOT / "python_packages"
# Copy python outputs but drop any pre-existing egg-info that may carry absolute paths.
shutil.copytree(
    PYTHON_PACKAGES_DIR,
    STAGING_SRC,
    ignore=shutil.ignore_patterns("*.egg-info"),
)

SRC_DIR = os.path.relpath(STAGING_SRC, ROOT)

buddy_pkgs = find_packages(where=SRC_DIR, include=["buddy*"])
mlir_pkgs = find_namespace_packages(where=SRC_DIR, include=["buddy_mlir*"])
wrapper_pkgs = find_packages(where="tools", include=["buddy_tools*"])
packages = sorted(set(buddy_pkgs + mlir_pkgs + wrapper_pkgs))

package_dir = {
    "": SRC_DIR,
    "buddy_tools": "tools/buddy_tools",
}


class build_py(_build_py):
    """Copy prebuilt artifacts (Python, bin, lib) into the wheel."""

    def run(self):
        self._extra_outputs = []
        super().run()

        tools_root = Path(self.build_lib) / "buddy_tools"
        self._copy_tree(BIN_DIR, tools_root / "bin", allow_missing=True)
        self._copy_tree(LIB_DIR, tools_root / "lib", allow_missing=True)

    def get_outputs(self, include_bytecode: int = 1):
        outputs = super().get_outputs(include_bytecode)
        return outputs + getattr(self, "_extra_outputs", [])

    # Helpers
    def _copy_tree(self, src: Path, dst: Path, allow_missing: bool = False):
        src = Path(src)
        if not src.exists():
            if allow_missing:
                self.warn(f"Skip missing path: {src}")
                return
            raise FileNotFoundError(f"Expected path not found: {src}")

        # Avoid recursive self-copy if src overlaps dst.
        try:
            if dst.resolve().is_relative_to(src.resolve()):
                raise SystemExit(f"Refusing to copy {src} into itself ({dst})")
        except AttributeError:
            # Python <3.9 compatibility: manual check
            src_resolved = src.resolve()
            dst_resolved = dst.resolve()
            if str(dst_resolved).startswith(str(src_resolved)):
                raise SystemExit(f"Refusing to copy {src} into itself ({dst})")

        for path in src.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            self._extra_outputs.append(str(target))


ENTRY_POINTS = {
    "console_scripts": [
        "buddy-opt=buddy_tools.cli:buddy_opt",
        "buddy-translate=buddy_tools.cli:buddy_translate",
        "buddy-llc=buddy_tools.cli:buddy_llc",
        "buddy-lsp-server=buddy_tools.cli:buddy_lsp_server",
        "buddy-frontendgen=buddy_tools.cli:buddy_frontendgen",
        "buddy-audio-container-test=buddy_tools.cli:buddy_audio_container_test",
        "buddy-text-container-test=buddy_tools.cli:buddy_text_container_test",
        "buddy-container-test=buddy_tools.cli:buddy_container_test",
    ]
}


def _resolve_install_requires() -> list[str]:
    """Pin torch version when TORCH_VERSION is provided."""
    torch_ver = os.environ.get("TORCH_VERSION", "").strip()
    if torch_ver:
        return [f"torch=={torch_ver}"]
    return []


class build(_build):
    def initialize_options(self):
        super().initialize_options()
        # Use a staging directory separate from the CMake build tree.
        self.build_base = str(CMAKE_BUILD / "py-build")


class bdist_wheel(_bdist_wheel):
    """Mark the wheel as non-pure and optionally set a build tag."""

    def finalize_options(self):
        super().finalize_options()
        # Force platform-specific wheel since we bundle native libs.
        self.root_is_pure = False

        torch_ver = os.environ.get("TORCH_VERSION", "unknown")
        mlir_major = "unknown"

        llvm_version_path = (
            ROOT / "llvm" / "cmake" / "Modules" / "LLVMVersion.cmake"
        )
        if llvm_version_path.is_file():
            for line in llvm_version_path.read_text(
                encoding="utf-8"
            ).splitlines():
                line = line.strip()
                if line.startswith("set(LLVM_VERSION_MAJOR"):
                    parts = line.split()
                    if len(parts) >= 2:
                        mlir_major = parts[1].rstrip(")")
                    break

        self.build_number = (
            f"1torch{torch_ver.replace('.', '')}_2mlir{mlir_major}"
        )


class BinaryDistribution(Distribution):
    """Force a non-pure wheel since we bundle native binaries."""

    def has_ext_modules(self):
        return True


setup(
    packages=packages,
    package_dir=package_dir,
    include_package_data=True,
    cmdclass={"build_py": build_py, "build": build, "bdist_wheel": bdist_wheel},
    distclass=BinaryDistribution,
    install_requires=_resolve_install_requires(),
    entry_points=ENTRY_POINTS,
    zip_safe=False,
)
