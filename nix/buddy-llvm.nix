{ stdenv
, cmake
, ninja
, python3
, fetchFromGitHub
}:

let
  pythonEnv = python3.withPackages (ps: [
    ps.numpy
    ps.pybind11
    ps.nanobind
    ps.pyyaml
    ps.ml-dtypes
  ]);
in
stdenv.mkDerivation rec {
  pname = "llvm-for-buddy-mlir";
  # Keep the fork, revision, and source hash in sync with the llvm submodule.
  version = "163d6a753b583aa6b216cd54da198e2769d249c6";
  src = fetchFromGitHub {
    owner = "RuyiAI-Stack";
    repo = "llvm-project";
    rev = version;
    hash = "sha256-PbRI/8J36jvMf+GEW8JPDWRuZs/U3yoUzsuDO7pYO3w=";
  };

  requiredSystemFeatures = [ "big-parallel" ];

  propagatedBuildInputs = [
    pythonEnv
  ];

  nativeBuildInputs = [
    cmake
    ninja
  ];

  cmakeDir = "../llvm";
  cmakeFlags = [
    "-DLLVM_ENABLE_PROJECTS=mlir;clang"
    "-DLLVM_TARGETS_TO_BUILD=host;RISCV"
    "-DLLVM_ENABLE_ASSERTIONS=ON"
    "-DLLVM_INCLUDE_BENCHMARKS=OFF"
    "-DCMAKE_BUILD_TYPE=Release"
    # required for MLIR python binding
    "-DMLIR_ENABLE_BINDINGS_PYTHON=ON"
    "-DPython3_EXECUTABLE=${pythonEnv}/bin/python3"
    "-DPython_EXECUTABLE=${pythonEnv}/bin/python3"
    # required for not, FileCheck...
    "-DLLVM_INSTALL_UTILS=ON"
    # Avoid C2y extension warnings from LLVM's third-party code with Clang 22.
    "-DCMAKE_CXX_FLAGS=-Wno-c2y-extensions"
  ];

  outputs = [ "out" "lib" "dev" ];

  postInstall = ''
    # buddy-mlir have custom RVV backend that required LLVM backend,
    # and those LLVM backend headers require this config.h header file.
    # However for LLVM, this config.h is meant to be used on build phase only,
    # so it will not be installed for cmake install.
    # We have to do some hack
    cp -v "include/llvm/Config/config.h" "$dev/include/llvm/Config/config.h"

    # move llvm-config to $dev to resolve a circular dependency
    moveToOutput "bin/llvm-config*" "$dev"

    # move all lib files to $lib except lib/cmake
    moveToOutput "lib" "$lib"
    moveToOutput "lib/cmake" "$dev"

    # move python source files to $dev so buddy-mlir can find them via cmake
    moveToOutput "src/python" "$dev"

    # patch configuration files so each path points to the new $lib or $dev paths
    substituteInPlace "$dev/lib/cmake/llvm/LLVMConfig.cmake" \
      --replace 'set(LLVM_BINARY_DIR "''${LLVM_INSTALL_PREFIX}")' 'set(LLVM_BINARY_DIR "'"$lib"'")' \
      --replace 'set(LLVM_LIBRARY_DIRS "''${LLVM_INSTALL_PREFIX}/lib''${LLVM_LIBDIR_SUFFIX}")' 'set(LLVM_LIBRARY_DIRS "'"$lib"'/lib")' \
      --replace 'set(LLVM_LIBRARY_DIR "''${LLVM_INSTALL_PREFIX}/lib''${LLVM_LIBDIR_SUFFIX}")' 'set(LLVM_LIBRARY_DIR "'"$lib"'/lib")'
    substituteInPlace \
      "$dev/lib/cmake/llvm/LLVMExports-release.cmake" \
      "$dev/lib/cmake/mlir/MLIRTargets-release.cmake" \
      --replace "\''${_IMPORT_PREFIX}/lib/lib" "$lib/lib/lib" \
      --replace "\''${_IMPORT_PREFIX}/lib/objects-Release" "$lib/lib/objects-Release" \
      --replace "$out/bin/llvm-config" "$dev/bin/llvm-config" # patch path for llvm-config
  '';
}
