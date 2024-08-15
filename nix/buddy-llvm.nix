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
    ps.pyyaml
    ps.ml-dtypes
  ]);
in
stdenv.mkDerivation rec {
  name = "llvm-for-buddy-mlir";
  version = "6c59f0e1b0fb56c909ad7c9aad4bde37dc006ae0";
  src = fetchFromGitHub {
    owner = "llvm";
    repo = "llvm-project";
    rev = version;
    hash = "sha256-bMJJ2q1hSh7m0ewclHOmIe7lOHv110rz/P7D3pw8Uiw=";
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
    "-DLLVM_ENABLE_PROJECTS=mlir"
    "-DLLVM_TARGETS_TO_BUILD=host;RISCV"
    "-DLLVM_ENABLE_ASSERTIONS=ON"
    "-DCMAKE_BUILD_TYPE=Release"
    # required for MLIR python binding
    "-DMLIR_ENABLE_BINDINGS_PYTHON=ON"
    # required for not, FileCheck...
    "-DLLVM_INSTALL_UTILS=ON"
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

    # patch configuration files so each path points to the new $lib or $dev paths
    substituteInPlace "$dev/lib/cmake/llvm/LLVMConfig.cmake" \
      --replace 'set(LLVM_BINARY_DIR "''${LLVM_INSTALL_PREFIX}")' 'set(LLVM_BINARY_DIR "'"$lib"'")'
    substituteInPlace \
      "$dev/lib/cmake/llvm/LLVMExports-release.cmake" \
      "$dev/lib/cmake/mlir/MLIRTargets-release.cmake" \
      --replace "\''${_IMPORT_PREFIX}/lib/lib" "$lib/lib/lib" \
      --replace "\''${_IMPORT_PREFIX}/lib/objects-Release" "$lib/lib/objects-Release" \
      --replace "$out/bin/llvm-config" "$dev/bin/llvm-config" # patch path for llvm-config
  '';
}
