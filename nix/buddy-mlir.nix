{ lib
, stdenv
, buddy-llvm
, cmake
, ninja
, llvmPkgs
, libjpeg
, libpng
, zlib-ng
, ccls
}:
let
  self = stdenv.mkDerivation {
    pname = "buddy-mlir";
    version = "unstable-2026-02-12";

    src = with lib.fileset; toSource {
      root = ./..;
      fileset = unions [
        ./../backend
        ./../cmake
        ./../examples
        ./../frontend
        ./../midend
        ./../scripts
        ./../tests
        ./../tools
        ./../thirdparty
        ./../CMakeLists.txt
        ./../flake.lock
        ./../flake.nix
      ];
    };

    nativeBuildInputs = [
      cmake
      ninja
      llvmPkgs.bintools
    ];

    buildInputs = [
      buddy-llvm
    ];

    cmakeFlags = [
      "-DMLIR_DIR=${buddy-llvm.dev}/lib/cmake/mlir"
      "-DLLVM_DIR=${buddy-llvm.dev}/lib/cmake/llvm"
      "-DLLVM_ENABLE_ASSERTIONS=ON"
      "-DCMAKE_BUILD_TYPE=Release"
      "-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON"
      "-DLLVM_MAIN_SRC_DIR=${buddy-llvm.src}/llvm"
      "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    ];

    passthru = {
      llvm = buddy-llvm;
      devShell = self.overrideAttrs (old: {
        nativeBuildInputs = old.nativeBuildInputs ++ [
          libjpeg
          libpng
          zlib-ng
          ccls
        ];
      });
    };

    # No need to do check, and it also takes too much time to finish.
    doCheck = false;
  };
in
self
