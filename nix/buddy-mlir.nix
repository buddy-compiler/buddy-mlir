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
    version = "unstable-2024-07-18";

    src = with lib.fileset; toSource {
      root = ./..;
      fileset = unions [
        ./../backend
        ./../cmake
        ./../examples
        ./../frontend
        ./../midend
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
      "-DLLVM_MAIN_SRC_DIR=${buddy-llvm.src}/llvm"
      "-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON"
      "-DCMAKE_BUILD_TYPE=Release"
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
