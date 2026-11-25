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
, flatbuffers
, numactl
, python3
, fetchurl
, runCommand
, writeText
, git
, gcc
, pkg-config
, perl
, pre-commit
}:
let
  pythonEnv = python3.withPackages (ps: [
    ps.numpy
    ps.pybind11
    ps.nanobind
    ps.pyyaml
    ps.ml-dtypes
    ps.lit
  ]);
  cli11Header = fetchurl {
    url = "https://github.com/CLIUtils/CLI11/releases/download/v2.6.2/CLI11.hpp";
    hash = "sha256-InoW/l+fitqAw8QJSSR1U29ZfnvYOmwm6sw8jBSakpU=";
  };
  cli11Source = runCommand "cli11-2.6.2" { } ''
    mkdir -p "$out"
    cp ${cli11Header} "$out/CLI11.hpp"
  '';
  # LLVM runs LLVM_EXTERNAL_LIT through Python3_EXECUTABLE, so point it at a
  # Python launcher rather than nixpkgs' compiled entry-point wrapper.
  litLauncher = writeText "buddy-llvm-lit.py" ''
    from lit.main import main

    raise SystemExit(main())
  '';
  self = stdenv.mkDerivation {
    pname = "buddy-mlir";
    version = "unstable-2026-11-24";

    src = with lib.fileset; toSource {
      root = ./..;
      fileset = unions [
        ./../cmake
        ./../examples
        ./../frontend
        ./../midend
        ./../models
        ./../runtime
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
      flatbuffers
      llvmPkgs.bintools
    ];

    buildInputs = [
      buddy-llvm
      flatbuffers
      numactl
      pythonEnv
    ];

    cmakeFlags = [
      "-DMLIR_DIR=${buddy-llvm.dev}/lib/cmake/mlir"
      "-DLLVM_DIR=${buddy-llvm.dev}/lib/cmake/llvm"
      "-DLLVM_MAIN_SRC_DIR=${buddy-llvm.src}/llvm"
      "-DLLVM_ENABLE_ASSERTIONS=ON"
      "-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON"
      "-DPython3_EXECUTABLE=${pythonEnv}/bin/python3"
      "-DPython_EXECUTABLE=${pythonEnv}/bin/python3"
      "-DLLVM_EXTERNAL_LIT=${litLauncher}"
      "-DFETCHCONTENT_SOURCE_DIR_CLI11_HPP=${cli11Source}"
      "-DCMAKE_BUILD_TYPE=Release"
      "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    ];

    passthru = {
      llvm = buddy-llvm;
      devShell = self.overrideAttrs (old: {
        nativeBuildInputs = old.nativeBuildInputs ++ [
          ccls
          git
          gcc
          pkg-config
          perl
          pre-commit
        ];
        buildInputs = old.buildInputs ++ [
          libjpeg
          libpng
          zlib-ng
          llvmPkgs.openmp
        ];
        shellHook = ''
          export PYTHONPATH="${buddy-llvm}/python_packages/mlir_core''${PYTHONPATH:+:''${PYTHONPATH}}"
        '';
      });
    };

    # No need to do check, and it also takes too much time to finish.
    doCheck = false;
  };
in
self
