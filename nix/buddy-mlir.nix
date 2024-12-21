{ lib
, stdenv
, fetchFromGitHub
, buddy-llvm
, cmake
, ninja
, llvmPkgs
, python3
}:
stdenv.mkDerivation (finalAttrs: {
  name = "buddy-mlir";

  # Manually specify buddy-mlir source here to avoic rebuild on miscellaneous files
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
  ];

  # This fixup script concatenate the LLVM and Buddy python module into one directory for easier import
  postFixup = ''
    mkdir -p $out/lib/python${python3.pythonVersion}/site-packages
    cp -vr $out/python_packages/buddy $out/lib/python${python3.pythonVersion}/site-packages/
    cp -vr ${buddy-llvm}/python_packages/mlir_core/mlir $out/lib/python${python3.pythonVersion}/site-packages/
  '';

  passthru = {
    llvm = buddy-llvm;

    # Below three fields are nixpkgs black magic that allow site-packages automatically imported with nixpkgs hooks
    # Update here only when you know what will happen
    pythonModule = python3;
    pythonPath = [ ];
    requiredPythonModules = [ ];

    # nix run .#buddy-mlir.pyenv to start a python with PyTorch/LLVM MLIR/Buddy Frontend support
    pyenv = python3.withPackages (ps: [
      finalAttrs.finalPackage
      ps.torch

      # mobilenet
      ps.torchvision

      # tinyllama
      ps.transformers
      ps.accelerate
    ]);
  };

  # No need to do check, and it also takes too much time to finish.
  doCheck = false;
})
