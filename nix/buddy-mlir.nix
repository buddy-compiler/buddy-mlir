{ lib
, fetchFromGitHub
, cmake
, ninja
, llvmPackages_16

  # Required by graphic frontend
, libjpeg
, libpng
, zlib-ng

  # Compile Options, can be disabled using override. Eg:
  #
  # ```nix
  # # overlay.nix
  # final: prev:
  # {
  #   yourBuddyMlir = prev.buddy-mlir.override {
  #     enablePython3Bindings = false;
  #   };
  # }
  #
, enablePython3Bindings ? true
  #
  # We need to pin python to 3.10.* for two reasons:
  #
  #   1. PyTorch in current Nixpkgs doens't support 3.11 yet.
  #   2. The current mainline of PyTorch do have 3.11 support, but they bypass a large amount of tests for Python 3.11,
  #      which is not reliable to be used.
  #
, python310
, python310Packages
}:
let
  # Using git submodule to obtain the llvm source is really slow.
  # So here I use tarball to save time from git index.
  llvmSrc = fetchFromGitHub {
    owner = "llvm";
    repo = "llvm-project";
    rev = "6c59f0e1b0fb56c909ad7c9aad4bde37dc006ae0";
    sha256 = "sha256-bMJJ2q1hSh7m0ewclHOmIe7lOHv110rz/P7D3pw8Uiw";
  };
in
# Use clang instead of gcc to build
llvmPackages_16.stdenv.mkDerivation {
  pname = "buddy-mlir";
  version = "unstable-2023-11-07+rev=38bfd56";

  srcs = [
    llvmSrc
    ../.
  ];
  sourceRoot = "llvm-project";
  unpackPhase = ''
    sourceArray=($srcs)
    cp -r ''${sourceArray[0]} llvm-project
    cp -r ''${sourceArray[1]} buddy-mlir

    # Directories copied from nix store are read only
    chmod -R u+w llvm-project buddy-mlir
  '';

  # Tablegen in latest commit have bug. See llvm-projects issue #68166
  prePatch = "pushd $NIX_BUILD_TOP/llvm-project";
  patches = [ ./tblgen.patch ];
  postPatch = "popd";

  nativeBuildInputs = [
    cmake
    ninja
    python310
    llvmPackages_16.bintools # use lld instead of ld to link
  ];

  buildInputs = [
    libjpeg
    libpng
    zlib-ng
  ] ++ lib.optionals enablePython3Bindings [
    # Required by Python binding
    python310Packages.pybind11
  ];

  # Required by E2E
  propagatedBuildInputs = lib.optionals enablePython3Bindings (with python310Packages; [
    # The CPython extension will need same major version of Python to load the share library,
    # so it is better to add python 310 here for downstream user, to avoid potential problems.
    python310
    numpy
    transformers
    pyyaml
    torch
  ]);

  cmakeDir = "../llvm";
  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DLLVM_ENABLE_PROJECTS=mlir"
    "-DLLVM_TARGETS_TO_BUILD=host;RISCV"
    "-DLLVM_ENABLE_ASSERTIONS=ON"
    "-DLLVM_USE_LINKER=lld"

    "-DLLVM_EXTERNAL_PROJECTS=buddy-mlir"
    "-DLLVM_EXTERNAL_BUDDY_MLIR_SOURCE_DIR=../../buddy-mlir"
  ] ++ lib.optionals enablePython3Bindings [
    "-DMLIR_ENABLE_BINDINGS_PYTHON=ON"
    "-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON"
  ];

  checkTarget = "check-mlir check-buddy";

  # Python exeutable in Nix have wrapper that can automatically append libraries into PYTHONPATH
  # But our Python bindings are not packaged in pip spec. Here is the manual fix for automatically
  # adding Python module `buddy` and `mlir` into PYTHONPATH.
  postFixup = lib.optionalString enablePython3Bindings ''
    local sitePkgPath="$out/python_packages/${python310.sitePackages}"
    mkdir -p $sitePkgPath

    mv $out/python_packages/buddy $sitePkgPath
    mv $out/python_packages/mlir_core/mlir $sitePkgPath
    rm -r $out/python_packages/mlir_core

    mkdir -p "$out/nix-support"
    echo "$out/python_packages" >> "$out/nix-support/propagated-build-inputs"
  '';
}
