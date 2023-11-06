{ cmake, ninja, python3, llvmPackages_16, fetchFromGitHub, libjpeg, libpng, zlib-ng }:
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

  nativeBuildInputs = [ cmake ninja python3 llvmPackages_16.bintools libjpeg libpng zlib-ng ];

  cmakeDir = "../llvm";
  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DLLVM_ENABLE_PROJECTS=mlir"
    "-DLLVM_TARGETS_TO_BUILD=host;RISCV"
    "-DLLVM_ENABLE_ASSERTIONS=ON"
    "-DLLVM_USE_LINKER=lld"

    "-DLLVM_EXTERNAL_PROJECTS=buddy-mlir"
    "-DLLVM_EXTERNAL_BUDDY_MLIR_SOURCE_DIR=../../buddy-mlir"
  ];

  checkTarget = "check-mlir check-buddy";
}
