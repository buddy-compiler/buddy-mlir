final: prev:
{
  # Add an alias here can help future migration
  # Updated to llvmPackages_22 as llvmPackages_17 has been removed from nixpkgs
  llvmPkgs = final.llvmPackages_22;
  # Use clang instead of gcc to compile, to avoid gcc13 miscompile issue.
  buddy-llvm = final.callPackage ./buddy-llvm.nix { stdenv = final.llvmPkgs.stdenv; };
  buddy-mlir = final.callPackage ./buddy-mlir.nix { stdenv = final.llvmPkgs.stdenv; };
}
