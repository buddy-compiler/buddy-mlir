final: prev:
{
  # Add an alias here can help future migration
  llvmPkgs = final.llvmPackages_16;
  buddy-mlir = final.callPackage ./buddy-mlir.nix { };
}
