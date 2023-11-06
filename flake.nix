{
  description = "Generic devshell setup";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }@inputs:
    let
      overlay = import ./nix/overlay.nix;
      pkgsForSys = system: import nixpkgs { overlays = [ overlay ]; inherit system; };
    in
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = pkgsForSys system;
          mkLLVMShell = pkgs.mkShell.override { stdenv = pkgs.llvmPkgs.stdenv; };
        in
        {
          # Help other use packages in this flake
          legacyPackages = pkgs;

          devShells.default = mkLLVMShell {
            buildInputs = with pkgs; [
              # buddy-mlir build tools
              cmake
              ninja
              python3
              llvmPkgs.bintools # For ld.lld

              # buddy-mlir libraries
              libjpeg
              libpng
              zlib-ng
            ];

            postHook = ''
              export PATH="${pkgs.clang-tools}/bin:$PATH"
            '';
          };

          formatter = pkgs.nixpkgs-fmt;
        }) //
    {
      # Other system-independent attr
      inherit inputs;
      overlays.default = overlay;
    };
}
