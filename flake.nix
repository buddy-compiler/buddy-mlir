{
  description = "Buddy MLIR nix flake setup";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }@inputs:
    let
      overlay = import ./nix/overlay.nix;
    in
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs { overlays = [ overlay ]; inherit system; };
        in
        {
          # Help other use packages in this flake
          legacyPackages = pkgs;

          # A shell with buddy-mlir tools and Python environment
          # run: nix develop .
          devShells.default = pkgs.mkShell {
            nativeBuildInputs = with pkgs; [
              buddy-mlir
              buddy-mlir.llvm
              buddy-mlir.pyenv
            ];
          };

          # A shell for writing buddy-mlir code
          # run: nix develop .#buddy-mlir
          devShells.buddy-mlir = pkgs.mkShell {
            nativeBuildInputs = with pkgs;
              # Add following extra packages to buddy-mlir developement environment
              pkgs.buddy-mlir.nativeBuildInputs ++ [
                libjpeg
                libpng
                zlib-ng
                ccls
              ];
          };

          # run: nix build .
          packages.default = pkgs.buddy-mlir;

          formatter = pkgs.nixpkgs-fmt;
        }) //
    {
      # Other system-independent attr
      inherit inputs;
      overlays.default = overlay;
    };
}
