{
  description = "Generic devshell setup";

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

          devShells.default = pkgs.buddy-mlir.devShell;

          formatter = pkgs.nixpkgs-fmt;
        }) //
    {
      # Other system-independent attr
      inherit inputs;
      overlays.default = overlay;
    };
}
