{
  description = "Generic devshell setup";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }@inputs:
    let
      myOverlay = import ./nix/overlay.nix;
    in
    # Iterate over all nix supported system
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs { overlays = [ myOverlay ]; inherit system; };
        in
        {
          legacyPackages = pkgs;
          devShells.default = pkgs.mkShell
            {
              buildInputs = [ pkgs.buddy-mlir ];
            };
        }) //
    {
      # Other system-independent attr
      inherit inputs;
      overlays.default = myOverlay;
    };
}
