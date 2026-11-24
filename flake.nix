{
  description = "Reproducible buddy-mlir tools and development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    let
      overlay = import ./nix/overlay.nix;
    in
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs { overlays = [ overlay ]; inherit system; };
          buddyPrograms = [
            "buddy-cli"
            "buddy-frontendgen"
            "buddy-llc"
            "buddy-lsp-server"
            "buddy-opt"
            "buddy-server"
            "buddy-translate"
            "rax-inspect"
            "rax-pack"
          ];
          mkBuddyApp = program: {
            type = "app";
            program = "${pkgs.buddy-mlir}/bin/${program}";
            meta.description = "Run ${program} from buddy-mlir";
          };
        in
        {
          # Help other use packages in this flake
          legacyPackages = pkgs;

          packages = {
            inherit (pkgs) buddy-llvm buddy-mlir;
            default = pkgs.buddy-mlir;
          };

          apps = pkgs.lib.genAttrs buddyPrograms mkBuddyApp // {
            default = mkBuddyApp "buddy-opt";
          };

          devShells.default = pkgs.buddy-mlir.devShell;

          formatter = pkgs.nixpkgs-fmt;
        }) //
    {
      # Other system-independent attr
      overlays.default = overlay;
    };
}
