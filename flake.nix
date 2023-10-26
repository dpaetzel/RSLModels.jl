{
  inputs = {
    nixpkgs.url = "github:dpaetzel/nixpkgs/dpaetzel/nixos-config";
    systems.url = "github:nix-systems/default";
    devenv.url = "github:cachix/devenv";
  };

  nixConfig = {
    extra-trusted-public-keys =
      "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw=";
    extra-substituters = "https://devenv.cachix.org";
  };

  outputs = { self, nixpkgs, devenv, systems, ... }@inputs:
    let forEachSystem = nixpkgs.lib.genAttrs (import systems);
    in {
      devShells = forEachSystem (system:
        let pkgs = nixpkgs.legacyPackages.${system};
        in rec {
          default = devenv.lib.mkShell {
            inherit inputs pkgs;
            modules = [{
              # https://devenv.sh/reference/options/
              languages.python.enable = true;
              languages.python.package =
                # setuptools is required by mlflow, it seems.
                pkgs.python310.withPackages (ps: [ ps.matplotlib ps.mlflow ps.setuptools ]);

              languages.julia.enable = true;
            }];
          };
          devShell.${system} = default;
        });
    };
}
