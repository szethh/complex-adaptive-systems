{
  description = "Complex Adaptive Systems";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
    nix-latex = {
      url = "github:szethh/nix-latex";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      nix-latex,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        latex-inputs-base = (nix-latex.devShellFor system).buildInputs;
        latex-inputs = latex-inputs-base ++ [
          pkgs.biber
          (pkgs.texlive.combine {
            inherit (pkgs.texlive)
              biblatex
              csquotes
              enumitem
              environ
              latexindent
              multirow
              scheme-medium
              todonotes
              xypic
              ;
          })
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [ ] ++ latex-inputs;
        };
      }
    );
}
