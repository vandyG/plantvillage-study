{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
    buildInputs = [
    pkgs.python313
    pkgs.uv
    pkgs.pandoc
    pkgs.texliveFull
  ];
}