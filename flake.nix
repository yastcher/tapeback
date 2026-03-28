{
  description = "tapeback — local meeting recorder with transcription for Obsidian";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        # Development shell — for contributors
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python313
            ffmpeg
            pipewire.pulse
            uv
          ];

          shellHook = ''
            echo "tapeback dev shell — run 'uv sync' to install Python deps"
          '';
        };

        # Convenience wrappers — run via uvx, NOT reproducible Nix packages.
        # Require network access on first run to fetch from PyPI.
        #   nix run github:yastcher/echo-vault            # base
        #   nix run github:yastcher/echo-vault#llm        # + summaries
        #   nix run github:yastcher/echo-vault#diarize    # + speaker diarization
        #   nix run github:yastcher/echo-vault#full       # everything
        packages = let
          mkWrapper = extras: pkgs.writeShellScriptBin "tapeback" ''
            export PATH="${pkgs.lib.makeBinPath [ pkgs.ffmpeg pkgs.pipewire.pulse ]}:$PATH"
            exec ${pkgs.uv}/bin/uvx "tapeback${extras}" "$@"
          '';
        in {
          default  = mkWrapper "";
          llm      = mkWrapper "[llm]";
          diarize  = mkWrapper "[diarize]";
          full     = mkWrapper "[llm,diarize]";
        };
      }
    );
}
