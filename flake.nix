{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/release-26.05";
    utils.url = "github:numtide/flake-utils";
    naersk.url = "github:nmattia/naersk";
    naersk.inputs.nixpkgs.follows = "nixpkgs";
    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      utils,
      naersk,
      rust-overlay,
    }:
    utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };

        # The declared MSRV (Cargo.toml `rust-version`). Floor drivers:
        #   * `#[expect(...)]` lint attribute (stable since 1.81)
        #   * edition 2024
        rustMsrv = pkgs.rust-bin.stable."1.85.0".default;
        rustStable = pkgs.rust-bin.stable."1.93.0".default;
        rust = rustMsrv;

        # Override the version used in naersk
        naersk-lib = naersk.lib."${system}".override {
          cargo = rust;
          rustc = rust;
        };

        # naersk with stable rust for reproducible CI checks
        naersk-lib-ci = naersk.lib."${system}".override {
          cargo = rustStable;
          rustc = rustStable;
        };

        bacon = pkgs.bacon;

        # ── CI matrix ───────────────────────────────────────────────────────
        # Each job builds the workspace offline from the committed Cargo.lock
        # (no network in the sandbox). `nix flake check` runs the FULL matrix
        # (see `checks` below); GitHub CI fans out only over the lean subset
        # exposed as `packages.test.*` via `nix build .#test.<name>`.
        cargoVendor = pkgs.rustPlatform.importCargoLock { lockFile = ./Cargo.lock; };

        # Build one check-only derivation: a fixed toolchain, the vendored
        # deps wired up offline by cargoSetupHook, and a single cargo command.
        mkCargoCheck =
          {
            name,
            toolchain ? rustStable,
            phase,
          }:
          pkgs.stdenv.mkDerivation {
            inherit name;
            src = ./.;
            cargoDeps = cargoVendor;
            nativeBuildInputs = [
              toolchain
              pkgs.rustPlatform.cargoSetupHook
            ];
            # Check-only: nothing to install, and the default fixup/strip
            # phases have nothing useful to do.
            dontFixup = true;
            buildPhase = ''
              runHook preBuild
              export HOME=$(mktemp -d)
              ${phase}
              runHook postBuild
            '';
            installPhase = "touch $out";
          };

        # Lean subset — fast, needs no external corpus (the committed synth-*
        # fixtures are the required real-decode subset; see golden_hash.rs).
        ciStable = mkCargoCheck {
          name = "stringpod-test-stable";
          phase = "cargo test --workspace --offline";
        };
        ciClippy = mkCargoCheck {
          name = "stringpod-clippy";
          phase = "cargo clippy --workspace --all-targets --offline -- -D warnings";
        };
        ciFmt = mkCargoCheck {
          name = "stringpod-fmt";
          phase = "cargo fmt --all -- --check";
        };
        # MSRV guarantee is for library/binary consumers, so build (not test):
        # dev-only deps like criterion needn't satisfy 1.80 themselves.
        ciMsrv = mkCargoCheck {
          name = "stringpod-msrv-1.85";
          toolchain = rustMsrv;
          phase = "cargo build --workspace --offline";
        };

      in
      rec {
        # `nix flake check` runs the FULL local matrix:
        checks = {
          stable = ciStable;
          clippy = ciClippy;
          fmt = ciFmt;
          msrv = ciMsrv;
        };

        # Lean CI subset: `nix build .#test.<name>` (used by GitHub CI matrix)
        test = {
          stable = ciStable;
          clippy = ciClippy;
          fmt = ciFmt;
          msrv = ciMsrv;
        };

        # Minimal shell for cargo-deny CI check — avoids pulling in 
        # the rest of the full devShell.  Usage: nix develop .#deny
        devShells.deny = pkgs.mkShell {
          nativeBuildInputs = [
            pkgs.cargo-deny
            pkgs.git
            rust
          ];
        };

        # `nix develop`
        devShells.default = pkgs.mkShell {
          COMMIT_HASH = self.rev or (pkgs.lib.removeSuffix "-dirty" self.dirtyRev or "unknown-not-in-git");
          # we only link with mold in our dev environment for build speed. CI can use the old school rust linker
          shellHook = ''
            #export RUSTFLAGS="-C link-arg=-fuse-ld=mold"
            # Set shell for cmake builds
            export CONFIG_SHELL="${pkgs.bash}/bin/bash"
            export SHELL="${pkgs.bash}/bin/bash"
          '';
          # supply the specific rust version
          nativeBuildInputs = [
            bacon
            pkgs.bash
            pkgs.cargo-audit
            pkgs.cargo-bloat
            pkgs.cargo-crev
            pkgs.cargo-deny
            pkgs.cargo-features-manager
            pkgs.cargo-flamegraph
            pkgs.cargo-insta
            pkgs.cargo-license
            pkgs.cargo-llvm-cov
            pkgs.cargo-llvm-lines
            pkgs.cargo-machete
            pkgs.cargo-mutants
            pkgs.cargo-nextest
            pkgs.cargo-outdated
            pkgs.cargo-readme
            pkgs.cargo-shear
            #pkgs.cargo-udeps
            pkgs.cargo-vet
            pkgs.cargo-expand
            #rust.rust-analyzer
            rust
          ];
        };
      }
    );
}
# {
