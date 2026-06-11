# === Build ===

build:
    cargo build --release

# === Lint ===

clippy:
    cargo clippy -- -D warnings

check: clippy

# === Test ===

test:
    cargo test

# === Run ===

serve rom:
    cargo run --release -- --rom {{rom}}

train rom args='':
    cargo run --release -- --rom {{rom}} --train {{args}}

default:
    just --list
