# === Build ===

build:
    cargo build --release

# === Test ===

test:
    cargo test

check:
    cargo build
    cargo clippy -- -D warnings

# === Run ===

serve rom:
    cargo run --release -- --rom {{rom}}

train rom args='':
    cargo run --release -- --rom {{rom}} --train {{args}}

# Default: just serve
default:
    just --list
