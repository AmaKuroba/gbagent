.PHONY: test test-all lint typecheck

# Run tests with JAX backend (default)
test:
	KERAS_BACKEND=jax python -m pytest tests/ -v --tb=short

# Run tests with all available backends
test-all:
	python -m pytest tests/ -v --tb=short

# Lint with ruff
lint:
	ruff check gbagent/ train.py sweep.py tests/

# Type check with mypy (requires stubs)
typecheck:
	mypy gbagent/ train.py sweep.py
