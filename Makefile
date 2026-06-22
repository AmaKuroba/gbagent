.PHONY: test test-all lint typecheck

# Run all non-keras tests (works without TensorFlow)
test:
	KERAS_BACKEND=numpy python -m pytest tests/ -v -k "not keras" --tb=short

# Run all tests (requires TensorFlow)
test-all:
	python -m pytest tests/ -v --tb=short

# Lint with ruff
lint:
	ruff check gbagent/ train.py sweep.py tests/

# Type check with mypy (requires stubs)
typecheck:
	mypy gbagent/ train.py sweep.py
