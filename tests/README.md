# Test Suite

Comprehensive pytest test suite covering all functions in the transformer training codebase.

## Running Tests

### Run All Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=. --cov-report=html
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Property-based tests only
pytest -m property
```

### Run Specific Test File

```bash
pytest tests/unit/test_attention.py
```

### Run with Verbose Output

```bash
pytest -v
```

## Test Coverage

The test suite aims for 100% function coverage and includes:

- **Unit Tests**: Test each function/class in isolation
- **Integration Tests**: Test components working together
- **Property-Based Tests**: Verify mathematical properties and invariants
- **Edge Cases**: Boundary conditions, error handling, invalid inputs

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.property`: Property-based tests

## Fixtures

Shared fixtures are defined in `conftest.py`:

- Config fixtures: `small_config`, `gpt_config`, `llama_config`, `olmo_config`, `moe_config`
- Model fixtures: `model_with_einops`, `model_without_einops`, `llama_model_with_einops`, `moe_model`
- Tokenizer fixtures: `character_tokenizer`, `bpe_tokenizer`, `simple_bpe_tokenizer`
- Data fixtures: `sample_text`, `sample_tokens`, `sample_batch_tokens`, `sample_csv_file`
- Training fixtures: `training_args`, `finetuning_args`
- Device fixture: `device` (CPU for deterministic tests)

## Notes

- Tests use CPU device for deterministic results
- Random seeds are set for reproducibility
- Tests use small model configs for fast execution
- All tests are isolated (no shared state)

