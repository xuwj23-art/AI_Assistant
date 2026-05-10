# Tests

Unit and integration tests for the project. `conftest.py` automatically adds `src/` and the project root to `sys.path`, so both `pytest` and direct `python tests/xxx.py` can import `core.*`.

## Quick Verification

```bash
# 1) No backend, no data — environment and dependency health check
python tests/test_environment.py

# 2) No backend — Stage 3 visualisation unit tests
python tests/test_stage3_viz.py --unit

# 3) Requires running backend — API smoke tests
python tests/_smoke_api.py
```

## Using pytest

```bash
pytest tests/ -v
pytest tests/test_stage3_viz.py::test_trend_analysis_basic -v
```

## Test Files

| File | Backend needed | Network needed | Description |
|------|:-:|:-:|-------------|
| `test_environment.py` | ✗ | ✗ | Python / Torch / dependency version check |
| `test_torch_simple.py` | ✗ | ✗ | Basic Torch tensor operations |
| `test_bertopic_install.py` | ✗ | ✗ | BERTopic installation check |
| `test_embedding.py` | ✗ | ✗ | Sentence Transformer embedding test |
| `test_data.py` | ✗ | ✗ | Data loading utilities |
| `test_generate_vectors.py` | ✗ | ✗ | Embedding generation pipeline |
| `test_summarizer.py` | ✗ | ✗ | Local T5 summariser |
| `test_stage3_viz.py` | ✗ | ✗ | Topic/trend visualisation modules |
| `test_stage2_integration.py` | ✗ | ✓ | Multi-source aggregator integration |
| `test_rag_core.py` | ✗ | ✗ | RAG service core logic |
| `test_rag_api.py` | ✓ | ✗ | RAG API endpoints |
| `test_smoke_fixes.py` | ✓ | ✗ | Regression smoke tests |
| `_smoke_api.py` | ✓ | ✗ | Full API smoke test suite |
