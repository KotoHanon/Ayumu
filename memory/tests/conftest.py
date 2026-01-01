# conftest.py
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--run-real-llm",
        action="store_true",
        default=False,
        help="Run tests that require real LLM API access",
    )
