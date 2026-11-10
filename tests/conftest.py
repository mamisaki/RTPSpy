"""Shared fixtures and marker handling for the RTPSpy tests.

Markers (see pyproject.toml):
    gpu   needs a CUDA device (skipped when torch.cuda is not available)
    afni  needs the AFNI programs on PATH (skipped when they are missing)
    slow  long tests
"""
import os
import shutil
import logging
from pathlib import Path

import pytest

# Qt widgets must not need a display in the tests
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TEST_DIR = Path(__file__).resolve().parent


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def pytest_runtest_setup(item):
    if item.get_closest_marker("gpu") and not _cuda_available():
        pytest.skip("CUDA is not available")
    if item.get_closest_marker("afni") and shutil.which("3dvolreg") is None:
        pytest.skip("AFNI programs are not on PATH")


@pytest.fixture(scope="session")
def test_dir():
    """Directory of the test data files (tests/)."""
    return TEST_DIR


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def device(request):
    """Torch device to run on: cpu, and cuda (gpu marker)."""
    return request.param


@pytest.fixture(autouse=True)
def _quiet_logging():
    logging.getLogger().setLevel(logging.ERROR)
    yield
