"""Make ``src`` importable when pytest is run from the repository root."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA GPU and pretrained weights")
