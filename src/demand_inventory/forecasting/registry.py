"""Optional forecasting-framework availability checks without importing heavy stacks."""

from __future__ import annotations

import importlib.util

MODEL_IMPORTS = {
    "lightgbm": ("lightgbm", "pip install -r requirements-prophet-lightgbm.txt"),
    "prophet": ("prophet", "pip install -r requirements-prophet-lightgbm.txt"),
    "darts": ("darts", "pip install -r requirements-darts.txt"),
    "pytorch_forecasting": (
        "pytorch_forecasting",
        "Install a CPU/CUDA-compatible PyTorch build, then pip install -r requirements-pytorch-forecasting.txt",
    ),
}


def get_model_availability() -> dict[str, dict[str, str | bool]]:
    availability: dict[str, dict[str, str | bool]] = {}
    for model_name, (module_name, install_command) in MODEL_IMPORTS.items():
        available = importlib.util.find_spec(module_name) is not None
        availability[model_name] = {
            "available": available,
            "status": "installed" if available else "not_installed",
            "import_name": module_name,
            "install_instruction": install_command,
        }
    return availability
