"""
Load saved Tetris models with PyTorch safe mode (weights_only=True).

Safe mode builds only the classes in SAFE_CLASSES, so a model file cannot run other code.
"""
import zipfile

import torch
import torch.nn as nn

from src.deep_q_network import DeepQNetwork

# The classes that a saved DeepQNetwork object contains.
SAFE_CLASSES = [DeepQNetwork, nn.Sequential, nn.Linear, nn.ReLU]

LEGACY_FORMAT_MESSAGE = (
    "{path} uses the old PyTorch save format (before PyTorch 1.6). Safe mode cannot read it.\n"
    "If you trust this file, convert it one time with:\n"
    "    uv run python scripts/convert_legacy_model.py --path {path}"
)


def load_model(path):
    """Load a full DeepQNetwork object that torch.save(model, path) wrote. The model is on the CPU."""
    if not zipfile.is_zipfile(path):
        raise RuntimeError(LEGACY_FORMAT_MESSAGE.format(path=path))
    torch.serialization.add_safe_globals(SAFE_CLASSES)
    return torch.load(path, map_location="cpu", weights_only=True)
