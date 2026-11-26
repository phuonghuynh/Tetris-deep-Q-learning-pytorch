"""
One-time conversion of a model file from the old PyTorch save format to the new format.

WARNING: this script reads the old file with weights_only=False. In that mode, the file can
run any Python code on your computer. Run this script only if you trust the file.

After the conversion, src/model_io.load_model() reads the file in safe mode.
To restore the original file of the author: git checkout -- trained_models/tetris
"""
import argparse
import sys
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.deep_q_network import DeepQNetwork  # noqa: E402
from src.model_io import load_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", default=str(ROOT / "trained_models" / "tetris"),
                        help="Model file to convert in place")
    args = parser.parse_args()

    if zipfile.is_zipfile(args.path):
        print(f"{args.path} already uses the new format. No change.")
        return

    model = torch.load(args.path, map_location="cpu", weights_only=False)
    if not isinstance(model, DeepQNetwork):
        raise TypeError(f"Expected a DeepQNetwork, got {type(model).__name__}. No change.")

    torch.save(model, args.path)
    check = load_model(args.path)
    sample = torch.rand(5, 4)
    assert torch.allclose(model(sample), check(sample)), "Converted model gives different outputs"
    print(f"Converted {args.path} to the new format. Safe mode can now load it.")


if __name__ == "__main__":
    main()
