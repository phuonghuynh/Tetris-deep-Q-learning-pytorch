# [PYTORCH] Deep Q-learning for playing Tetris

## Introduction

Here is my python source code for training an agent to play Tetris. It could be seen as a very basic example of Reinforcement Learning's application.
<p align="center">
  <img src="demo/tetris.gif" width=600><br/>
  <i>Tetris demo</i>
</p>

The demo could also be found at [youtube demo](https://youtu.be/g96x6uATAR8)

## How to use my code

With my code, you can:
* **Train your model from scratch** by running **python train.py**
* **Test your trained model** by running **python test.py**

## Trained models

You could find my trained model at **trained_models/tetris**
 
## Requirements

* **Python 3.12** (pinned in `.python-version`)
* **PyTorch**, **NumPy**, **Pillow**, **OpenCV**, **matplotlib**, **tensorboardX**, **TensorBoard**
* Exact versions: [requirements.txt](requirements.txt) (run the code) and [requirements-dev.txt](requirements-dev.txt) (Jupyter).

## Setup with uv

[uv](https://docs.astral.sh/uv/) installs Python and all packages from `pyproject.toml` and `uv.lock`:

```bash
uv sync                                         # create .venv and install everything
uv run python test.py                           # watch the trained agent play
mkdir -p my_models
uv run python train.py --saved_path my_models   # train your own agent
uv run tensorboard --logdir tensorboard         # see the training charts
```

Use `--saved_path my_models` for training. The default path overwrites the trained model in `trained_models/tetris`.

Without uv: `pip install -r requirements.txt -r requirements-dev.txt`.

To update the requirement files after a change to `pyproject.toml`:

```bash
uv export --format requirements-txt --no-hashes --no-emit-project --no-dev -o requirements.txt
uv export --format requirements-txt --no-hashes --no-emit-project --only-dev -o requirements-dev.txt
```

## Model files and safe loading

`test.py` loads models with [src/model_io.py](src/model_io.py), which uses PyTorch safe mode (`weights_only=True`).
Safe mode builds only the network classes, so a model file cannot run other code.

The file `trained_models/tetris` uses the old PyTorch save format (before PyTorch 1.6), which safe mode cannot read.
If you trust the file, convert it one time:

```bash
uv run python scripts/convert_legacy_model.py
```

The script loads the old file in unsafe mode (`weights_only=False`) and saves it again in the new format.
To restore the original file: `git checkout -- trained_models/tetris`.

## Learn with the notebooks

The [notebooks/](notebooks) folder holds 4 lessons for beginners, in English and Vietnamese (Tiếng Việt). Run them in order:

1. [01_game_and_state.ipynb](notebooks/01_game_and_state.ipynb): the board, the 4 state features, actions, afterstates, and the reward.
2. [02_network_and_decisions.ipynb](notebooks/02_network_and_decisions.ipynb): the network, ReLU, a forward pass by hand, how the agent chooses a move, and epsilon-greedy.
3. [03_training_loop.ipynb](notebooks/03_training_loop.ipynb): replay memory, the Bellman target, the loss, and a 5-minute training run.
4. [04_reward_lab.ipynb](notebooks/04_reward_lab.ipynb): train the same agent with 5 reward functions and compare what each one learns (about 7 minutes).

```bash
uv run jupyter lab notebooks
```

In VS Code, open a notebook and select the kernel `.venv/bin/python`.
