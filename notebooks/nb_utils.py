"""
Helper functions for the learning notebooks: draw boards, list afterstates, and play games.

These helpers do not draw on screen with OpenCV. They use matplotlib, so the pictures
show inside the notebook.
"""
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.tetris import Tetris

PIECE_NAMES = ["O", "T", "S", "Z", "I", "L", "J"]
FEATURE_NAMES = ["lines_cleared", "holes", "bumpiness", "height"]
COLORS = np.array(Tetris.piece_colors, dtype=float) / 255.0


def draw_board(board, ax=None, title=None):
    """Draw a board (a list of rows; 0 = empty cell, 1-7 = piece id)."""
    grid = np.array(board)
    if ax is None:
        _, ax = plt.subplots(figsize=(2.2, 4.4 * grid.shape[0] / 20))
    ax.imshow(COLORS[grid], interpolation="nearest")
    ax.grid(False)  # src/tetris.py sets the "ggplot" style, which draws extra grid lines
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#555555", linewidth=0.4)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def draw_boards(boards, titles=None, cols=5):
    """Draw several boards in a grid of small pictures."""
    rows = (len(boards) + cols - 1) // cols
    cell_h = max(len(b) for b in boards) / max(len(b[0]) for b in boards)
    fig, axes = plt.subplots(rows, cols, figsize=(1.9 * cols, (1.9 * cell_h + 0.6) * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i < len(boards):
            draw_board(boards[i], ax, titles[i] if titles else None)
        else:
            ax.axis("off")
    fig.tight_layout()
    plt.show()


def board_from_text(text):
    """Make a 20x10 board from text. '#' = block, '.' = empty. Missing top rows are empty."""
    rows = [[1 if c == "#" else 0 for c in line.strip()] for line in text.strip().splitlines()]
    empty = [[0] * 10 for _ in range(20 - len(rows))]
    return empty + rows


def features_text(features):
    """Short text for a 4-number state, for picture titles."""
    lines, holes, bump, height = [int(v) for v in features]
    return f"lines={lines} holes={holes}\nbump={bump} height={height}"


def afterstates(env):
    """
    Same logic as Tetris.get_next_states(), but each item also keeps the board.

    Returns a list of dicts: {"action": (x, rotation), "board": board, "features": tensor}.
    """
    results = []
    piece = [row[:] for row in env.piece]
    if env.ind == 0:
        num_rotations = 1
    elif env.ind in (2, 3, 4):
        num_rotations = 2
    else:
        num_rotations = 4
    for rotation in range(num_rotations):
        for x in range(env.width - len(piece[0]) + 1):
            test_piece = [row[:] for row in piece]
            pos = {"x": x, "y": 0}
            while not env.check_collision(test_piece, pos):
                pos["y"] += 1
            env.truncate(test_piece, pos)
            board = env.store(test_piece, pos)
            # get_state_properties() can delete rows of the list it gets, so give it a copy.
            features = env.get_state_properties([row[:] for row in board])
            results.append({"action": (x, rotation), "board": board, "features": features})
        piece = env.rotate(piece)
    return results


def score_afterstates(model, options):
    """Return the network value of each afterstate in options (a list from afterstates())."""
    with torch.no_grad():
        return model(torch.stack([o["features"] for o in options]))[:, 0]


def draw_top_choices(model, options, k=5):
    """Draw the k afterstates with the highest network value. #1 is the move the agent takes."""
    values = score_afterstates(model, options)
    order = torch.argsort(values, descending=True)[:k].tolist()
    titles = [f"#{n + 1} {options[i]['action']} value={values[i]:.1f}\n{features_text(options[i]['features'])}"
              for n, i in enumerate(order)]
    draw_boards([options[i]["board"] for i in order], titles, cols=k)


def feature_sweep(model, base=(0, 2, 6, 30), ranges=((0, 4), (0, 20), (0, 30), (0, 120))):
    """Change one feature at a time (others fixed at base) and plot the network value."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    for k, (ax, name, (low, high)) in enumerate(zip(axes, FEATURE_NAMES, ranges)):
        values = torch.linspace(low, high, 50)
        x = torch.tensor(base, dtype=torch.float32).repeat(50, 1)
        x[:, k] = values
        with torch.no_grad():
            ax.plot(values, model(x)[:, 0])
        ax.set_title(f"change only '{name}'", fontsize=10)
        ax.set_xlabel(name)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("network value")
    fig.suptitle(f"other features fixed at {dict(zip(FEATURE_NAMES, base))}", fontsize=9)
    fig.tight_layout()
    plt.show()


def random_policy(env):
    """Choose a random placement."""
    return random.choice(list(env.get_next_states().keys()))


def greedy_policy(model, use_move_reward=False, gamma=0.99):
    """
    Return a policy that chooses the placement with the highest network value.

    use_move_reward=True: choose by reward + gamma * value. The reward is 1 + lines² × width.
    """
    def policy(env):
        actions, states = zip(*env.get_next_states().items())
        with torch.no_grad():
            values = model(torch.stack(states))[:, 0]
        if use_move_reward:
            lines = torch.stack(states)[:, 0]
            values = 1 + lines ** 2 * env.width + gamma * values
        return actions[torch.argmax(values).item()]
    return policy


def play_game(policy, max_pieces=500, snapshot_every=0, seed=None):
    """
    Play one game without a window. Stop at game over or after max_pieces.

    Returns a dict with score, pieces, lines, and board snapshots (if snapshot_every > 0).
    """
    if seed is not None:
        random.seed(seed)
    env = Tetris(width=10, height=20, block_size=30)
    env.reset()
    snapshots = []
    done = False
    while not done and env.tetrominoes < max_pieces:
        _, done = env.step(policy(env), render=False)
        if snapshot_every and env.tetrominoes % snapshot_every == 0:
            snapshots.append(([row[:] for row in env.board], f"piece {env.tetrominoes}"))
    return {"score": env.score, "pieces": env.tetrominoes, "lines": env.cleared_lines,
            "game_over": done, "snapshots": snapshots, "final_board": env.board}


def evaluate(policy, games=5, max_pieces=500, seed=0):
    """Play several games. Return the average pieces, lines, and score."""
    results = [play_game(policy, max_pieces, seed=seed + i) for i in range(games)]
    return {key: float(np.mean([r[key] for r in results])) for key in ("pieces", "lines", "score")}
