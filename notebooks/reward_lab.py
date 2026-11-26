"""
Reward lab: train the same agent with different reward functions and compare the results.

Only the reward changes. The game, the network, the seed and the settings stay the same.
"""
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from src.deep_q_network import DeepQNetwork
from src.tetris import Tetris

# Each reward function gets: n = lines cleared by this move, over = True at game over, w = board width.
REWARDS = {
    "author: 1 + N²×w": lambda n, over, w: 1 + n ** 2 * w,
    "no square: 1 + N×w": lambda n, over, w: 1 + n * w,
    "no ×w: 1 + N²": lambda n, over, w: 1 + n ** 2,
    "survival only: 1": lambda n, over, w: 1,
    "lines only: N²×w": lambda n, over, w: n ** 2 * w,
}


class RewardTetris(Tetris):
    """The same Tetris game, but step() returns reward_fn(lines, game_over, width)."""

    def __init__(self, reward_fn, **kwargs):
        self.reward_fn = reward_fn
        super().__init__(**kwargs)

    def step(self, action, render=False, video=None):
        lines_before = self.cleared_lines
        _, done = super().step(action, render=render, video=video)
        return self.reward_fn(self.cleared_lines - lines_before, done, self.width), done


def reward_table(rewards, width=10):
    """Text table: the reward of each function for 0 to 4 cleared lines, and for a move that ends the game."""
    rows = [f"{'reward function':<26}" + "".join(f"{n} lines".rjust(9) for n in range(5)) + "game over".rjust(11)]
    for name, fn in rewards.items():
        rows.append(f"{name:<26}" + "".join(f"{fn(n, False, width):>9}" for n in range(5))
                    + f"{fn(0, True, width):>11}")
    return "\n".join(rows)


def train_dqn(reward_fn, hp, seed=123, log_every=0):
    """
    The training loop of Lesson 3, Part 5, as a function.

    Returns (model, history). history has one row per game: (epoch, pieces, lines).
    """
    torch.manual_seed(seed)
    random.seed(seed)
    env = RewardTetris(reward_fn, width=10, height=20, block_size=30)
    model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    criterion = nn.MSELoss()
    memory = deque(maxlen=hp["replay_memory_size"])
    history = []

    state = env.reset()
    epoch, start = 0, time.time()
    while epoch < hp["num_epochs"]:
        epsilon = hp["final_epsilon"] + max(hp["num_decay_epochs"] - epoch, 0) * (
            hp["initial_epsilon"] - hp["final_epsilon"]) / hp["num_decay_epochs"]
        actions, states = zip(*env.get_next_states().items())
        states = torch.stack(states)
        if random.random() <= epsilon:
            index = random.randint(0, len(actions) - 1)
        else:
            with torch.no_grad():
                index = torch.argmax(model(states)[:, 0]).item()
        next_state = states[index]

        reward, done = env.step(actions[index])
        memory.append([state, reward, next_state, done])
        if not done and env.tetrominoes < hp["max_pieces"]:
            state = next_state
            continue

        history.append((epoch, env.tetrominoes, env.cleared_lines))
        state = env.reset()
        if len(memory) < hp["replay_memory_size"] / 10:
            continue

        epoch += 1
        batch = random.sample(memory, min(len(memory), hp["batch_size"]))
        s, r, ns, d = zip(*batch)
        s, ns = torch.stack(s), torch.stack(ns)
        r = torch.tensor(r, dtype=torch.float32)[:, None]
        d = torch.tensor(d, dtype=torch.float32)[:, None]
        with torch.no_grad():
            y = r + hp["gamma"] * model(ns) * (1 - d)
        loss = criterion(model(s), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if log_every and epoch % log_every == 0:
            last = np.array(history[-20:])
            print(f"  epoch {epoch:>4}  epsilon={epsilon:.2f}  last 20 games: pieces={last[:, 1].mean():6.1f}"
                  f"  lines={last[:, 2].mean():6.1f}  time={time.time() - start:4.0f}s")
    return model.eval(), history
