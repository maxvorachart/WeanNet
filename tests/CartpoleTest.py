import csv
import math
import os
import pickle
import random
import tempfile
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import gymnasium as gym
except ImportError as e:
    raise SystemExit(
        "Gymnasium is not installed.\n"
        "In the PyCharm terminal run:\n"
        "pip install \"gymnasium[classic-control]\" torch matplotlib numpy"
    ) from e

def select_device():
    """Use CUDA only when the installed PyTorch build supports this GPU."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        major, minor = torch.cuda.get_device_capability()
        supported_arches = set(torch.cuda.get_arch_list())
        if f"sm_{major}{minor}" in supported_arches:
            return torch.device("cuda")
    except (RuntimeError, AttributeError):
        pass
    return torch.device("cpu")


DEVICE = select_device()

METHODS = ["Regular", "Reset", "WarmStart", "Distilled", "WeanNet", "PNN-Reset", "PNN"]
GRAVITIES = [9.8, 14.7, 4.9, 19.6, 9.8]
EPISODES_PER_GENERATION = 300
NUM_SEEDS = 10
EVALUATION_EPISODES = 30

HIDDEN_SIZES = (64, 64)
LEARNING_RATE = 0.003
ENTROPY_COEF = 0.01
GAMMA = 0.99
UPDATE_EVERY_EPISODES = 10
MAX_EPISODE_STEPS = 500

LATERAL_INIT_SCALE = 0.1
LAMBDA_INIT = 1.0
LAMBDA_MIN = 0.0
DECAY_EPISODES = EPISODES_PER_GENERATION // 2
PARENT_OUTPUT = True
BASE_SEED = 12345




EVALUATION_SEEDS = tuple(BASE_SEED + index for index in range(NUM_SEEDS))
PNN_TUNING_SEEDS = tuple(BASE_SEED + 100_000 + index for index in range(3))
PNN_LEARNING_RATES = [0.001, 0.003, 0.01]
PNN_LATERAL_INIT_SCALES = [0.01, 0.1]

DISTILL_STEPS = 150
DISTILL_LR = 0.003
DISTILL_BUFFER_SIZE = 2048
DISTILL_BATCH_SIZE = 256
DISTILL_TEMPERATURE = 1.0

SMOOTH_WINDOW = 20

RAW_CSV = "cartpole_gravity_raw.csv"
SUMMARY_CSV = "cartpole_gravity_summary.csv"
EVALUATION_CSV = "cartpole_gravity_evaluation.csv"
EVALUATION_SUMMARY_CSV = "cartpole_gravity_evaluation_summary.csv"
PARENT_REMOVAL_CSV = "cartpole_parent_removal.csv"
PNN_TUNING_CSV = "cartpole_pnn_tuning.csv"
PNN_SELECTION_FILE = "cartpole_selected_pnn_config.txt"
PDF_GRAPH = "cartpole_gravity_drift.pdf"
PNG_GRAPH = "cartpole_gravity_drift.png"




CHECKPOINT_VERSION = 1
RESUME_FROM_CHECKPOINT = True
CHECKPOINT_FILE = "cartpole_gravity_checkpoint.pkl"


def checkpoint_signature():
    """Describe every setting that must match before a checkpoint is reused."""
    return {
        "version": CHECKPOINT_VERSION,
        "methods": tuple(METHODS),
        "gravities": tuple(GRAVITIES),
        "episodes_per_generation": EPISODES_PER_GENERATION,
        "evaluation_seeds": tuple(EVALUATION_SEEDS),
        "pnn_tuning_seeds": tuple(PNN_TUNING_SEEDS),
        "hidden_sizes": tuple(HIDDEN_SIZES),
        "learning_rate": LEARNING_RATE,
        "entropy_coef": ENTROPY_COEF,
        "gamma": GAMMA,
        "update_every_episodes": UPDATE_EVERY_EPISODES,
        "lateral_init_scale": LATERAL_INIT_SCALE,
        "lambda_init": LAMBDA_INIT,
        "lambda_min": LAMBDA_MIN,
        "decay_episodes": DECAY_EPISODES,
        "parent_output": PARENT_OUTPUT,
        "pnn_learning_rates": tuple(PNN_LEARNING_RATES),
        "pnn_lateral_init_scales": tuple(PNN_LATERAL_INIT_SCALES),
    }


def save_checkpoint(checkpoint):
    """Atomically replace the checkpoint so a forced close cannot corrupt it."""
    directory = os.path.dirname(os.path.abspath(CHECKPOINT_FILE))
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(CHECKPOINT_FILE)}.",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "wb") as file:
            pickle.dump(checkpoint, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, CHECKPOINT_FILE)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def new_checkpoint():
    return {
        "version": CHECKPOINT_VERSION,
        "signature": checkpoint_signature(),
        "status": "running",
        "pnn_tuning_scores": {},
        "selected_pnn_config": None,
        "evaluation_tasks": {},
    }


def load_or_create_checkpoint():
    if RESUME_FROM_CHECKPOINT and os.path.isfile(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "rb") as file:
                checkpoint = pickle.load(file)
            if checkpoint.get("signature") == checkpoint_signature():
                return checkpoint, True
            print("Existing checkpoint does not match this configuration; starting fresh.")
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError) as error:
            print(f"Could not read checkpoint ({error}); starting fresh.")

    checkpoint = new_checkpoint()
    save_checkpoint(checkpoint)
    return checkpoint, False


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_linear(layer, generator=None, scale=1.0):
    fan_in = layer.weight.shape[1]
    bound = math.sqrt(6.0 / fan_in) * scale
    with torch.no_grad():
        if generator is None:
            layer.weight.uniform_(-bound, bound)
        else:
            layer.weight.uniform_(-bound, bound, generator=generator)
        layer.bias.zero_()


def make_generator(seed, device="cpu"):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


class MLP(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, action_dim, seed):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_sizes = tuple(hidden_sizes)
        self.action_dim = action_dim
        sizes = [obs_dim, *hidden_sizes, action_dim]
        self.layers = nn.ModuleList()
        gen = make_generator(seed)
        for i in range(len(sizes) - 1):
            layer = nn.Linear(sizes[i], sizes[i + 1])
            init_linear(layer, gen)
            self.layers.append(layer)

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
        return h

    def forward_with_activations(self, x):
        acts = [x]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
            acts.append(h)
        return acts

    def reset_output_layer(self, seed):



        init_linear(
            self.layers[-1],
            make_generator(seed, device=self.layers[-1].weight.device),
        )

    def snapshot(self):
        snap = MLP(self.obs_dim, self.hidden_sizes, self.action_dim, 0).to(DEVICE)
        snap.load_state_dict(self.state_dict())
        snap.eval()
        for p in snap.parameters():
            p.requires_grad_(False)
        return snap


class LateralNet(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, action_dim, seed, parent=None,
                 use_decay=True, parent_output=True, lateral_init_scale=0.1):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_sizes = tuple(hidden_sizes)
        self.action_dim = action_dim
        self.parent = parent
        self.use_decay = use_decay
        self.parent_output = parent_output

        if self.parent is not None:
            self.parent.eval()
            for p in self.parent.parameters():
                p.requires_grad_(False)

        sizes = [obs_dim, *hidden_sizes, action_dim]
        self.layers = nn.ModuleList()
        self.laterals = nn.ModuleList()
        gen = make_generator(seed)

        for i in range(len(sizes) - 1):
            own = nn.Linear(sizes[i], sizes[i + 1])
            init_linear(own, gen)
            self.layers.append(own)

            if parent is not None and i >= 1:
                lat = nn.Linear(sizes[i], sizes[i + 1], bias=False)
                bound = math.sqrt(6.0 / lat.weight.shape[1]) * lateral_init_scale
                with torch.no_grad():
                    lat.weight.uniform_(-bound, bound, generator=gen)
                self.laterals.append(lat)
            else:
                self.laterals.append(nn.Identity())

        if parent is not None and parent_output:
            self.output_extra = nn.Linear(action_dim, action_dim, bias=False)
            bound = math.sqrt(6.0 / action_dim) * lateral_init_scale
            with torch.no_grad():
                self.output_extra.weight.uniform_(-bound, bound, generator=gen)
        else:
            self.output_extra = None

    def forward(self, x, lateral_scale=1.0):
        if self.parent is None:
            scale = 0.0
            parent_acts = None
        else:
            scale = lateral_scale if self.use_decay else 1.0
            with torch.no_grad():
                parent_acts = self.parent.forward_with_activations(x)

        h = x
        for i, layer in enumerate(self.layers):
            out = layer(h)
            if self.parent is not None and i >= 1:
                out = out + scale * self.laterals[i](parent_acts[i])
            if i == len(self.layers) - 1 and self.output_extra is not None:
                out = out + scale * self.output_extra(parent_acts[-1])
            if i < len(self.layers) - 1:
                out = F.relu(out)
            h = out
        return h

    def forward_with_activations(self, x):
        acts = [x]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
            acts.append(h)
        return acts

    def active_snapshot(self):
        snap = MLP(self.obs_dim, self.hidden_sizes, self.action_dim, 0).to(DEVICE)
        state = {}
        for i, layer in enumerate(self.layers):
            state[f"layers.{i}.weight"] = layer.weight.detach().clone()
            state[f"layers.{i}.bias"] = layer.bias.detach().clone()
        snap.load_state_dict(state)
        snap.eval()
        for p in snap.parameters():
            p.requires_grad_(False)
        return snap


class ProgressiveColumn(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, action_dim, seed, num_parents,
                 parent_output=True, lateral_init_scale=0.1):
        super().__init__()
        sizes = [obs_dim, *hidden_sizes, action_dim]
        self.parent_output = parent_output
        self.layers = nn.ModuleList()
        self.laterals = nn.ModuleList()
        self.output_extras = nn.ModuleList()
        gen = make_generator(seed)

        for i in range(len(sizes) - 1):
            own = nn.Linear(sizes[i], sizes[i + 1])
            init_linear(own, gen)
            self.layers.append(own)

            stage = nn.ModuleList()
            if i >= 1:
                for _ in range(num_parents):
                    lat = nn.Linear(sizes[i], sizes[i + 1], bias=False)
                    bound = math.sqrt(6.0 / lat.weight.shape[1]) * lateral_init_scale
                    with torch.no_grad():
                        lat.weight.uniform_(-bound, bound, generator=gen)
                    stage.append(lat)
            self.laterals.append(stage)

        if parent_output:
            for _ in range(num_parents):
                ex = nn.Linear(action_dim, action_dim, bias=False)
                bound = math.sqrt(6.0 / action_dim) * lateral_init_scale
                with torch.no_grad():
                    ex.weight.uniform_(-bound, bound, generator=gen)
                self.output_extras.append(ex)

    def forward(self, x, parent_acts_list=None):
        parent_acts_list = parent_acts_list or []
        h = x
        acts = [x]

        for i, layer in enumerate(self.layers):
            out = layer(h)
            if i >= 1:
                for pidx, parent_acts in enumerate(parent_acts_list):
                    out = out + self.laterals[i][pidx](parent_acts[i])
            if i == len(self.layers) - 1 and self.parent_output:
                for pidx, parent_acts in enumerate(parent_acts_list):
                    out = out + self.output_extras[pidx](parent_acts[-1])
            if i < len(self.layers) - 1:
                out = F.relu(out)
            h = out
            acts.append(h)
        return h, acts


class ProgressiveNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, action_dim, seed,
                 parent_output=True, lateral_init_scale=0.1):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_sizes = tuple(hidden_sizes)
        self.action_dim = action_dim
        self.seed = seed
        self.parent_output = parent_output
        self.lateral_init_scale = lateral_init_scale
        self.columns = nn.ModuleList()
        self.add_column()

    def add_column(self):
        idx = len(self.columns)
        col = ProgressiveColumn(
            self.obs_dim, self.hidden_sizes, self.action_dim,
            self.seed + idx * 10007, idx,
            self.parent_output, self.lateral_init_scale
        ).to(DEVICE)
        self.columns.append(col)

    def freeze_and_grow(self):
        for col in self.columns:
            for p in col.parameters():
                p.requires_grad_(False)
        self.add_column()

    def forward(self, x):
        activations = []
        logits = None
        for col in self.columns:
            logits, acts = col(x, activations)
            activations.append(acts)
        return logits

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


class BaseTrainer:
    name = "Base"

    def __init__(self, obs_dim, hidden_sizes, action_dim, seed):
        self.obs_dim = obs_dim
        self.hidden_sizes = tuple(hidden_sizes)
        self.action_dim = action_dim
        self.seed = seed
        self.net = MLP(obs_dim, hidden_sizes, action_dim, seed).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)

    def on_generation(self, generation_index, state_bank):
        pass

    def current_lambda(self):
        return None

    def logits(self, obs):
        return self.net(obs)

    def act(self, obs):
        logits = self.logits(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def update(self, log_probs, returns, entropies):
        if not log_probs:
            return
        lp = torch.cat(log_probs)
        rt = torch.cat(returns)
        en = torch.cat(entropies)
        if rt.numel() > 1:
            rt = (rt - rt.mean()) / (rt.std(unbiased=False) + 1e-8)
        loss = -(lp * rt).mean() - ENTROPY_COEF * en.mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()


class RegularTrainer(BaseTrainer):
    name = "Regular"


class ResetTrainer(BaseTrainer):
    name = "Reset"

    def on_generation(self, generation_index, state_bank):
        self.net = MLP(self.obs_dim, self.hidden_sizes, self.action_dim,
                       self.seed + generation_index * 10000).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)


class WarmStartTrainer(BaseTrainer):
    name = "WarmStart"

    def on_generation(self, generation_index, state_bank):
        self.net.reset_output_layer(self.seed + generation_index * 20000)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)


class DistilledTrainer(BaseTrainer):
    name = "Distilled"

    def on_generation(self, generation_index, state_bank):
        teacher = self.net
        teacher.eval()
        student = MLP(self.obs_dim, self.hidden_sizes, self.action_dim,
                      self.seed + generation_index * 40000).to(DEVICE)

        if len(state_bank) > 0:
            states = torch.from_numpy(np.asarray(state_bank, dtype=np.float32)).to(DEVICE)
            with torch.no_grad():
                teacher_probs = F.softmax(teacher(states) / DISTILL_TEMPERATURE, dim=-1)
            opt = torch.optim.Adam(student.parameters(), lr=DISTILL_LR)
            for _ in range(DISTILL_STEPS):
                count = min(DISTILL_BATCH_SIZE, states.shape[0])
                idx = torch.randint(0, states.shape[0], (count,), device=DEVICE)
                student_log_probs = F.log_softmax(student(states[idx]) / DISTILL_TEMPERATURE, dim=-1)
                loss = F.kl_div(student_log_probs, teacher_probs[idx], reduction="batchmean")
                opt.zero_grad()
                loss.backward()
                opt.step()

        self.net = student
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)


class WeanNetTrainer(BaseTrainer):
    name = "WeanNet"

    def __init__(self, obs_dim, hidden_sizes, action_dim, seed):
        self.obs_dim = obs_dim
        self.hidden_sizes = tuple(hidden_sizes)
        self.action_dim = action_dim
        self.seed = seed
        self.episodes_since_generation = 0
        self.net = LateralNet(obs_dim, hidden_sizes, action_dim, seed, parent=None,
                              use_decay=True, parent_output=PARENT_OUTPUT,
                              lateral_init_scale=LATERAL_INIT_SCALE).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)

    def on_generation(self, generation_index, state_bank):
        parent = self.net.active_snapshot()
        self.net = LateralNet(
            self.obs_dim, self.hidden_sizes, self.action_dim,
            self.seed + generation_index * 30000,
            parent=parent, use_decay=True,
            parent_output=PARENT_OUTPUT,
            lateral_init_scale=LATERAL_INIT_SCALE
        ).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.episodes_since_generation = 0

    def current_lambda(self):
        if self.net.parent is None:
            return 0.0
        progress = min(self.episodes_since_generation / max(DECAY_EPISODES, 1), 1.0)
        return LAMBDA_MIN + (LAMBDA_INIT - LAMBDA_MIN) * (1.0 - progress)

    def logits(self, obs):
        return self.net(obs, lateral_scale=self.current_lambda())

    def episode_finished(self):
        if self.net.parent is not None:
            self.episodes_since_generation += 1


class PNNResetTrainer(WeanNetTrainer):
    name = "PNN-Reset"

    def __init__(self, obs_dim, hidden_sizes, action_dim, seed):
        self.obs_dim = obs_dim
        self.hidden_sizes = tuple(hidden_sizes)
        self.action_dim = action_dim
        self.seed = seed
        self.episodes_since_generation = 0
        self.net = LateralNet(obs_dim, hidden_sizes, action_dim, seed, parent=None,
                              use_decay=False, parent_output=PARENT_OUTPUT,
                              lateral_init_scale=LATERAL_INIT_SCALE).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)

    def on_generation(self, generation_index, state_bank):
        parent = self.net.active_snapshot()
        self.net = LateralNet(
            self.obs_dim, self.hidden_sizes, self.action_dim,
            self.seed + generation_index * 31000,
            parent=parent, use_decay=False,
            parent_output=PARENT_OUTPUT,
            lateral_init_scale=LATERAL_INIT_SCALE
        ).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.episodes_since_generation = 0

    def current_lambda(self):
        return 1.0 if self.net.parent is not None else 0.0

    def episode_finished(self):
        pass


class PNNTrainer(BaseTrainer):
    name = "PNN"

    def __init__(
        self,
        obs_dim,
        hidden_sizes,
        action_dim,
        seed,
        learning_rate=LEARNING_RATE,
        lateral_init_scale=LATERAL_INIT_SCALE,
    ):
        self.obs_dim = obs_dim
        self.hidden_sizes = tuple(hidden_sizes)
        self.action_dim = action_dim
        self.seed = seed
        self.learning_rate = float(learning_rate)
        self.lateral_init_scale = float(lateral_init_scale)
        self.net = ProgressiveNetwork(
            obs_dim, hidden_sizes, action_dim, seed,
            parent_output=PARENT_OUTPUT,
            lateral_init_scale=self.lateral_init_scale,
        ).to(DEVICE)
        self.optimizer = torch.optim.Adam(
            self.net.trainable_parameters(),
            lr=self.learning_rate,
        )

    def on_generation(self, generation_index, state_bank):
        self.net.freeze_and_grow()
        self.optimizer = torch.optim.Adam(
            self.net.trainable_parameters(),
            lr=self.learning_rate,
        )


TRAINER_CLASSES = {
    "Regular": RegularTrainer,
    "Reset": ResetTrainer,
    "WarmStart": WarmStartTrainer,
    "Distilled": DistilledTrainer,
    "WeanNet": WeanNetTrainer,
    "PNN-Reset": PNNResetTrainer,
    "PNN": PNNTrainer,
}


def discounted_returns(rewards, gamma):
    out = []
    g = 0.0
    for reward in reversed(rewards):
        g = reward + gamma * g
        out.append(g)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32, device=DEVICE)


def make_trainer(method_name, obs_dim, hidden_sizes, action_dim, seed, pnn_config=None):
    if method_name == "PNN":
        return PNNTrainer(
            obs_dim,
            hidden_sizes,
            action_dim,
            seed,
            **(pnn_config or {}),
        )
    return TRAINER_CLASSES[method_name](obs_dim, hidden_sizes, action_dim, seed)


def parameter_count(net):
    return sum(parameter.numel() for parameter in net.parameters())


def evaluation_episode_seeds(seed, generation):
    return tuple(
        seed * 1_000_003 + generation * 10_007 + 700_000 + episode
        for episode in range(EVALUATION_EPISODES)
    )


def evaluate_greedy(logits_fn, gravity, episode_seeds, comparison_logits_fn=None):
    """Evaluate a fixed policy without sampling training actions or touching RNG state."""
    env = gym.make("CartPole-v1", max_episode_steps=MAX_EPISODE_STEPS)
    env.unwrapped.gravity = float(gravity)
    returns = []
    max_abs_logit_difference = 0.0
    action_disagreements = 0
    compared_states = 0

    try:
        with torch.inference_mode():
            for episode_seed in episode_seeds:
                obs, _ = env.reset(seed=int(episode_seed))
                env.action_space.seed(int(episode_seed) + 17)
                total_reward = 0.0
                done = False

                while not done:
                    obs_t = torch.tensor(
                        obs,
                        dtype=torch.float32,
                        device=DEVICE,
                    ).unsqueeze(0)
                    logits = logits_fn(obs_t)

                    if comparison_logits_fn is not None:
                        comparison_logits = comparison_logits_fn(obs_t)
                        max_abs_logit_difference = max(
                            max_abs_logit_difference,
                            float((logits - comparison_logits).abs().max().item()),
                        )
                        action_disagreements += int(
                            logits.argmax(dim=-1).item()
                            != comparison_logits.argmax(dim=-1).item()
                        )
                        compared_states += 1

                    action = int(logits.argmax(dim=-1).item())
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += float(reward)
                    done = terminated or truncated

                returns.append(total_reward)
    finally:
        env.close()

    return {
        "mean_return": float(np.mean(returns)),
        "max_abs_logit_difference": max_abs_logit_difference,
        "greedy_action_disagreements": action_disagreements,
        "compared_states": compared_states,
    }


def evaluate_trainer(trainer, gravity, episode_seeds):
    was_training = trainer.net.training
    trainer.net.eval()
    try:
        return evaluate_greedy(trainer.logits, gravity, episode_seeds)
    finally:
        trainer.net.train(was_training)


def verify_parent_removal(trainer, seed, generation, gravity):
    """Compare the zero-gain attached model with a physically detached MLP."""
    if (
        not isinstance(trainer, WeanNetTrainer)
        or isinstance(trainer, PNNResetTrainer)
        or trainer.net.parent is None
    ):
        return None

    lambda_before_removal = trainer.current_lambda()
    if not np.isclose(lambda_before_removal, 0.0, atol=1e-12):
        raise RuntimeError(
            "Deployment verification requires the WeanNet parent gain to have "
            "decayed to zero."
        )

    detached = trainer.net.active_snapshot()
    episode_seeds = evaluation_episode_seeds(seed, generation)
    was_training = trainer.net.training
    trainer.net.eval()
    detached.eval()

    try:
        attached = evaluate_greedy(
            lambda obs: trainer.net(obs, lateral_scale=0.0),
            gravity,
            episode_seeds,
            comparison_logits_fn=detached,
        )
        detached_result = evaluate_greedy(detached, gravity, episode_seeds)
    finally:
        trainer.net.train(was_training)

    return {
        "lambda_before_removal": float(lambda_before_removal),
        "attached_mean_return": attached["mean_return"],
        "detached_mean_return": detached_result["mean_return"],
        "detached_minus_attached_return": (
            detached_result["mean_return"] - attached["mean_return"]
        ),
        "max_abs_logit_difference": attached["max_abs_logit_difference"],
        "greedy_action_disagreements": attached["greedy_action_disagreements"],
        "compared_states": attached["compared_states"],
        "attached_parameters": parameter_count(trainer.net),
        "detached_parameters": parameter_count(detached),
        "evaluation_episodes": len(episode_seeds),
    }


def run_one_method(method_name, seed, pnn_config=None):
    seed_everything(seed)
    env = gym.make("CartPole-v1", max_episode_steps=MAX_EPISODE_STEPS)
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)
    trainer = make_trainer(
        method_name,
        obs_dim,
        HIDDEN_SIZES,
        action_dim,
        seed,
        pnn_config=pnn_config,
    )
    state_bank = deque(maxlen=DISTILL_BUFFER_SIZE)

    rows = []
    evaluation_rows = []
    removal_rows = []
    batch_log_probs, batch_returns, batch_entropies = [], [], []
    global_episode = 0

    for generation, gravity in enumerate(GRAVITIES):
        env.unwrapped.gravity = float(gravity)
        if generation > 0:
            trainer.on_generation(generation, list(state_bank))

        for episode_in_generation in range(EPISODES_PER_GENERATION):
            episode_seed = seed * 1_000_003 + generation * 10_007 + episode_in_generation
            obs, _ = env.reset(seed=episode_seed)
            env.action_space.seed(episode_seed + 17)

            rewards, log_probs, entropies = [], [], []
            done = False

            while not done:
                state_bank.append(np.asarray(obs, dtype=np.float32).copy())
                obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                action, log_prob, entropy = trainer.act(obs_t)
                obs, reward, terminated, truncated, _ = env.step(int(action.item()))
                done = terminated or truncated
                rewards.append(float(reward))
                log_probs.append(log_prob.reshape(1))
                entropies.append(entropy.reshape(1))

            batch_log_probs.append(torch.cat(log_probs))
            batch_returns.append(discounted_returns(rewards, GAMMA))
            batch_entropies.append(torch.cat(entropies))

            rows.append({
                "method": method_name,
                "seed": seed,
                "generation": generation,
                "gravity": gravity,
                "episode_in_generation": episode_in_generation,
                "global_episode": global_episode,
                "return": float(sum(rewards)),
                "lambda": trainer.current_lambda() if trainer.current_lambda() is not None else np.nan,
            })

            if hasattr(trainer, "episode_finished"):
                trainer.episode_finished()

            global_episode += 1

            if len(batch_log_probs) >= UPDATE_EVERY_EPISODES:
                trainer.update(batch_log_probs, batch_returns, batch_entropies)
                batch_log_probs, batch_returns, batch_entropies = [], [], []

        if batch_log_probs:
            trainer.update(batch_log_probs, batch_returns, batch_entropies)
            batch_log_probs, batch_returns, batch_entropies = [], [], []

        verification = verify_parent_removal(trainer, seed, generation, gravity)
        if verification is None:
            evaluation = evaluate_trainer(
                trainer,
                gravity,
                evaluation_episode_seeds(seed, generation),
            )
            evaluation_rows.append({
                "method": method_name,
                "seed": seed,
                "generation": generation,
                "gravity": gravity,
                "greedy_eval_return": evaluation["mean_return"],
                "lambda": (
                    trainer.current_lambda()
                    if trainer.current_lambda() is not None
                    else np.nan
                ),
                "resident_parameters": parameter_count(trainer.net),
                "detached_eval_return": np.nan,
                "detached_parameters": np.nan,
                "detached_minus_attached_return": np.nan,
                "max_abs_logit_difference": np.nan,
                "greedy_action_disagreements": np.nan,
                "parent_present": int(
                    isinstance(trainer.net, LateralNet)
                    and trainer.net.parent is not None
                ),
                "evaluation_episodes": EVALUATION_EPISODES,
            })
        else:
            evaluation_rows.append({
                "method": method_name,
                "seed": seed,
                "generation": generation,
                "gravity": gravity,
                "greedy_eval_return": verification["attached_mean_return"],
                "lambda": verification["lambda_before_removal"],
                "resident_parameters": verification["attached_parameters"],
                "detached_eval_return": verification["detached_mean_return"],
                "detached_parameters": verification["detached_parameters"],
                "detached_minus_attached_return": (
                    verification["detached_minus_attached_return"]
                ),
                "max_abs_logit_difference": verification["max_abs_logit_difference"],
                "greedy_action_disagreements": (
                    verification["greedy_action_disagreements"]
                ),
                "parent_present": 1,
                "evaluation_episodes": verification["evaluation_episodes"],
            })
            removal_rows.append({
                "method": method_name,
                "seed": seed,
                "generation": generation,
                "gravity": gravity,
                **verification,
            })

    env.close()
    return rows, evaluation_rows, removal_rows


def moving_average(values, window):
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    q = deque()
    running = 0.0
    for i, value in enumerate(values):
        q.append(value)
        running += value
        if len(q) > window:
            running -= q.popleft()
        out[i] = running / len(q)
    return out


def mean_and_sd(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    return float(values.mean()), float(values.std(ddof=1)) if values.size > 1 else 0.0


def save_csvs(rows, evaluation_rows, removal_rows):
    raw_fields = [
        "method",
        "seed",
        "generation",
        "gravity",
        "episode_in_generation",
        "global_episode",
        "return",
        "lambda",
    ]
    with open(RAW_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for method in METHODS:
        for generation, gravity in enumerate(GRAVITIES):
            seed_means = []
            for seed in EVALUATION_SEEDS:
                values = [
                    row["return"]
                    for row in rows
                    if row["method"] == method
                    and row["seed"] == seed
                    and row["generation"] == generation
                ]
                seed_means.append(float(np.mean(values)))
            mean_return, sd_return = mean_and_sd(seed_means)
            summary.append({
                "method": method,
                "generation": generation,
                "gravity": gravity,
                "mean_return": mean_return,
                "sd_across_seeds": sd_return,
                "num_seeds": len(EVALUATION_SEEDS),
            })

    fields = [
        "method",
        "generation",
        "gravity",
        "mean_return",
        "sd_across_seeds",
        "num_seeds",
    ]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    evaluation_fields = [
        "method",
        "seed",
        "generation",
        "gravity",
        "greedy_eval_return",
        "lambda",
        "resident_parameters",
        "detached_eval_return",
        "detached_parameters",
        "detached_minus_attached_return",
        "max_abs_logit_difference",
        "greedy_action_disagreements",
        "parent_present",
        "evaluation_episodes",
    ]
    with open(EVALUATION_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=evaluation_fields)
        writer.writeheader()
        writer.writerows(evaluation_rows)

    evaluation_summary = []
    for method in METHODS:
        for generation, gravity in enumerate(GRAVITIES):
            matching = [
                row
                for row in evaluation_rows
                if row["method"] == method and row["generation"] == generation
            ]
            attached_mean, attached_sd = mean_and_sd(
                [row["greedy_eval_return"] for row in matching]
            )
            detached_mean, detached_sd = mean_and_sd(
                [row["detached_eval_return"] for row in matching]
            )
            delta_mean, delta_sd = mean_and_sd(
                [row["detached_minus_attached_return"] for row in matching]
            )
            evaluation_summary.append({
                "method": method,
                "generation": generation,
                "gravity": gravity,
                "attached_mean_return": attached_mean,
                "attached_sd_across_seeds": attached_sd,
                "detached_mean_return": detached_mean,
                "detached_sd_across_seeds": detached_sd,
                "detached_minus_attached_mean": delta_mean,
                "detached_minus_attached_sd": delta_sd,
                "num_seeds": len(matching),
            })

    evaluation_summary_fields = [
        "method",
        "generation",
        "gravity",
        "attached_mean_return",
        "attached_sd_across_seeds",
        "detached_mean_return",
        "detached_sd_across_seeds",
        "detached_minus_attached_mean",
        "detached_minus_attached_sd",
        "num_seeds",
    ]
    with open(EVALUATION_SUMMARY_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=evaluation_summary_fields)
        writer.writeheader()
        writer.writerows(evaluation_summary)

    removal_fields = [
        "method",
        "seed",
        "generation",
        "gravity",
        "lambda_before_removal",
        "attached_mean_return",
        "detached_mean_return",
        "detached_minus_attached_return",
        "max_abs_logit_difference",
        "greedy_action_disagreements",
        "compared_states",
        "attached_parameters",
        "detached_parameters",
        "evaluation_episodes",
    ]
    with open(PARENT_REMOVAL_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=removal_fields)
        writer.writeheader()
        writer.writerows(removal_rows)


def plot_results(rows):
    by_method_seed = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_method_seed[row["method"]][row["seed"]].append((row["global_episode"], row["return"]))

    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    total_episodes = len(GRAVITIES) * EPISODES_PER_GENERATION
    x = np.arange(total_episodes)

    for method in METHODS:
        curves = []
        for seed in sorted(by_method_seed[method]):
            pairs = sorted(by_method_seed[method][seed])
            vals = np.asarray([v for _, v in pairs], dtype=np.float64)
            curves.append(moving_average(vals, SMOOTH_WINDOW))
        arr = np.stack(curves)
        mean = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=1)
        line = ax.plot(x, mean, linewidth=2.0, label=method)[0]
        ax.fill_between(x, np.maximum(0, mean - sd), np.minimum(MAX_EPISODE_STEPS, mean + sd),
                        alpha=0.12, color=line.get_color(), linewidth=0)

    for g in range(1, len(GRAVITIES)):
        ax.axvline(g * EPISODES_PER_GENERATION, linestyle="--", linewidth=1.0, alpha=0.55, color="black")

    for g, gravity in enumerate(GRAVITIES):
        center = g * EPISODES_PER_GENERATION + EPISODES_PER_GENERATION / 2
        ax.text(center, MAX_EPISODE_STEPS * 0.97, f"g = {gravity:g}", ha="center", va="top", fontsize=9)

    ax.set_xlim(0, total_episodes - 1)
    ax.set_ylim(0, MAX_EPISODE_STEPS)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode return")
    ax.set_title(
        f"CartPole under sequential gravity changes\n"
        f"Mean over {NUM_SEEDS} seeds; shaded region = ±1 SD; {SMOOTH_WINDOW}-episode moving average"
    )
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(PDF_GRAPH, bbox_inches="tight")
    fig.savefig(PNG_GRAPH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(rows):
    print("\n=== Mean return per generation (mean ± SD across seeds) ===")
    for method in METHODS:
        print(f"\n{method}")
        for generation, gravity in enumerate(GRAVITIES):
            seed_means = []
            for seed in EVALUATION_SEEDS:
                vals = [
                    row["return"]
                    for row in rows
                    if row["method"] == method
                    and row["seed"] == seed
                    and row["generation"] == generation
                ]
                seed_means.append(np.mean(vals))
            arr = np.asarray(seed_means)
            print(f"  Gen {generation + 1} | g={gravity:4.1f} | {arr.mean():7.2f} ± {arr.std(ddof=1):6.2f}")


def run_pnn_tuning(checkpoint):
    """Select full-PNN settings on held-out seeds before final evaluation."""
    print("\n=== Full-PNN tuning on held-out validation seeds ===")
    rows = []
    best_config = None
    best_score = -float("inf")

    for learning_rate in PNN_LEARNING_RATES:
        for lateral_init_scale in PNN_LATERAL_INIT_SCALES:
            config = {
                "learning_rate": learning_rate,
                "lateral_init_scale": lateral_init_scale,
            }
            seed_scores = []

            print(
                "Tuning PNN "
                f"| lr={learning_rate:g} "
                f"| lateral_init={lateral_init_scale:g}"
            )
            for seed in PNN_TUNING_SEEDS:
                task_key = (float(learning_rate), float(lateral_init_scale), int(seed))
                score = checkpoint["pnn_tuning_scores"].get(task_key)
                if score is None:
                    _, evaluation_rows, _ = run_one_method(
                        "PNN",
                        seed,
                        pnn_config=config,
                    )
                    post_shift_returns = [
                        row["greedy_eval_return"]
                        for row in evaluation_rows
                        if row["generation"] > 0
                    ]
                    score = float(np.mean(post_shift_returns))
                    checkpoint["pnn_tuning_scores"][task_key] = score
                    save_checkpoint(checkpoint)
                else:
                    print(f"  Resuming saved tuning seed {seed}")
                seed_scores.append(score)

            mean_score, sd_score = mean_and_sd(seed_scores)
            rows.append({
                "learning_rate": learning_rate,
                "lateral_init_scale": lateral_init_scale,
                "selection_metric": "mean_greedy_eval_return_post_shift_generations",
                "validation_seed_count": len(PNN_TUNING_SEEDS),
                "validation_seeds": ",".join(map(str, PNN_TUNING_SEEDS)),
                "mean_score": mean_score,
                "sd_across_validation_seeds": sd_score,
                "selected": False,
            })

            if mean_score > best_score:
                best_score = mean_score
                best_config = config

    for row in rows:
        row["selected"] = (
            row["learning_rate"] == best_config["learning_rate"]
            and row["lateral_init_scale"] == best_config["lateral_init_scale"]
        )

    fields = [
        "learning_rate",
        "lateral_init_scale",
        "selection_metric",
        "validation_seed_count",
        "validation_seeds",
        "mean_score",
        "sd_across_validation_seeds",
        "selected",
    ]
    with open(PNN_TUNING_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with open(PNN_SELECTION_FILE, "w", encoding="utf-8") as file:
        file.write(f"learning_rate={best_config['learning_rate']}\n")
        file.write(f"lateral_init_scale={best_config['lateral_init_scale']}\n")
        file.write("selection_metric=mean_greedy_eval_return_post_shift_generations\n")
        file.write("validation_seeds=" + ",".join(map(str, PNN_TUNING_SEEDS)) + "\n")
        file.write("evaluation_seeds=" + ",".join(map(str, EVALUATION_SEEDS)) + "\n")

    print(
        "Selected PNN configuration "
        f"| lr={best_config['learning_rate']:g} "
        f"| lateral_init={best_config['lateral_init_scale']:g}"
    )
    checkpoint["selected_pnn_config"] = dict(best_config)
    save_checkpoint(checkpoint)
    return best_config


def main():
    checkpoint, resumed = load_or_create_checkpoint()
    print(f"Device: {DEVICE}")
    print(f"Gravities: {GRAVITIES}")
    print(f"Episodes/generation: {EPISODES_PER_GENERATION}")
    print(f"Evaluation seeds: {len(EVALUATION_SEEDS)}")
    print(f"PNN tuning seeds: {len(PNN_TUNING_SEEDS)} (held out)")
    print(f"WeanNet lambda: {LAMBDA_INIT} -> {LAMBDA_MIN} over {DECAY_EPISODES} episodes")
    print(f"Parent output: {PARENT_OUTPUT}\n")
    if resumed:
        print(
            "Resuming from checkpoint: "
            f"{len(checkpoint['pnn_tuning_scores'])} tuning seeds and "
            f"{len(checkpoint['evaluation_tasks'])} evaluation runs already saved.\n"
        )

    try:
        selected_pnn_config = run_pnn_tuning(checkpoint)

        all_rows = []
        all_evaluation_rows = []
        all_removal_rows = []
        for method in METHODS:
            for seed_idx, seed in enumerate(EVALUATION_SEEDS, start=1):
                task_key = (method, int(seed))
                task_result = checkpoint["evaluation_tasks"].get(task_key)
                if task_result is None:
                    print(
                        f"Running {method:10s} "
                        f"| evaluation seed {seed_idx}/{len(EVALUATION_SEEDS)}"
                    )
                    rows, evaluation_rows, removal_rows = run_one_method(
                        method,
                        seed,
                        pnn_config=selected_pnn_config if method == "PNN" else None,
                    )
                    task_result = {
                        "rows": rows,
                        "evaluation_rows": evaluation_rows,
                        "removal_rows": removal_rows,
                    }
                    checkpoint["evaluation_tasks"][task_key] = task_result
                    save_checkpoint(checkpoint)
                else:
                    print(
                        f"Resuming {method:10s} "
                        f"| evaluation seed {seed_idx}/{len(EVALUATION_SEEDS)}"
                    )

                all_rows.extend(task_result["rows"])
                all_evaluation_rows.extend(task_result["evaluation_rows"])
                all_removal_rows.extend(task_result["removal_rows"])

        save_csvs(all_rows, all_evaluation_rows, all_removal_rows)
        plot_results(all_rows)
        print_summary(all_rows)

        checkpoint["status"] = "complete"
        save_checkpoint(checkpoint)
        print("\nFinished.")
        print(f"Raw CSV:     {RAW_CSV}")
        print(f"Summary CSV: {SUMMARY_CSV}")
        print(f"Evaluation:  {EVALUATION_CSV}")
        print(f"Removal CSV: {PARENT_REMOVAL_CSV}")
        print(f"PNN tuning:  {PNN_TUNING_CSV}")
        print(f"PDF graph:   {PDF_GRAPH}")
        print(f"PNG graph:   {PNG_GRAPH}")
        print(f"Checkpoint:  {CHECKPOINT_FILE}")
    except BaseException:



        print(f"\nStopped. Resume later from: {CHECKPOINT_FILE}")
        raise


if __name__ == "__main__":
    main()
