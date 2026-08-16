"""Concept-drift experiment comparing WeanNet with transfer-learning baselines."""

import csv
import math
import multiprocessing
import os
import pickle
import shutil
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom")

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


# config













RUN_MODE = "all"

OUTPUT_DIR = "concept_drift_revision_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)



CHECKPOINT_VERSION = 1
RESUME_FROM_CHECKPOINT = True
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "concept_drift_checkpoint.pkl")
RUN_CHECKPOINT = None




GPU_WORKERS = 4




CHECKPOINT_SAFETY_BACKUP = CHECKPOINT_FILE + ".pre_parallel_backup"


# defaults

@dataclass(frozen=True)
class Config:
    lr: float = 0.02
    temperature: float = 1.0
    entropy_coef: float = 0.0
    ema_beta: float = 0.1

    lateral_init_scale: float = 0.1
    lambda_init: float = 1.0
    lambda_min: float = 0.0
    decay_batches: int = 300
    parent_output: bool = True


    pnn_lr: float = 0.02
    pnn_lateral_init_scale: float = 0.1

    distill_steps: int = 150
    distill_lr: float = 0.05

    generations: int = 10
    generation_length: int = 600
    batch_size: int = 32
    num_parallel: int = 8
    hidden_sizes: tuple = (64, 64)

    num_states: int = 64
    num_actions: int = 8
    feature_dim: int = 24
    drift_rate: float = 0.1
    reset_decoys_each_generation: bool = False

    baseline_mode: str = "batchZscore"


DEFAULT = Config()
EVALUATION_ENV_SEEDS = list(range(10))
PNN_TUNING_ENV_SEEDS = [100, 101, 102]
assert set(EVALUATION_ENV_SEEDS).isdisjoint(PNN_TUNING_ENV_SEEDS)
SEED_BASE = 100000

METHOD_ORDER = [
    "Regular",
    "Reset",
    "WarmStart",
    "Distilled",
    "WeanNet",
    "PNN-Reset",
    "PNN",
]

METRIC_KEYS = [
    "coreAcc",
    "driftAcc",
    "driftTrap",
    "coreTrap",
    "rewCore",
    "rewDrift",
    "rewOverall",
    "entAll",
    "entDrift",
]


def checkpoint_signature():
    """Describe the manuscript run before accepting a saved checkpoint."""
    return {
        "version": CHECKPOINT_VERSION,
        "run_mode": RUN_MODE,
        "default_config": repr(DEFAULT),
        "method_order": tuple(METHOD_ORDER),
        "evaluation_env_seeds": tuple(EVALUATION_ENV_SEEDS),
        "pnn_tuning_env_seeds": tuple(PNN_TUNING_ENV_SEEDS),
        "seed_base": SEED_BASE,
    }


def save_checkpoint(checkpoint):
    """Atomically replace the checkpoint so an abrupt close leaves it valid."""
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
        "completed_runs": {},
    }


def load_or_create_checkpoint():
    if RESUME_FROM_CHECKPOINT and os.path.isfile(CHECKPOINT_FILE):


        if not os.path.exists(CHECKPOINT_SAFETY_BACKUP):
            try:
                shutil.copy2(CHECKPOINT_FILE, CHECKPOINT_SAFETY_BACKUP)
                print(f"Safety backup created: {os.path.abspath(CHECKPOINT_SAFETY_BACKUP)}")
            except OSError as error:
                raise RuntimeError(
                    "Refusing to modify the existing checkpoint because its "
                    f"safety backup could not be created: {error}"
                ) from error

        try:
            with open(CHECKPOINT_FILE, "rb") as file:
                checkpoint = pickle.load(file)
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError) as error:
            raise RuntimeError(
                "Existing checkpoint could not be read. Refusing to start fresh "
                "or overwrite it. Restore/check the checkpoint manually. "
                f"Backup: {os.path.abspath(CHECKPOINT_SAFETY_BACKUP)}"
            ) from error

        if checkpoint.get("signature") != checkpoint_signature():
            raise RuntimeError(
                "Existing checkpoint does not match this configuration. "
                "Refusing to start fresh or overwrite old data. "
                f"Checkpoint: {os.path.abspath(CHECKPOINT_FILE)} | "
                f"Backup: {os.path.abspath(CHECKPOINT_SAFETY_BACKUP)}"
            )

        return checkpoint, True


    checkpoint = new_checkpoint()
    save_checkpoint(checkpoint)
    return checkpoint, False


def run_checkpoint_key(method_name, env_seed, train_seed, cfg, probe_every, scope):


    return (
        scope,
        method_name,
        int(env_seed),
        int(train_seed),
        repr(cfg),
        int(probe_every),
    )


# env

class DriftEnv:
    def __init__(
        self,
        sequence_seed=0,
        num_states=64,
        num_actions=8,
        feature_dim=24,
        drift_rate=0.1,
        reset_decoys_each_generation=False,
    ):
        self.rng = np.random.RandomState(sequence_seed)
        self.num_states = num_states
        self.num_actions = num_actions
        self.feature_dim = feature_dim
        self.drift_rate = drift_rate
        self.reset_decoys_each_generation = reset_decoys_each_generation

        ingredients = self.rng.normal(0, 1, size=(20, feature_dim)).astype(np.float32)
        self.features = np.zeros((num_states, feature_dim), dtype=np.float32)
        for s in range(num_states):
            pick = self.rng.choice(20, size=3, replace=False)
            v = ingredients[pick].mean(axis=0)
            self.features[s] = v / (np.linalg.norm(v) + 1e-8)

        self.targets = self.rng.randint(0, num_actions, size=num_states)
        self.decoys = np.full(num_states, -1, dtype=np.int64)
        self.core_mask = np.ones(num_states, dtype=bool)
        self.drift_mask = np.zeros(num_states, dtype=bool)
        self.reset_attempt()

    @property
    def obs_dim(self):
        return self.feature_dim

    def set_generation(self, generation_index):
        if generation_index == 0:
            return

        num_drift = max(1, int(self.num_states * self.drift_rate))
        drift_indices = self.rng.choice(self.num_states, size=num_drift, replace=False)

        self.core_mask[:] = True
        self.core_mask[drift_indices] = False
        self.drift_mask[:] = False
        self.drift_mask[drift_indices] = True

        if self.reset_decoys_each_generation:
            self.decoys[:] = -1

        for s in drift_indices:
            self.decoys[s] = self.targets[s]
            new_action = self.rng.randint(0, self.num_actions)
            while new_action == self.decoys[s]:
                new_action = self.rng.randint(0, self.num_actions)
            self.targets[s] = new_action

    def reset_attempt(self):
        self.current_state = self.rng.randint(0, self.num_states)

    def get_obs(self):
        return self.features[self.current_state].copy()

    def all_state_features(self):
        return self.features.copy()

    def step(self, action):
        target = self.targets[self.current_state]
        decoy = self.decoys[self.current_state]

        if action == target:
            reward = 1.0
        elif action == decoy and decoy >= 0:
            reward = 0.5
        else:
            reward = -0.1

        return reward, True, self.current_state, False


class VectorizedDriftEnv:
    def __init__(self, num_envs, sequence_seed=0, **env_kwargs):


        self.envs = [DriftEnv(sequence_seed, **env_kwargs) for _ in range(num_envs)]
        self.n = num_envs

    @property
    def obs_dim(self):
        return self.envs[0].obs_dim

    @property
    def num_actions(self):
        return self.envs[0].num_actions

    def set_generation(self, generation_index):
        for env in self.envs:
            env.set_generation(generation_index)

    def all_state_features(self):
        return self.envs[0].all_state_features()

    def get_obs(self):
        return np.stack([env.get_obs() for env in self.envs])

    def step(self, actions):
        rewards = np.zeros(self.n, dtype=np.float32)
        dones = np.zeros(self.n, dtype=bool)
        states = np.zeros(self.n, dtype=np.int64)

        for i, env in enumerate(self.envs):
            reward, done, state, _ = env.step(int(actions[i]))
            rewards[i] = reward
            dones[i] = done
            states[i] = state
            env.reset_attempt()

        return rewards, dones, states, np.zeros(self.n, dtype=bool)



# models

def init_uniform_batched(num_parallel, out_size, in_size, bound, seed):
    tensors = []
    for i in range(num_parallel):
        gen = torch.Generator()
        gen.manual_seed(int(seed * 1000003 + i * 1009 + out_size * 17 + in_size))
        tensor = torch.empty(out_size, in_size).uniform_(-bound, bound, generator=gen)
        tensors.append(tensor)
    return torch.stack(tensors)


class BatchedMLP(nn.Module):
    def __init__(self, sizes, num_parallel, base_seed):
        super().__init__()
        self.sizes = list(sizes)
        self.num_parallel = num_parallel
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for i in range(1, len(sizes)):
            bound = math.sqrt(6 / sizes[i - 1])
            weight = init_uniform_batched(
                num_parallel,
                sizes[i],
                sizes[i - 1],
                bound,
                base_seed + i,
            )
            self.weights.append(nn.Parameter(weight))
            self.biases.append(nn.Parameter(torch.zeros(num_parallel, sizes[i])))

    def forward(self, x):
        h = x
        for i in range(len(self.weights)):
            h = torch.einsum("nij,nj->ni", self.weights[i], h) + self.biases[i]
            if i < len(self.weights) - 1:
                h = F.relu(h)
        return h

    def reset_layers(self, layer_indices, base_seed):
        with torch.no_grad():
            for i in layer_indices:
                bound = math.sqrt(6 / self.sizes[i])
                weight = init_uniform_batched(
                    self.num_parallel,
                    self.sizes[i + 1],
                    self.sizes[i],
                    bound,
                    base_seed + i,
                )
                self.weights[i].copy_(weight.to(self.weights[i].device))
                self.biases[i].zero_()


class BatchedLateralNet(nn.Module):
    def __init__(
        self,
        sizes,
        num_parallel,
        base_seed,
        parents=None,
        use_decay=True,
        parent_output=True,
        lateral_init_scale=0.1,
    ):
        super().__init__()
        self.sizes = list(sizes)
        self.num_parallel = num_parallel
        self.use_decay = use_decay
        self.parent_output = parent_output
        self.lateral_init_scale = lateral_init_scale

        parents = [] if parents is None else parents
        self.num_parents = len(parents)

        for column_idx, (parent_weights, parent_biases) in enumerate(parents):
            for layer_idx, weight in enumerate(parent_weights):
                self.register_buffer(f"p{column_idx}W{layer_idx}", weight)
            for layer_idx, bias in enumerate(parent_biases):
                self.register_buffer(f"p{column_idx}B{layer_idx}", bias)

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.laterals = nn.ParameterDict()
        self.output_extras = nn.ParameterDict()

        length = len(sizes)
        for i in range(1, length):
            own_input_size = sizes[i - 1]
            bound = math.sqrt(6 / own_input_size)

            weight = init_uniform_batched(
                num_parallel,
                sizes[i],
                own_input_size,
                bound,
                base_seed + i * 11,
            )
            self.weights.append(nn.Parameter(weight))
            self.biases.append(nn.Parameter(torch.zeros(num_parallel, sizes[i])))

            if i >= 2 and self.num_parents > 0:
                for c in range(self.num_parents):
                    lat = init_uniform_batched(
                        num_parallel,
                        sizes[i],
                        sizes[i - 1],
                        bound * lateral_init_scale,
                        base_seed + i * 23 + c * 101,
                    )
                    self.laterals[f"l{i}C{c}"] = nn.Parameter(lat)

            if parent_output and i == length - 1 and self.num_parents > 0:
                for c in range(self.num_parents):
                    extra = init_uniform_batched(
                        num_parallel,
                        sizes[i],
                        sizes[-1],
                        bound * lateral_init_scale,
                        base_seed + i * 37 + c * 103,
                    )
                    self.output_extras[f"oeC{c}"] = nn.Parameter(extra)

    @property
    def has_parents(self):
        return self.num_parents > 0

    def column_forward(self, c, x):
        activations = [x]
        length = len(self.sizes) - 1

        for layer_idx in range(length):
            weight = getattr(self, f"p{c}W{layer_idx}")
            bias = getattr(self, f"p{c}B{layer_idx}")
            h = torch.einsum("nij,nj->ni", weight, activations[-1]) + bias
            if layer_idx < length - 1:
                h = F.relu(h)
            activations.append(h)

        return activations

    def forward(self, x, lateral_scale=1.0):
        if not self.use_decay:
            lateral_scale = 1.0

        parent_activations = [
            self.column_forward(c, x) for c in range(self.num_parents)
        ]

        h = x
        length = len(self.weights)

        for i in range(length):
            out = torch.einsum("nij,nj->ni", self.weights[i], h) + self.biases[i]

            if i + 1 >= 2 and self.num_parents > 0:
                for c in range(self.num_parents):
                    key = f"l{i + 1}C{c}"
                    if key in self.laterals:
                        out = out + lateral_scale * torch.einsum(
                            "nij,nj->ni",
                            self.laterals[key],
                            parent_activations[c][i],
                        )

            if self.parent_output and i == length - 1 and self.num_parents > 0:
                for c in range(self.num_parents):
                    key = f"oeC{c}"
                    if key in self.output_extras:
                        out = out + lateral_scale * torch.einsum(
                            "nij,nj->ni",
                            self.output_extras[key],
                            parent_activations[c][-1],
                        )

            if i < length - 1:
                out = F.relu(out)

            h = out

        return h

    def detach_active(self):
        return (
            [weight.detach().clone() for weight in self.weights],
            [bias.detach().clone() for bias in self.biases],
        )

    def export_student_only(self):
        """Create the actual deployment model with no parent or lateral state."""
        device = self.weights[0].device
        student = BatchedMLP(self.sizes, self.num_parallel, base_seed=0).to(device)
        with torch.no_grad():
            for target, source in zip(student.weights, self.weights):
                target.copy_(source)
            for target, source in zip(student.biases, self.biases):
                target.copy_(source)
        student.eval()
        for parameter in student.parameters():
            parameter.requires_grad_(False)
        return student


class PersistentPNN(nn.Module):
    def __init__(
        self,
        sizes,
        num_parallel,
        base_seed,
        parent_output=True,
        lateral_init_scale=0.1,
    ):
        super().__init__()
        self.sizes = list(sizes)
        self.num_parallel = num_parallel
        self.base_seed = base_seed
        self.parent_output = parent_output
        self.lateral_init_scale = lateral_init_scale
        self.columns = nn.ModuleList()
        self.add_column()

    def add_column(self):
        c = len(self.columns)
        column = nn.Module()
        column.weights = nn.ParameterList()
        column.biases = nn.ParameterList()
        column.laterals = nn.ParameterDict()
        column.gates = nn.ParameterDict()

        length = len(self.sizes)
        for i in range(1, length):
            bound = math.sqrt(6 / self.sizes[i - 1])
            weight = init_uniform_batched(
                self.num_parallel,
                self.sizes[i],
                self.sizes[i - 1],
                bound,
                self.base_seed + c * 7919 + i * 11,
            )
            column.weights.append(nn.Parameter(weight))
            column.biases.append(nn.Parameter(torch.zeros(self.num_parallel, self.sizes[i])))

            if i >= 2 and c > 0:
                for j in range(c):
                    lat = init_uniform_batched(
                        self.num_parallel,
                        self.sizes[i],
                        self.sizes[i - 1],
                        bound * self.lateral_init_scale,
                        self.base_seed + c * 131 + i * 23 + j * 101,
                    )
                    column.laterals[f"l{i}C{j}"] = nn.Parameter(lat)

            if self.parent_output and i == length - 1 and c > 0:
                for j in range(c):
                    extra = init_uniform_batched(
                        self.num_parallel,
                        self.sizes[-1],
                        self.sizes[-1],
                        bound * self.lateral_init_scale,
                        self.base_seed + c * 137 + j * 103,
                    )
                    column.laterals[f"oeC{j}"] = nn.Parameter(extra)

        if c > 0:
            for j in range(c):
                column.gates[f"gC{j}"] = nn.Parameter(
                    torch.full((self.num_parallel,), 1.0 / c)
                )

        self.columns.append(column)

    def forward(self, x):
        column_activations = []
        length = len(self.sizes) - 1

        for c, column in enumerate(self.columns):
            activations = [x]
            h = x

            for i in range(length):
                out = (
                    torch.einsum("nij,nj->ni", column.weights[i], h)
                    + column.biases[i]
                )

                if i + 1 >= 2 and c > 0:
                    for j in range(c):
                        key = f"l{i + 1}C{j}"
                        if key in column.laterals:
                            gate = column.gates[f"gC{j}"].unsqueeze(-1)
                            out = out + gate * torch.einsum(
                                "nij,nj->ni",
                                column.laterals[key],
                                column_activations[j][i],
                            )

                if self.parent_output and i == length - 1 and c > 0:
                    for j in range(c):
                        key = f"oeC{j}"
                        if key in column.laterals:
                            gate = column.gates[f"gC{j}"].unsqueeze(-1)
                            out = out + gate * torch.einsum(
                                "nij,nj->ni",
                                column.laterals[key],
                                column_activations[j][-1],
                            )

                if i < length - 1:
                    out = F.relu(out)

                activations.append(out)
                h = out

            column_activations.append(activations)

        return column_activations[-1][-1]

    def freeze_and_grow(self):
        for column in self.columns:
            for parameter in column.parameters():
                parameter.requires_grad_(False)
        self.add_column()

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]



# training

def sample_action(logits, temperature):
    log_probs = F.log_softmax(logits / temperature, dim=-1)
    probs = log_probs.exp()
    actions = torch.multinomial(probs, 1).squeeze(-1)
    chosen_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
    entropy = -(probs * log_probs).sum(-1).mean()
    return actions, chosen_log_probs, entropy


def normalize_per_slot(x, eps=1e-8):
    return (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + eps)


class RegularTrainer:
    name = "Regular"
    needs_generation = False

    def __init__(self, sizes, num_parallel, base_seed, cfg):
        self.sizes = list(sizes)
        self.num_parallel = num_parallel
        self.base_seed = base_seed
        self.cfg = cfg
        self.net = BatchedMLP(sizes, num_parallel, base_seed).to(DEVICE)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=cfg.lr)

    def on_generation(self, generation_index):
        pass

    def set_distill_states(self, features):
        pass

    def act(self, obs):
        return sample_action(self.net(obs), self.cfg.temperature)

    def update(self, log_prob_buffer, advantages, entropy_buffer=None):
        loss = -(torch.stack(log_prob_buffer, 0) * advantages).sum()
        if self.cfg.entropy_coef and entropy_buffer:
            loss = loss - self.cfg.entropy_coef * torch.stack(entropy_buffer).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ResetTrainer(RegularTrainer):
    name = "Reset"
    needs_generation = True

    def __init__(self, sizes, num_parallel, base_seed, cfg):
        super().__init__(sizes, num_parallel, base_seed, cfg)
        self.gen = 0

    def on_generation(self, generation_index):
        self.gen += 1
        self.net = BatchedMLP(
            self.sizes,
            self.num_parallel,
            self.base_seed + 10000 * self.gen,
        ).to(DEVICE)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.cfg.lr)


class WarmStartTrainer(RegularTrainer):
    name = "WarmStart"
    needs_generation = True

    def __init__(self, sizes, num_parallel, base_seed, cfg):
        super().__init__(sizes, num_parallel, base_seed, cfg)
        self.gen = 0

    def on_generation(self, generation_index):
        self.gen += 1
        self.net.reset_layers(
            [len(self.net.weights) - 1],
            self.base_seed + 20000 * self.gen,
        )
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.cfg.lr)


class DistilledTrainer(RegularTrainer):
    name = "Distilled"
    needs_generation = True

    def __init__(self, sizes, num_parallel, base_seed, cfg):
        super().__init__(sizes, num_parallel, base_seed, cfg)
        self.gen = 0
        self.distill_states = None

    def set_distill_states(self, features):
        self.distill_states = torch.from_numpy(features).to(DEVICE)

    def on_generation(self, generation_index):
        self.gen += 1
        state_matrix = self.distill_states
        batched = state_matrix.unsqueeze(0).expand(self.num_parallel, -1, -1)

        with torch.no_grad():
            teacher_logits = torch.stack(
                [self.net(batched[:, o, :]) for o in range(state_matrix.shape[0])],
                dim=1,
            )
            teacher_log_probs = F.log_softmax(
                teacher_logits / self.cfg.temperature,
                dim=-1,
            )
            teacher_probs = teacher_log_probs.exp()

        self.net = BatchedMLP(
            self.sizes,
            self.num_parallel,
            self.base_seed + 40000 * self.gen,
        ).to(DEVICE)

        distill_optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=self.cfg.distill_lr,
        )

        for _ in range(self.cfg.distill_steps):
            student_logits = torch.stack(
                [self.net(batched[:, o, :]) for o in range(state_matrix.shape[0])],
                dim=1,
            )
            student_log_probs = F.log_softmax(
                student_logits / self.cfg.temperature,
                dim=-1,
            )
            kl = (
                teacher_probs * (teacher_log_probs - student_log_probs)
            ).sum(-1).mean()

            distill_optimizer.zero_grad()
            kl.backward()
            distill_optimizer.step()

        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.cfg.lr)


class WeanNetTrainer(RegularTrainer):
    name = "WeanNet"
    needs_generation = True

    def __init__(self, sizes, num_parallel, base_seed, cfg):
        self.sizes = list(sizes)
        self.num_parallel = num_parallel
        self.base_seed = base_seed
        self.cfg = cfg

        self.net = BatchedLateralNet(
            sizes,
            num_parallel,
            base_seed,
            parents=None,
            use_decay=True,
            parent_output=cfg.parent_output,
            lateral_init_scale=cfg.lateral_init_scale,
        ).to(DEVICE)

        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=cfg.lr)
        self.gen = 0
        self.batches_since_rebirth = 0

    def on_generation(self, generation_index):
        self.gen += 1
        parent = self.net.detach_active()

        self.net = BatchedLateralNet(
            self.sizes,
            self.num_parallel,
            self.base_seed + 30000 * self.gen,
            parents=[parent],
            use_decay=True,
            parent_output=self.cfg.parent_output,
            lateral_init_scale=self.cfg.lateral_init_scale,
        ).to(DEVICE)

        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.cfg.lr)
        self.batches_since_rebirth = 0

    def scale(self):
        if not self.net.has_parents:
            return 0.0

        progress = min(
            self.batches_since_rebirth / max(self.cfg.decay_batches, 1),
            1.0,
        )
        return self.cfg.lambda_min + (
            self.cfg.lambda_init - self.cfg.lambda_min
        ) * (1.0 - progress)

    def act(self, obs):
        return sample_action(
            self.net(obs, lateral_scale=self.scale()),
            self.cfg.temperature,
        )

    def update(self, log_prob_buffer, advantages, entropy_buffer=None):
        loss = -(torch.stack(log_prob_buffer, 0) * advantages).sum()
        if self.cfg.entropy_coef and entropy_buffer:
            loss = loss - self.cfg.entropy_coef * torch.stack(entropy_buffer).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.net.has_parents:
            self.batches_since_rebirth += 1


class PNNResetTrainer(WeanNetTrainer):
    name = "PNN-Reset"

    def __init__(self, sizes, num_parallel, base_seed, cfg):
        self.sizes = list(sizes)
        self.num_parallel = num_parallel
        self.base_seed = base_seed
        self.cfg = cfg

        self.net = BatchedLateralNet(
            sizes,
            num_parallel,
            base_seed,
            parents=None,
            use_decay=False,
            parent_output=cfg.parent_output,
            lateral_init_scale=cfg.lateral_init_scale,
        ).to(DEVICE)

        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=cfg.lr)
        self.gen = 0
        self.batches_since_rebirth = 0

    def on_generation(self, generation_index):
        self.gen += 1
        parent = self.net.detach_active()

        self.net = BatchedLateralNet(
            self.sizes,
            self.num_parallel,
            self.base_seed + 31000 * self.gen,
            parents=[parent],
            use_decay=False,
            parent_output=self.cfg.parent_output,
            lateral_init_scale=self.cfg.lateral_init_scale,
        ).to(DEVICE)

        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.cfg.lr)
        self.batches_since_rebirth = 0

    def scale(self):
        return 1.0 if self.net.has_parents else 0.0


class PNNTrainer(RegularTrainer):
    name = "PNN"
    needs_generation = True

    def __init__(self, sizes, num_parallel, base_seed, cfg):
        self.sizes = list(sizes)
        self.num_parallel = num_parallel
        self.base_seed = base_seed
        self.cfg = cfg

        self.net = PersistentPNN(
            sizes,
            num_parallel,
            base_seed,
            parent_output=cfg.parent_output,
            lateral_init_scale=cfg.pnn_lateral_init_scale,
        ).to(DEVICE)

        self.optimizer = torch.optim.SGD(
            self.net.trainable_parameters(),
            lr=cfg.pnn_lr,
        )
        self.gen = 0

    def on_generation(self, generation_index):
        self.gen += 1
        self.net.freeze_and_grow()


        self.net.to(DEVICE)
        self.optimizer = torch.optim.SGD(
            self.net.trainable_parameters(),
            lr=self.cfg.pnn_lr,
        )


TRAINER_CLASSES = {
    cls.name: cls
    for cls in [
        RegularTrainer,
        ResetTrainer,
        WarmStartTrainer,
        DistilledTrainer,
        WeanNetTrainer,
        PNNResetTrainer,
        PNNTrainer,
    ]
}


# eval

def count_parameters(net):
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    resident = sum(p.numel() for p in net.parameters()) + sum(
        b.numel() for b in net.buffers()
    )
    return trainable, resident


def probe_network(
    trainer,
    env,
    lateral_scale_override=None,
    network_override=None,
):
    e0 = env.envs[0]
    features = torch.from_numpy(e0.all_state_features()).to(DEVICE)
    xb = features.unsqueeze(0).expand(trainer.num_parallel, -1, -1)
    network = trainer.net if network_override is None else network_override

    scale = trainer.scale() if hasattr(trainer, "scale") else None
    if lateral_scale_override is not None:
        scale = lateral_scale_override

    with torch.no_grad():
        if isinstance(network, BatchedLateralNet):
            logits = torch.stack(
                [
                    network(xb[:, state, :], lateral_scale=scale)
                    for state in range(e0.num_states)
                ],
                dim=1,
            )
        else:
            logits = torch.stack(
                [network(xb[:, state, :]) for state in range(e0.num_states)],
                dim=1,
            )

        probs = F.softmax(logits / trainer.cfg.temperature, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(-1)

    predictions = logits.argmax(-1).cpu().numpy()
    entropy = entropy.cpu().numpy()

    targets = e0.targets
    decoys = e0.decoys
    core_mask = e0.core_mask
    drift_mask = e0.drift_mask
    has_drift = bool(drift_mask.any())

    acc = {key: [] for key in METRIC_KEYS}

    for p in range(trainer.num_parallel):
        pred = predictions[p]
        rewards = np.full(len(targets), -0.1, dtype=np.float32)
        rewards[pred == targets] = 1.0
        decoy_hit = (pred == decoys) & (decoys >= 0)
        rewards[decoy_hit] = 0.5

        acc["coreAcc"].append(
            (pred[core_mask] == targets[core_mask]).mean()
            if core_mask.any()
            else np.nan
        )
        acc["coreTrap"].append(
            ((pred[core_mask] == decoys[core_mask]) & (decoys[core_mask] >= 0)).mean()
            if core_mask.any()
            else np.nan
        )
        acc["rewCore"].append(
            rewards[core_mask].mean() if core_mask.any() else np.nan
        )
        acc["rewOverall"].append(rewards.mean())
        acc["entAll"].append(entropy[p].mean())

        if has_drift:
            acc["driftAcc"].append((pred[drift_mask] == targets[drift_mask]).mean())
            acc["driftTrap"].append(
                ((pred[drift_mask] == decoys[drift_mask]) & (decoys[drift_mask] >= 0)).mean()
            )
            acc["rewDrift"].append(rewards[drift_mask].mean())
            acc["entDrift"].append(entropy[p][drift_mask].mean())
        else:
            acc["driftAcc"].append(np.nan)
            acc["driftTrap"].append(np.nan)
            acc["rewDrift"].append(np.nan)
            acc["entDrift"].append(np.nan)

    return {key: float(np.nanmean(values)) for key, values in acc.items()}



# experiment loop

def make_env_kwargs(cfg):
    return dict(
        num_states=cfg.num_states,
        num_actions=cfg.num_actions,
        feature_dim=cfg.feature_dim,
        drift_rate=cfg.drift_rate,
        reset_decoys_each_generation=cfg.reset_decoys_each_generation,
    )


def run_method(method_name, env_seed, train_seed, cfg, probe_every=0):
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)

    env = VectorizedDriftEnv(
        cfg.num_parallel,
        sequence_seed=env_seed,
        **make_env_kwargs(cfg),
    )

    sizes = [env.obs_dim, *cfg.hidden_sizes, env.num_actions]
    trainer = TRAINER_CLASSES[method_name](
        sizes,
        cfg.num_parallel,
        train_seed,
        cfg,
    )
    trainer.set_distill_states(env.all_state_features())

    value_table = np.zeros((cfg.num_parallel, cfg.num_states), dtype=np.float32)
    generation_stats = defaultdict(list)
    trainable_params = []
    resident_params = []
    deployed_trainable_params = []
    deployed_resident_params = []
    curve = []

    obs = torch.from_numpy(env.get_obs()).to(DEVICE)
    global_batch = 0

    for generation in range(cfg.generations):
        env.set_generation(generation)

        if generation > 0 and trainer.needs_generation:
            trainer.on_generation(generation)

        for batch in range(cfg.generation_length):
            reward_buffer = np.zeros(
                (cfg.batch_size, cfg.num_parallel),
                dtype=np.float32,
            )
            state_buffer = np.zeros(
                (cfg.batch_size, cfg.num_parallel),
                dtype=np.int64,
            )
            log_prob_buffer = []
            entropy_buffer = []

            for step in range(cfg.batch_size):
                actions, log_prob, entropy = trainer.act(obs)
                rewards, _, state_indices, _ = env.step(actions.cpu().numpy())

                reward_buffer[step] = rewards
                state_buffer[step] = state_indices
                log_prob_buffer.append(log_prob)
                entropy_buffer.append(entropy)
                obs = torch.from_numpy(env.get_obs()).to(DEVICE)

            if cfg.baseline_mode == "stateEma":
                centered = np.zeros_like(reward_buffer)
                for step in range(cfg.batch_size):
                    for p in range(cfg.num_parallel):
                        state = state_buffer[step, p]
                        centered[step, p] = (
                            reward_buffer[step, p] - value_table[p, state]
                        )
                        value_table[p, state] = (
                            (1 - cfg.ema_beta) * value_table[p, state]
                            + cfg.ema_beta * reward_buffer[step, p]
                        )

                advantages_np = centered / (
                    centered.std(0, keepdims=True) + 1e-8
                )
            else:
                advantages_np = normalize_per_slot(reward_buffer)

            advantages = torch.from_numpy(advantages_np).to(DEVICE)
            trainer.update(log_prob_buffer, advantages, entropy_buffer)
            global_batch += 1

            if probe_every and batch % probe_every == 0:
                probe = probe_network(trainer, env)
                probe["globalBatch"] = global_batch
                curve.append(probe)

        probe = probe_network(trainer, env)
        for key, value in probe.items():
            generation_stats[key].append(value)
        generation_stats["lambdaAtProbe"].append(
            trainer.scale() if hasattr(trainer, "scale") else np.nan
        )




        if isinstance(trainer, WeanNetTrainer) and not isinstance(trainer, PNNResetTrainer):
            deployed = trainer.net.export_student_only()
            pruned = probe_network(
                trainer,
                env,
                network_override=deployed,
            )
            for key in ["coreAcc", "driftAcc", "rewOverall"]:
                generation_stats[f"pruned_{key}"].append(pruned[key])
            deployed_trainable, deployed_resident = count_parameters(deployed)
        else:
            for key in ["coreAcc", "driftAcc", "rewOverall"]:
                generation_stats[f"pruned_{key}"].append(np.nan)
            deployed_trainable, deployed_resident = np.nan, np.nan

        trainable, resident = count_parameters(trainer.net)
        trainable_params.append(trainable)
        resident_params.append(resident)
        deployed_trainable_params.append(deployed_trainable)
        deployed_resident_params.append(deployed_resident)

    out = {key: np.asarray(values) for key, values in generation_stats.items()}
    out["paramsTrain"] = np.asarray(trainable_params)
    out["paramsRes"] = np.asarray(resident_params)
    out["paramsDeployTrain"] = np.asarray(deployed_trainable_params)
    out["paramsDeployRes"] = np.asarray(deployed_resident_params)
    out["curve"] = curve
    return out


def run_method_worker(method_name, env_seed, train_seed, cfg, probe_every):
    """Run one independent method/seed job inside a spawned worker process.

    Workers never read or write the shared checkpoint. Only the parent process
    performs checkpoint I/O, preventing concurrent pickle writes.
    """


    torch.set_num_threads(1)

    if DEVICE.type == "cuda":
        torch.cuda.set_device(0)

    result = run_method(
        method_name,
        env_seed,
        train_seed,
        cfg,
        probe_every=probe_every,
    )



    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return result


def collect_runs(
    method_names,
    cfg,
    env_seeds=None,
    progress_label="",
):
    """Collect method/seed runs, executing missing checkpoint entries in parallel.

    Existing checkpoint entries are reused exactly as before. Missing entries
    are submitted to independent spawned processes. The parent process alone
    writes completed results back to the original checkpoint after each job.
    """
    if env_seeds is None:
        env_seeds = EVALUATION_ENV_SEEDS



    results_by_method = {
        method_name: [None] * len(env_seeds)
        for method_name in method_names
    }

    jobs = []

    for method_name in method_names:
        for index, env_seed in enumerate(env_seeds):
            train_seed = SEED_BASE + env_seed * 997
            task_key = run_checkpoint_key(
                method_name,
                env_seed,
                train_seed,
                cfg,
                probe_every=0,
                scope=progress_label,
            )

            result = (
                RUN_CHECKPOINT["completed_runs"].get(task_key)
                if RUN_CHECKPOINT is not None
                else None
            )

            if result is not None:
                print(
                    f"{progress_label}{method_name:11s} | "
                    f"seed {index + 1}/{len(env_seeds)} | env {env_seed} (resumed)"
                )
                results_by_method[method_name][index] = result
            else:
                jobs.append(
                    (
                        method_name,
                        index,
                        env_seed,
                        train_seed,
                        task_key,
                    )
                )

    if jobs:
        worker_count = max(1, min(int(GPU_WORKERS), len(jobs)))
        print(
            f"{progress_label}Launching {len(jobs)} missing run(s) with "
            f"{worker_count} parallel GPU worker(s)."
        )



        mp_context = multiprocessing.get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp_context,
        ) as executor:
            pending = {}

            for method_name, index, env_seed, train_seed, task_key in jobs:
                print(
                    f"{progress_label}{method_name:11s} | "
                    f"seed {index + 1}/{len(env_seeds)} | env {env_seed} (queued)"
                )

                future = executor.submit(
                    run_method_worker,
                    method_name,
                    env_seed,
                    train_seed,
                    cfg,
                    0,
                )

                pending[future] = (
                    method_name,
                    index,
                    env_seed,
                    task_key,
                )

            for future in as_completed(pending):
                method_name, index, env_seed, task_key = pending[future]



                result = future.result()
                results_by_method[method_name][index] = result

                if RUN_CHECKPOINT is not None:
                    RUN_CHECKPOINT["completed_runs"][task_key] = result
                    save_checkpoint(RUN_CHECKPOINT)

                print(
                    f"{progress_label}{method_name:11s} | "
                    f"seed {index + 1}/{len(env_seeds)} | env {env_seed} DONE"
                )

    results = {}

    for method_name in method_names:
        runs = defaultdict(list)

        for result in results_by_method[method_name]:
            if result is None:
                raise RuntimeError(f"Missing result for method {method_name}")

            for key, value in result.items():
                if key == "curve":
                    continue
                runs[key].append(value)

        results[method_name] = {
            key: np.stack(values)
            for key, values in runs.items()
            if len(values) > 0
        }

    return results

def final_mean_sd(result_for_method, metric):
    values = result_for_method[metric][:, -1]
    return float(np.nanmean(values)), float(np.nanstd(values, ddof=1))


def write_csv(filename, rows, fieldnames=None):
    path = os.path.join(OUTPUT_DIR, filename)
    if not rows:
        return path

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{filename}.",
        suffix=".tmp",
        dir=OUTPUT_DIR,
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    return path


def save_figure(fig, stem):
    pdf_path = os.path.join(OUTPUT_DIR, f"{stem}.pdf")
    png_path = os.path.join(OUTPUT_DIR, f"{stem}.png")
    for path, kwargs in [
        (pdf_path, {"bbox_inches": "tight"}),
        (png_path, {"bbox_inches": "tight", "dpi": 300}),
    ]:
        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{stem}.",
            suffix=os.path.splitext(path)[1],
            dir=OUTPUT_DIR,
        )
        os.close(fd)
        try:
            fig.savefig(temporary_path, **kwargs)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
    plt.close(fig)
    return pdf_path, png_path



# entropy sweep

def run_main_entropy_sweep(base_cfg):
    entropy_values = [0.0, 0.05, 0.1, 0.2]
    all_results = {}
    rows = []

    print("\n=== REVISED MAIN ENTROPY SWEEP ===")

    for entropy in entropy_values:
        cfg = replace(base_cfg, entropy_coef=entropy)
        label = f"{entropy:g}"
        print(f"\n--- entropy = {label} ---")

        results = collect_runs(
            METHOD_ORDER,
            cfg,
            progress_label=f"e={label} | ",
        )
        all_results[label] = results

        for method in METHOD_ORDER:
            for metric in ["coreAcc", "driftAcc", "driftTrap", "rewOverall"]:
                mean, sd = final_mean_sd(results[method], metric)
                rows.append(
                    {
                        "entropy": entropy,
                        "method": method,
                        "metric": metric,
                        "mean": mean,
                        "sd": sd,
                    }
                )

    write_csv("main_entropy_summary.csv", rows)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plot_specs = [
        ("driftAcc", "Drift-found accuracy", (0, 1)),
        ("driftTrap", "Drift-trap rate", (0, 1)),
        ("coreAcc", "Core-state accuracy", (0, 1)),
        ("rewOverall", "Overall reward", None),
    ]

    labels = [f"{value:g}" for value in entropy_values]

    for axis, (metric, title, ylim) in zip(axes.flat, plot_specs):
        for method in METHOD_ORDER:
            means = []
            sds = []
            for label in labels:
                mean, sd = final_mean_sd(all_results[label][method], metric)
                means.append(mean)
                sds.append(sd)

            axis.errorbar(
                entropy_values,
                means,
                yerr=sds,
                marker="o",
                linewidth=1.8,
                capsize=2,
                label=method,
            )

        axis.set_title(title)
        axis.set_xlabel("Entropy coefficient")
        axis.grid(alpha=0.25)
        if ylim is not None:
            axis.set_ylim(*ylim)

    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("Concept-drift benchmark across entropy settings")
    fig.tight_layout()
    save_figure(fig, "main_entropy_metrics")

    zero_results = all_results["0"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    generations = np.arange(1, base_cfg.generations + 1)

    for method in METHOD_ORDER:
        arr = zero_results[method]["coreAcc"]
        mean = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0, ddof=1)
        line = ax.plot(generations, mean, marker="o", linewidth=1.8, label=method)[0]
        ax.fill_between(
            generations,
            np.maximum(0, mean - sd),
            np.minimum(1, mean + sd),
            alpha=0.10,
            color=line.get_color(),
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Core-state accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Core retention across generations (entropy = 0)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    save_figure(fig, "main_core_retention")



# pnn tuning

def run_pnn_tuning(base_cfg, filename="pnn_tuning.csv"):
    print("\n=== PNN TUNING SCREEN (HELD-OUT ENVIRONMENT SEEDS) ===")

    tuning_seeds = PNN_TUNING_ENV_SEEDS
    learning_rates = [0.005, 0.02, 0.05]
    lateral_scales = [0.1, 1.0]
    rows = []

    best = None
    best_score = -float("inf")

    for learning_rate in learning_rates:
        for lateral_scale in lateral_scales:
            cfg = replace(
                base_cfg,
                entropy_coef=0.0,
                pnn_lr=learning_rate,
                pnn_lateral_init_scale=lateral_scale,
            )

            print(
                f"\nPNN lr={learning_rate:g}, "
                f"lateral_init={lateral_scale:g}"
            )

            result = collect_runs(
                ["PNN"],
                cfg,
                env_seeds=tuning_seeds,
                progress_label="tune | ",
            )["PNN"]

            reward_mean, reward_sd = final_mean_sd(result, "rewOverall")
            core_mean, core_sd = final_mean_sd(result, "coreAcc")
            drift_mean, drift_sd = final_mean_sd(result, "driftAcc")

            row = {
                "pnn_lr": learning_rate,
                "pnn_lateral_init_scale": lateral_scale,
                "reward_mean": reward_mean,
                "reward_sd": reward_sd,
                "core_mean": core_mean,
                "core_sd": core_sd,
                "drift_mean": drift_mean,
                "drift_sd": drift_sd,
                "validation_seed_count": len(tuning_seeds),
                "validation_env_seeds": ",".join(map(str, tuning_seeds)),
                "selection_metric": "final_generation_overall_reward",
            }
            rows.append(row)

            if reward_mean > best_score:
                best_score = reward_mean
                best = (learning_rate, lateral_scale)

    write_csv(filename, rows)

    print(
        f"\nSelected PNN by final-generation overall reward: "
        f"lr={best[0]:g}, lateral_init={best[1]:g}"
    )

    return best



# lambda floor

def run_lambda_ablation(base_cfg):
    print("\n=== WEANNET LAMBDA_MIN ABLATION ===")

    lambda_values = [0.0, 0.05, 0.10]
    rows = []
    removal_rows = []
    deployment_size_rows = []

    for lambda_min in lambda_values:
        cfg = replace(
            base_cfg,
            entropy_coef=0.0,
            lambda_min=lambda_min,
            decay_batches=300,
        )

        print(f"\n--- lambda_min = {lambda_min:g} ---")
        result = collect_runs(
            ["WeanNet"],
            cfg,
            progress_label=f"lambda={lambda_min:g} | ",
        )["WeanNet"]

        row = {"lambda_min": lambda_min}
        for metric in ["coreAcc", "driftAcc", "rewOverall"]:
            mean, sd = final_mean_sd(result, metric)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
        rows.append(row)




        for metric in ["coreAcc", "driftAcc", "rewOverall"]:
            attached = result[metric][:, -1]
            pruned = result[f"pruned_{metric}"][:, -1]
            delta = pruned - attached

            removal_rows.append(
                {
                    "lambda_min": lambda_min,
                    "metric": metric,
                    "attached_mean": float(np.nanmean(attached)),
                    "attached_sd": float(np.nanstd(attached, ddof=1)),
                    "pruned_mean": float(np.nanmean(pruned)),
                    "pruned_sd": float(np.nanstd(pruned, ddof=1)),
                    "pruned_minus_attached_mean": float(np.nanmean(delta)),
                    "pruned_minus_attached_sd": float(np.nanstd(delta, ddof=1)),
                }
            )

        attached_params = result["paramsRes"][:, -1]
        deployed_params = result["paramsDeployRes"][:, -1]
        reduction_pct = 100.0 * (attached_params - deployed_params) / attached_params
        deployment_size_rows.append(
            {
                "lambda_min": lambda_min,
                "final_lambda_mean": float(np.nanmean(result["lambdaAtProbe"][:, -1])),
                "attached_resident_params_mean": float(np.nanmean(attached_params)),
                "attached_resident_params_sd": float(np.nanstd(attached_params, ddof=1)),
                "deployed_resident_params_mean": float(np.nanmean(deployed_params)),
                "deployed_resident_params_sd": float(np.nanstd(deployed_params, ddof=1)),
                "parameter_reduction_pct_mean": float(np.nanmean(reduction_pct)),
                "parameter_reduction_pct_sd": float(np.nanstd(reduction_pct, ddof=1)),
            }
        )

    write_csv("ablation_lambda_min.csv", rows)
    write_csv("ablation_parent_removal.csv", removal_rows)
    write_csv("ablation_deployment_size.csv", deployment_size_rows)
    return rows



# initial parent gain

def run_lambda_init_ablation(base_cfg):
    print("\n=== WEANNET LAMBDA_INIT ABLATION ===")

    lambda_init_values = [0.25, 0.5, 1.0]
    rows = []
    curve_rows = []

    for lambda_init in lambda_init_values:
        cfg = replace(
            base_cfg,
            entropy_coef=0.0,
            lambda_init=lambda_init,
            lambda_min=0.0,
            decay_batches=300,
        )

        print(f"\n--- lambda_init = {lambda_init:g} ---")
        result = collect_runs(
            ["WeanNet"],
            cfg,
            progress_label=f"lambda_init={lambda_init:g} | ",
        )["WeanNet"]

        row = {"lambda_init": lambda_init}
        for metric in ["coreAcc", "driftAcc", "rewOverall"]:
            mean, sd = final_mean_sd(result, metric)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
        rows.append(row)



        for generation in range(cfg.generations):
            for metric in ["coreAcc", "driftAcc", "rewOverall"]:
                values = result[metric][:, generation]
                curve_rows.append(
                    {
                        "lambda_init": lambda_init,
                        "generation": generation + 1,
                        "metric": metric,
                        "mean": float(np.nanmean(values)),
                        "sd": float(np.nanstd(values, ddof=1)),
                    }
                )

    write_csv("ablation_lambda_init.csv", rows)
    write_csv("ablation_lambda_init_by_generation.csv", curve_rows)
    return rows



# decay

def run_decay_ablation(base_cfg):
    print("\n=== WEANNET DECAY-HORIZON ABLATION ===")

    decay_values = [150, 300, 600]
    rows = []

    for decay_batches in decay_values:
        cfg = replace(
            base_cfg,
            entropy_coef=0.0,
            lambda_init=1.0,
            lambda_min=0.0,
            decay_batches=decay_batches,
        )

        print(f"\n--- decay_batches = {decay_batches} ---")
        result = collect_runs(
            ["WeanNet"],
            cfg,
            progress_label=f"decay={decay_batches} | ",
        )["WeanNet"]

        row = {"decay_batches": decay_batches}
        for metric in ["coreAcc", "driftAcc", "rewOverall"]:
            mean, sd = final_mean_sd(result, metric)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
        rows.append(row)

    write_csv("ablation_decay_horizon.csv", rows)
    return rows


def plot_schedule_ablations(lambda_min_rows, lambda_init_rows, decay_rows):
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    panels = [
        (
            axes[0],
            lambda_min_rows,
            "lambda_min",
            r"Final parent gain $\lambda_{min}$",
            "Parent-gain floor",
        ),
        (
            axes[1],
            lambda_init_rows,
            "lambda_init",
            r"Initial parent gain $\lambda_{init}$",
            "Initial parent gain",
        ),
        (
            axes[2],
            decay_rows,
            "decay_batches",
            "Decay horizon (batches)",
            "Decay horizon",
        ),
    ]

    for axis, rows, x_key, x_label, title in panels:
        for metric, label in [
            ("coreAcc", "Core accuracy"),
            ("driftAcc", "Drift-found accuracy"),
        ]:
            x = [row[x_key] for row in rows]
            y = [row[f"{metric}_mean"] for row in rows]
            err = [row[f"{metric}_sd"] for row in rows]
            axis.errorbar(
                x,
                y,
                yerr=err,
                marker="o",
                linewidth=1.8,
                capsize=3,
                label=label,
            )
        axis.set_xlabel(x_label)
        axis.set_ylabel("Accuracy")
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    fig.suptitle("WeanNet retention-adaptation schedule ablations")
    fig.tight_layout()
    save_figure(fig, "ablation_schedule")



# architecture

def run_architecture_ablation(base_cfg):
    print("\n=== ARCHITECTURE ABLATION ===")

    architectures = {
        "32-32": (32, 32),
        "64-64": (64, 64),
        "64-64-64": (64, 64, 64),
    }
    methods = ["Reset", "PNN-Reset", "WeanNet", "PNN"]
    rows = []

    for architecture_name, hidden_sizes in architectures.items():
        cfg = replace(
            base_cfg,
            entropy_coef=0.0,
            hidden_sizes=hidden_sizes,
            lambda_init=1.0,
            lambda_min=0.0,
            decay_batches=300,
        )





        if tuple(hidden_sizes) != tuple(base_cfg.hidden_sizes):
            best_lr, best_lateral = run_pnn_tuning(
                cfg,
                filename=(
                    "pnn_tuning_arch_"
                    + architecture_name.replace("-", "_")
                    + ".csv"
                ),
            )
            cfg = replace(
                cfg,
                pnn_lr=best_lr,
                pnn_lateral_init_scale=best_lateral,
            )

        print(f"\n--- architecture = {architecture_name} ---")
        results = collect_runs(
            methods,
            cfg,
            progress_label=f"arch={architecture_name} | ",
        )

        for method in methods:
            row = {
                "architecture": architecture_name,
                "method": method,
                "pnn_lr": cfg.pnn_lr if method == "PNN" else np.nan,
                "pnn_lateral_init_scale": (
                    cfg.pnn_lateral_init_scale if method == "PNN" else np.nan
                ),
            }
            for metric in ["coreAcc", "driftAcc", "rewOverall"]:
                mean, sd = final_mean_sd(results[method], metric)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_sd"] = sd
            rows.append(row)

    write_csv("ablation_architecture.csv", rows)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x = np.arange(len(architectures))
    labels = list(architectures.keys())

    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        means = [row["rewOverall_mean"] for row in selected]
        sds = [row["rewOverall_sd"] for row in selected]
        ax.errorbar(x, means, yerr=sds, marker="o", linewidth=1.8, capsize=3, label=method)

    ax.set_xticks(x, labels)
    ax.set_xlabel("Hidden-layer architecture")
    ax.set_ylabel("Final overall reward")
    ax.set_title("Architecture robustness")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "ablation_architecture")

    return rows



# parent output

def run_parent_output_ablation(base_cfg):
    print("\n=== PARENT-OUTPUT PATHWAY ABLATION ===")

    methods = ["WeanNet", "PNN-Reset", "PNN"]
    rows = []

    for parent_output in [False, True]:
        cfg = replace(
            base_cfg,
            entropy_coef=0.0,
            parent_output=parent_output,
            lambda_init=1.0,
            lambda_min=0.0,
            decay_batches=300,
        )

        state = "ON" if parent_output else "OFF"



        if parent_output != base_cfg.parent_output:
            best_lr, best_lateral = run_pnn_tuning(
                cfg,
                filename=f"pnn_tuning_parent_output_{state.lower()}.csv",
            )
            cfg = replace(
                cfg,
                pnn_lr=best_lr,
                pnn_lateral_init_scale=best_lateral,
            )

        print(f"\n--- parent output {state} ---")
        results = collect_runs(
            methods,
            cfg,
            progress_label=f"output={state} | ",
        )

        for method in methods:
            row = {
                "method": method,
                "parent_output": state,
                "pnn_lr": cfg.pnn_lr if method == "PNN" else np.nan,
                "pnn_lateral_init_scale": (
                    cfg.pnn_lateral_init_scale if method == "PNN" else np.nan
                ),
            }
            for metric in ["coreAcc", "driftAcc", "rewOverall"]:
                mean, sd = final_mean_sd(results[method], metric)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_sd"] = sd
            rows.append(row)

    write_csv("ablation_parent_output.csv", rows)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(methods))
    width = 0.34

    off = [next(row for row in rows if row["method"] == m and row["parent_output"] == "OFF") for m in methods]
    on = [next(row for row in rows if row["method"] == m and row["parent_output"] == "ON") for m in methods]

    ax.bar(
        x - width / 2,
        [row["rewOverall_mean"] for row in off],
        width,
        yerr=[row["rewOverall_sd"] for row in off],
        capsize=3,
        label="Output OFF",
    )
    ax.bar(
        x + width / 2,
        [row["rewOverall_mean"] for row in on],
        width,
        yerr=[row["rewOverall_sd"] for row in on],
        capsize=3,
        label="Output ON",
    )

    ax.set_xticks(x, methods)
    ax.set_ylabel("Final overall reward")
    ax.set_title("Parent-output pathway ablation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "ablation_parent_output")

    return rows



# run

VALID_RUN_MODES = {
    "all",
    "ablations",
    "main",
    "lambda",
    "lambda_init",
    "decay",
    "architecture",
    "parent_output",
    "pnn_tuning",
}


def mode_in(*names):
    return RUN_MODE in names or RUN_MODE == "all"


def save_pnn_selection(learning_rate, lateral_init_scale):
    path = os.path.join(OUTPUT_DIR, "selected_pnn_config.txt")
    with open(path, "w", encoding="utf-8") as file:
        file.write(f"pnn_lr={learning_rate}\n")
        file.write(f"pnn_lateral_init_scale={lateral_init_scale}\n")
        file.write("selection_metric=final_generation_overall_reward\n")
        file.write(
            "validation_env_seeds="
            + ",".join(map(str, PNN_TUNING_ENV_SEEDS))
            + "\n"
        )
        file.write(
            "evaluation_env_seeds="
            + ",".join(map(str, EVALUATION_ENV_SEEDS))
            + "\n"
        )


def run_mode_needs_pnn_selection():
    return mode_in(
        "ablations",
        "main",
        "architecture",
        "parent_output",
        "pnn_tuning",
    )


def main():
    global RUN_CHECKPOINT
    start = time.time()

    if RUN_MODE not in VALID_RUN_MODES:
        raise ValueError(
            f"Unsupported RUN_MODE={RUN_MODE!r}. "
            f"Choose one of {sorted(VALID_RUN_MODES)}."
        )
    if DEFAULT.lambda_min != 0.0:
        raise ValueError(
            "The revised default must end at lambda_min=0 for parent-free deployment."
        )
    if DEFAULT.decay_batches > DEFAULT.generation_length:
        raise ValueError(
            "The default decay horizon must finish within one generation."
        )

    RUN_CHECKPOINT, resumed = load_or_create_checkpoint()

    print(f"Device: {DEVICE}")
    print(f"Run mode: {RUN_MODE}")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Default hidden sizes: {DEFAULT.hidden_sizes}")
    print(f"Parallel GPU workers: {GPU_WORKERS}")
    print(
        f"Default WeanNet lambda: {DEFAULT.lambda_init:g} -> "
        f"{DEFAULT.lambda_min:g} over {DEFAULT.decay_batches} batches"
    )
    if resumed:
        print(
            "Resuming from checkpoint: "
            f"{len(RUN_CHECKPOINT['completed_runs'])} completed method/seed runs saved."
        )

    try:


        tuned_cfg = DEFAULT
        if run_mode_needs_pnn_selection():
            best_lr, best_lateral = run_pnn_tuning(DEFAULT)
            tuned_cfg = replace(
                DEFAULT,
                pnn_lr=best_lr,
                pnn_lateral_init_scale=best_lateral,
            )
            save_pnn_selection(best_lr, best_lateral)

        if mode_in("main"):
            run_main_entropy_sweep(tuned_cfg)

        lambda_rows = None
        lambda_init_rows = None
        decay_rows = None

        if mode_in("ablations", "lambda"):
            lambda_rows = run_lambda_ablation(tuned_cfg)

        if mode_in("ablations", "lambda_init"):
            lambda_init_rows = run_lambda_init_ablation(tuned_cfg)

        if mode_in("ablations", "decay"):
            decay_rows = run_decay_ablation(tuned_cfg)

        if (
            lambda_rows is not None
            and lambda_init_rows is not None
            and decay_rows is not None
        ):
            plot_schedule_ablations(lambda_rows, lambda_init_rows, decay_rows)

        if mode_in("ablations", "architecture"):
            run_architecture_ablation(tuned_cfg)

        if mode_in("ablations", "parent_output"):
            run_parent_output_ablation(tuned_cfg)

        RUN_CHECKPOINT["status"] = "complete"
        save_checkpoint(RUN_CHECKPOINT)
        elapsed = time.time() - start
        print(f"\nFinished in {elapsed / 60:.1f} minutes.")
        print(f"Results saved to: {os.path.abspath(OUTPUT_DIR)}")
        print(f"Checkpoint: {os.path.abspath(CHECKPOINT_FILE)}")
    except BaseException:


        print(f"\nStopped. Resume later from: {os.path.abspath(CHECKPOINT_FILE)}")
        raise


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

