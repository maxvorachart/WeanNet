"""Concept-drift experiment comparing WeanNet with transfer-learning baselines."""

import math
import time
import warnings
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='Degrees of freedom')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lr = 0.02
temperature = 1.0

lateral_init_scale = 0.1
lateral_scale_init = 1.0
lateral_scale_min = 0.05
lateral_decay_batches = 120

distill_steps = 150
distill_lr = 0.05

entropy_coef = 0.0
ema_beta = 0.1

# Environment

class DriftEnv:

    def __init__(self, sequence_seed=0, num_states=64, num_actions=8, feature_dim=24, drift_rate=0.1, reset_decoys_each_generation=False):
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
            self.features[s] = v / (np.linalg.norm(v) + 1e-08)
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
        t = self.targets[self.current_state]
        d = self.decoys[self.current_state]
        if action == t:
            reward = 1.0
        elif action == d and d >= 0:
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
        for e in self.envs:
            e.set_generation(generation_index)

    def all_state_features(self):
        return self.envs[0].all_state_features()

    def get_obs(self):
        return np.stack([e.get_obs() for e in self.envs])

    def step(self, actions):
        rewards = np.zeros(self.n, dtype=np.float32)
        dones = np.zeros(self.n, dtype=bool)
        states = np.zeros(self.n, dtype=np.int64)
        for i, e in enumerate(self.envs):
            r, d, s, _ = e.step(int(actions[i]))
            rewards[i], dones[i], states[i] = (r, d, s)
            e.reset_attempt()
        return rewards, dones, states, np.zeros(self.n, dtype=bool)

# Networks

def init_uniform_batched(num_parallel, out_size, in_size, bound, seed):
    tensors = []
    for i in range(num_parallel):
        gen = torch.Generator()
        gen.manual_seed(int(seed * 1000003 + i * 1009 + out_size * 17 + in_size))
        t = torch.empty(out_size, in_size).uniform_(-bound, bound, generator=gen)
        tensors.append(t)
    return torch.stack(tensors)

class BatchedMLP(nn.Module):

    def __init__(self, sizes, num_parallel, base_seed):
        super().__init__()
        self.sizes = sizes
        self.num_parallel = num_parallel
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for i in range(1, len(sizes)):
            bound = math.sqrt(6 / sizes[i - 1])
            w = init_uniform_batched(num_parallel, sizes[i], sizes[i - 1], bound, base_seed + i)
            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(torch.zeros(num_parallel, sizes[i])))

    def forward(self, x):
        h = x
        length = len(self.weights)
        for i in range(length):
            h = torch.einsum('nij,nj->ni', self.weights[i], h) + self.biases[i]
            if i < length - 1:
                h = F.relu(h)
        return h

    def reset_layers(self, layer_indices, base_seed):
        with torch.no_grad():
            for i in layer_indices:
                bound = math.sqrt(6 / self.sizes[i])
                w = init_uniform_batched(self.num_parallel, self.sizes[i + 1], self.sizes[i], bound, base_seed + i)
                self.weights[i].copy_(w.to(self.weights[i].device))
                self.biases[i].zero_()

class BatchedLateralNet(nn.Module):

    def __init__(self, sizes, num_parallel, base_seed, parents=None, use_decay=True):
        super().__init__()
        self.sizes = sizes
        self.num_parallel = num_parallel
        self.use_decay = use_decay
        if parents is None:
            parents = []
        self.num_parents = len(parents)
        for c, (pw, pb) in enumerate(parents):
            for li, w in enumerate(pw):
                self.register_buffer(f'p{c}W{li}', w)
            for li, b in enumerate(pb):
                self.register_buffer(f'p{c}B{li}', b)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.laterals = nn.ParameterDict()
        self.output_extras = nn.ParameterDict()
        length = len(sizes)
        for i in range(1, length):
            own_input_size = sizes[i - 1]
            bound = math.sqrt(6 / own_input_size)
            w = init_uniform_batched(num_parallel, sizes[i], own_input_size, bound, base_seed + i * 11)
            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(torch.zeros(num_parallel, sizes[i])))
            if i >= 2 and self.num_parents > 0:
                for c in range(self.num_parents):
                    lat = init_uniform_batched(num_parallel, sizes[i], sizes[i - 1], bound * lateral_init_scale, base_seed + i * 23 + c * 101)
                    self.laterals[f'l{i}C{c}'] = nn.Parameter(lat)
            if i == length - 1 and self.num_parents > 0:
                for c in range(self.num_parents):
                    ex = init_uniform_batched(num_parallel, sizes[i], sizes[-1], bound * lateral_init_scale, base_seed + i * 37 + c * 103)
                    self.output_extras[f'oeC{c}'] = nn.Parameter(ex)

    @property
    def has_parents(self):
        return self.num_parents > 0

    def column_forward(self, c, x):
        acts = [x]
        length = len(self.sizes) - 1
        for li in range(length):
            w = getattr(self, f'p{c}W{li}')
            b = getattr(self, f'p{c}B{li}')
            h = torch.einsum('nij,nj->ni', w, acts[-1]) + b
            if li < length - 1:
                h = F.relu(h)
            acts.append(h)
        return acts

    def forward(self, x, lateral_scale=1.0):
        if not self.use_decay:
            lateral_scale = 1.0
        parent_activations = [self.column_forward(c, x) for c in range(self.num_parents)]
        h = x
        length = len(self.weights)
        for i in range(length):
            out = torch.einsum('nij,nj->ni', self.weights[i], h) + self.biases[i]
            if i + 1 >= 2 and self.num_parents > 0:
                for c in range(self.num_parents):
                    if f'l{i + 1}C{c}' in self.laterals:
                        out = out + lateral_scale * torch.einsum('nij,nj->ni', self.laterals[f'l{i + 1}C{c}'], parent_activations[c][i])
            if i == length - 1 and self.num_parents > 0:
                for c in range(self.num_parents):
                    if f'oeC{c}' in self.output_extras:
                        out = out + lateral_scale * torch.einsum('nij,nj->ni', self.output_extras[f'oeC{c}'], parent_activations[c][-1])
            if i < length - 1:
                out = F.relu(out)
            h = out
        return h

    def detach_active(self):
        return ([w.detach().clone() for w in self.weights], [b.detach().clone() for b in self.biases])

class PersistentPNN(nn.Module):

    def __init__(self, sizes, num_parallel, base_seed):
        super().__init__()
        self.sizes = sizes
        self.num_parallel = num_parallel
        self.base_seed = base_seed
        self.columns = nn.ModuleList()
        self.add_column()

    def add_column(self):
        c = len(self.columns)
        col = nn.Module()
        col.weights = nn.ParameterList()
        col.biases = nn.ParameterList()
        col.laterals = nn.ParameterDict()
        col.gates = nn.ParameterDict()
        length = len(self.sizes)
        for i in range(1, length):
            bound = math.sqrt(6 / self.sizes[i - 1])
            w = init_uniform_batched(self.num_parallel, self.sizes[i], self.sizes[i - 1], bound, self.base_seed + c * 7919 + i * 11)
            col.weights.append(nn.Parameter(w))
            col.biases.append(nn.Parameter(torch.zeros(self.num_parallel, self.sizes[i])))
            if i >= 2 and c > 0:
                for j in range(c):
                    lat = init_uniform_batched(self.num_parallel, self.sizes[i], self.sizes[i - 1], bound * 1.0, self.base_seed + c * 131 + i * 23 + j * 101)
                    col.laterals[f'l{i}C{j}'] = nn.Parameter(lat)
            if i == length - 1 and c > 0:
                for j in range(c):
                    ex = init_uniform_batched(self.num_parallel, self.sizes[-1], self.sizes[-1], bound * 1.0, self.base_seed + c * 137 + j * 103)
                    col.laterals[f'oeC{j}'] = nn.Parameter(ex)
        if c > 0:
            for j in range(c):
                col.gates[f'gC{j}'] = nn.Parameter(torch.full((self.num_parallel,), 1.0 / c))
        self.columns.append(col)

    def forward(self, x):
        column_activations = []
        length = len(self.sizes) - 1
        for c, col in enumerate(self.columns):
            acts = [x]
            h = x
            for i in range(length):
                out = torch.einsum('nij,nj->ni', col.weights[i], h) + col.biases[i]
                if i + 1 >= 2 and c > 0:
                    for j in range(c):
                        if f'l{i + 1}C{j}' in col.laterals:
                            g = col.gates[f'gC{j}'].unsqueeze(-1)
                            out = out + g * torch.einsum('nij,nj->ni', col.laterals[f'l{i + 1}C{j}'], column_activations[j][i])
                if i == length - 1 and c > 0:
                    for j in range(c):
                        if f'oeC{j}' in col.laterals:
                            g = col.gates[f'gC{j}'].unsqueeze(-1)
                            out = out + g * torch.einsum('nij,nj->ni', col.laterals[f'oeC{j}'], column_activations[j][-1])
                if i < length - 1:
                    out = F.relu(out)
                acts.append(out)
                h = out
            column_activations.append(acts)
        return column_activations[-1][-1]

    def freeze_and_grow(self):
        for col in self.columns:
            for p in col.parameters():
                p.requires_grad_(False)
        self.add_column()

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

# Training methods

def sample_action(logits):
    lp = F.log_softmax(logits / temperature, dim=-1)
    p = lp.exp()
    actions = torch.multinomial(p, 1).squeeze(-1)
    return (actions, lp.gather(1, actions.unsqueeze(-1)).squeeze(-1), -(p * lp).sum(-1).mean())

def normalize_per_slot(x, eps=1e-08):
    return (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + eps)

class RegularTrainer:
    name = 'Regular'
    needs_generation = False

    def __init__(self, s, n, bs):
        self.sizes = s
        self.num_parallel = n
        self.base_seed = bs
        self.net = BatchedMLP(s, n, bs).to(device)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)

    def on_generation(self, generation_index):
        pass

    def set_distill_states(self, feats):
        pass

    def act(self, obs):
        a, lp, ent = sample_action(self.net(obs))
        return (a, lp, ent)

    def update(self, buf, adv, entropy_buffer=None):
        loss = -(torch.stack(buf, 0) * adv).sum()
        if entropy_coef and entropy_buffer:
            loss = loss - entropy_coef * torch.stack(entropy_buffer).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class ResetTrainer(RegularTrainer):
    name = 'Reset'
    needs_generation = True

    def __init__(self, s, n, bs):
        super().__init__(s, n, bs)
        self.gen = 0

    def on_generation(self, g):
        self.gen += 1
        self.net = BatchedMLP(self.sizes, self.num_parallel, self.base_seed + 10000 * self.gen).to(device)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)

class WarmStartTrainer(RegularTrainer):
    name = 'WarmStart'
    needs_generation = True

    def __init__(self, s, n, bs):
        super().__init__(s, n, bs)
        self.gen = 0

    def on_generation(self, g):
        self.gen += 1
        self.net.reset_layers([len(self.net.weights) - 1], self.base_seed + 20000 * self.gen)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)

class DistilledTrainer(RegularTrainer):
    name = 'Distilled'
    needs_generation = True

    def __init__(self, s, n, bs):
        super().__init__(s, n, bs)
        self.gen = 0
        self.distill_states = None

    def set_distill_states(self, feats):
        self.distill_states = torch.from_numpy(feats).to(device)

    def on_generation(self, g):
        self.gen += 1
        state_matrix = self.distill_states
        batched = state_matrix.unsqueeze(0).expand(self.num_parallel, -1, -1)
        with torch.no_grad():
            teacher_logits = torch.stack([self.net(batched[:, o, :]) for o in range(state_matrix.shape[0])], dim=1)
            teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
            teacher_probs = teacher_log_probs.exp()
        self.net = BatchedMLP(self.sizes, self.num_parallel, self.base_seed + 40000 * self.gen).to(device)
        distill_optimizer = torch.optim.SGD(self.net.parameters(), lr=distill_lr)
        for _ in range(distill_steps):
            student_logits = torch.stack([self.net(batched[:, o, :]) for o in range(state_matrix.shape[0])], dim=1)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(-1).mean()
            distill_optimizer.zero_grad()
            kl.backward()
            distill_optimizer.step()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)

class WeanNetTrainer(RegularTrainer):
    name = 'WeanNet'
    needs_generation = True

    def __init__(self, s, n, bs, decay_batches=None):
        self.sizes = s
        self.num_parallel = n
        self.base_seed = bs
        self.decay_batches = decay_batches if decay_batches else lateral_decay_batches
        self.net = BatchedLateralNet(s, n, bs, parents=None, use_decay=True).to(device)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        self.gen = 0
        self.batches_since_rebirth = 0

    def on_generation(self, g):
        self.gen += 1
        parent = self.net.detach_active()
        self.net = BatchedLateralNet(self.sizes, self.num_parallel, self.base_seed + 30000 * self.gen, parents=[parent], use_decay=True).to(device)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        self.batches_since_rebirth = 0

    def scale(self):
        if not self.net.has_parents:
            return 0.0
        prog = min(self.batches_since_rebirth / self.decay_batches, 1.0)
        return lateral_scale_min + (lateral_scale_init - lateral_scale_min) * (1 - prog)

    def act(self, obs):
        a, lp, ent = sample_action(self.net(obs, lateral_scale=self.scale()))
        return (a, lp, ent)

    def update(self, buf, adv, entropy_buffer=None):
        loss = -(torch.stack(buf, 0) * adv).sum()
        if entropy_coef and entropy_buffer:
            loss = loss - entropy_coef * torch.stack(entropy_buffer).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.net.has_parents:
            self.batches_since_rebirth += 1

class PNNResetTrainer(WeanNetTrainer):
    name = 'PNN-Reset'

    def __init__(self, s, n, bs, decay_batches=None):
        RegularTrainer.__init__(self, s, n, bs)
        self.net = BatchedLateralNet(s, n, bs, parents=None, use_decay=False).to(device)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        self.gen = 0
        self.batches_since_rebirth = 0

    def on_generation(self, g):
        self.gen += 1
        parent = self.net.detach_active()
        self.net = BatchedLateralNet(self.sizes, self.num_parallel, self.base_seed + 31000 * self.gen, parents=[parent], use_decay=False).to(device)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        self.batches_since_rebirth = 0

    def scale(self):
        return 1.0

class PNNTrainer(RegularTrainer):
    name = 'PNN'
    needs_generation = True

    def __init__(self, s, n, bs):
        self.sizes = s
        self.num_parallel = n
        self.base_seed = bs
        self.net = PersistentPNN(s, n, bs).to(device)
        self.optimizer = torch.optim.SGD(self.net.trainable_parameters(), lr=lr)
        self.gen = 0

    def on_generation(self, g):
        self.gen += 1
        self.net.freeze_and_grow()
        self.optimizer = torch.optim.SGD(self.net.trainable_parameters(), lr=lr)
# Evaluation

metric_keys = ['coreAcc', 'driftAcc', 'driftTrap', 'coreTrap', 'rewCore', 'rewDrift', 'rewOverall', 'entAll', 'entDrift']

def count_parameters(net):
    trainable = sum((p.numel() for p in net.parameters() if p.requires_grad))
    resident = sum((p.numel() for p in net.parameters())) + sum((b.numel() for b in net.buffers()))
    return (trainable, resident)

def probe_network(trainer, env):
    e0 = env.envs[0]
    feats = torch.from_numpy(e0.all_state_features()).to(device)
    xb = feats.unsqueeze(0).expand(trainer.num_parallel, -1, -1)
    scale = trainer.scale() if hasattr(trainer, 'scale') else None
    with torch.no_grad():
        if isinstance(trainer.net, BatchedLateralNet):
            logits = torch.stack([trainer.net(xb[:, s, :], lateral_scale=scale) for s in range(e0.num_states)], dim=1)
        else:
            logits = torch.stack([trainer.net(xb[:, s, :]) for s in range(e0.num_states)], dim=1)
        probs = F.softmax(logits / temperature, dim=-1)
        ent = -(probs * torch.log(probs + 1e-09)).sum(-1)
    preds = logits.argmax(-1).cpu().numpy()
    ent = ent.cpu().numpy()
    targets, decoys = (e0.targets, e0.decoys)
    cmask, dmask = (e0.core_mask, e0.drift_mask)
    has_drift = bool(dmask.any())
    acc = {k: [] for k in metric_keys}
    for p in range(trainer.num_parallel):
        pr = preds[p]
        r = np.full(len(targets), -0.1, dtype=np.float32)
        r[pr == targets] = 1.0
        decoy_hit = (pr == decoys) & (decoys >= 0)
        r[decoy_hit] = 0.5
        acc['coreAcc'].append((pr[cmask] == targets[cmask]).mean() if cmask.any() else np.nan)
        acc['coreTrap'].append(((pr[cmask] == decoys[cmask]) & (decoys[cmask] >= 0)).mean() if cmask.any() else np.nan)
        acc['rewCore'].append(r[cmask].mean() if cmask.any() else np.nan)
        acc['rewOverall'].append(r.mean())
        acc['entAll'].append(ent[p].mean())
        if has_drift:
            acc['driftAcc'].append((pr[dmask] == targets[dmask]).mean())
            acc['driftTrap'].append(((pr[dmask] == decoys[dmask]) & (decoys[dmask] >= 0)).mean())
            acc['rewDrift'].append(r[dmask].mean())
            acc['entDrift'].append(ent[p][dmask].mean())
        else:
            acc['driftAcc'].append(np.nan)
            acc['driftTrap'].append(np.nan)
            acc['rewDrift'].append(np.nan)
            acc['entDrift'].append(np.nan)
    return {k: float(np.nanmean(v)) for k, v in acc.items()}

# Experiment loop

def run_method(trainer_class, env_seed, train_seed, generations, generation_length, batch_size, num_parallel, hidden, env_kwargs, probe_every=0, baseline_mode='batchZscore'):
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    env = VectorizedDriftEnv(num_parallel, sequence_seed=env_seed, **env_kwargs)
    sizes = [env.obs_dim, hidden, hidden, env.num_actions]
    if trainer_class in (WeanNetTrainer, PNNResetTrainer):
        trainer = trainer_class(sizes, num_parallel, train_seed, decay_batches=lateral_decay_batches)
    else:
        trainer = trainer_class(sizes, num_parallel, train_seed)
    trainer.set_distill_states(env.all_state_features())
    num_states = env_kwargs.get('num_states', 64)
    value_table = np.zeros((num_parallel, num_states), dtype=np.float32)
    generation_stats = defaultdict(list)
    trainable_params, resident_params = ([], [])
    curve = []
    obs = torch.from_numpy(env.get_obs()).to(device)
    global_batch = 0
    for g in range(generations):
        env.set_generation(g)
        if g > 0 and trainer.needs_generation:
            trainer.on_generation(g)
        for b in range(generation_length):
            reward_buffer = np.zeros((batch_size, num_parallel), dtype=np.float32)
            state_buffer = np.zeros((batch_size, num_parallel), dtype=np.int64)
            log_prob_buffer, entropy_buffer = ([], [])
            for step in range(batch_size):
                actions, lp, ent = trainer.act(obs)
                r, d, state_indices, _ = env.step(actions.cpu().numpy())
                reward_buffer[step] = r
                state_buffer[step] = state_indices
                log_prob_buffer.append(lp)
                entropy_buffer.append(ent)
                obs = torch.from_numpy(env.get_obs()).to(device)
            if baseline_mode == 'stateEma':
                centered = np.zeros_like(reward_buffer)
                for st in range(batch_size):
                    for p in range(num_parallel):
                        s = state_buffer[st, p]
                        centered[st, p] = reward_buffer[st, p] - value_table[p, s]
                        value_table[p, s] = (1 - ema_beta) * value_table[p, s] + ema_beta * reward_buffer[st, p]
                advantages_np = centered / (centered.std(0, keepdims=True) + 1e-08)
            else:
                advantages_np = normalize_per_slot(reward_buffer)
            adv = torch.from_numpy(advantages_np).to(device)
            trainer.update(log_prob_buffer, adv, entropy_buffer)
            global_batch += 1
            if probe_every and b % probe_every == 0:
                pm = probe_network(trainer, env)
                pm['globalBatch'] = global_batch
                curve.append(pm)
        pm = probe_network(trainer, env)
        for k, v in pm.items():
            generation_stats[k].append(v)
        tr, res = count_parameters(trainer.net)
        trainable_params.append(tr)
        resident_params.append(res)
    out = {k: np.array(v) for k, v in generation_stats.items()}
    out['paramsTrain'] = np.array(trainable_params)
    out['paramsRes'] = np.array(resident_params)
    out['curve'] = curve
    return out
# Reporting and plots

colors = {'Regular': 'tab:gray', 'Reset': 'tab:orange', 'WarmStart': 'tab:green', 'Distilled': 'tab:purple', 'WeanNet': 'tab:blue', 'PNN-Reset': 'tab:cyan', 'PNN': 'tab:red'}
frozen = {'Distilled', 'PNN-Reset', 'PNN'}
keys = [('driftAcc', 'DRIFT FOUND 1.0'), ('driftTrap', 'DRIFT TRAP 0.5'), ('coreAcc', 'CORE (retention)'), ('rewOverall', 'REWARD (overall)')]

def plot_sweep_lines(ax, results, entropy_values, labels, names, metric, title, ylim=(-0.05, 1.05)):
    for n in names:
        ys = [np.nanmean(results[l][n][metric][:, -1]) for l in labels]
        es = [np.nanstd(results[l][n][metric][:, -1]) for l in labels]
        ls = '--' if n in frozen else '-'
        ax.errorbar(entropy_values, ys, yerr=es, marker='o', ls=ls, lw=2, ms=4, capsize=2, label=n, color=colors[n])
    ax.set_xlabel('entropy coefficient')
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    if ylim:
        ax.set_ylim(*ylim)

def main():
    global lateral_decay_batches, entropy_coef
    generations = 10
    generation_length = 600
    batch_size = 32
    num_parallel = 8
    hidden = 64
    env_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    num_replicates = 1
    seed_base = 100000
    entropy_values = [0.0, 0.05, 0.1, 0.2]
    lateral_decay_batches = int(generation_length * 0.5)
    kwargs = dict(num_states=64, num_actions=8, feature_dim=24, drift_rate=0.1)
    classes = [RegularTrainer, ResetTrainer, WarmStartTrainer, DistilledTrainer, WeanNetTrainer, PNNResetTrainer, PNNTrainer]
    names = [c.name for c in classes]
    labels = [f'{e:g}' for e in entropy_values]
    print(f'\n--- Entropy Sweep ---')
    print(f'Coefs: {entropy_values} | Gens: {generations} | Seeds: {len(env_seeds)} | Device: {device}\n')
    t0 = time.time()
    results = {l: {} for l in labels}
    for e, label in zip(entropy_values, labels):
        entropy_coef = e
        for cls in classes:
            runs = defaultdict(list)
            for es in env_seeds:
                for rep in range(num_replicates):
                    seed = seed_base + es * 997 + rep * 31
                    res = run_method(cls, es, seed, generations, generation_length, batch_size, num_parallel, hidden, kwargs, probe_every=0, baseline_mode='batchZscore')
                    for k in metric_keys:
                        runs[k].append(res[k])
            results[label][cls.name] = {k: np.stack(v) for k, v in runs.items()}
            f = lambda k: np.nanmean(results[label][cls.name][k][:, -1])
            print(f"Coef {e:g} | {cls.name:11s} | Core {f('coreAcc'):.2f} | Rew {f('rewOverall'):+.2f} | Drift1.0 {f('driftAcc'):.2f} | Trap0.5 {f('driftTrap'):.2f}")
    entropy_coef = 0.0
    print(f'\nTotal runtime: {time.time() - t0:.0f}s')

    def cell(l, n, k):
        return np.nanmean(results[l][n][k][:, -1])
    hdr = '  ' + f"{'Method':11s} " + ' '.join((f'{l:>7s}' for l in labels)) + f" {'Net':>7s}"
    print('\n--- Final Generation Stats ---')
    for k, label in keys:
        print(f'\n  {label}\n{hdr}')
        order = sorted(names, key=lambda n: cell(labels[0], n, k), reverse=True)
        for n in order:
            vals = [cell(l, n, k) for l in labels]
            net = vals[-1] - vals[0]
            print('  ' + f'{n:11s} ' + ' '.join((f'{v:7.2f}' for v in vals)) + f' {net:+7.2f}')
    print('\n--- Ablation: WeanNet vs PNN-Reset ---')
    print('  ' + f"{'Entropy':>8s} {'WeanNet':>9s} {'PNN-Reset':>10s} {'Gap':>7s}")
    for e, l in zip(entropy_values, labels):
        w = cell(l, 'WeanNet', 'driftAcc')
        p = cell(l, 'PNN-Reset', 'driftAcc')
        print('  ' + f'{e:8g} {w:9.2f} {p:10.2f} {w - p:+7.2f}')
    plastic = [n for n in names if n not in frozen]
    print('\n--- Group Means (Escape) ---')
    print('  ' + f"{'Entropy':>8s} {'Plastic':>9s} {'Frozen':>9s}")
    for e, l in zip(entropy_values, labels):
        pm = np.nanmean([cell(l, n, 'driftAcc') for n in plastic])
        fm = np.nanmean([cell(l, n, 'driftAcc') for n in frozen])
        print('  ' + f'{e:8g} {pm:9.2f} {fm:9.2f}')
    print()
    fig1, ax = plt.subplots(2, 2, figsize=(14, 10))
    plot_sweep_lines(ax[0, 0], results, entropy_values, labels, names, 'driftAcc', 'DRIFT FOUND 1.0 (escape)')
    plot_sweep_lines(ax[0, 1], results, entropy_values, labels, names, 'driftTrap', 'DRIFT TRAP 0.5 (stuck)')
    plot_sweep_lines(ax[1, 0], results, entropy_values, labels, names, 'coreAcc', 'CORE (retention)')
    plot_sweep_lines(ax[1, 1], results, entropy_values, labels, names, 'rewOverall', 'REWARD (overall)', ylim=None)
    fig1.suptitle('Entropy sweep (solid = plastic, dashed = frozen-parent)', fontsize=12)
    fig1.tight_layout()
    fig1.savefig('sweep1_metrics_vs_entropy.png', dpi=110)
    fig2, ax = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5), sharey=True)
    if len(labels) == 1:
        ax = [ax]
    for a, (e, l) in zip(ax, zip(entropy_values, labels)):
        for n in names:
            arr = results[l][n]['coreAcc']
            mean = np.nanmean(arr, 0)
            a.plot(np.arange(1, len(mean) + 1), mean, '-o', label=n, color=colors[n], lw=2, ms=4)
        a.set_title(f'core vs generation -- entropy {e:g}', fontsize=10)
        a.set_xlabel('Generation')
        a.set_ylabel('Core accuracy')
        a.set_ylim(-0.05, 1.05)
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig2.tight_layout()
    fig2.savefig('sweep2_core_curves.png', dpi=110)
    print('Charts generated: sweep1_metrics_vs_entropy.png, sweep2_core_curves.png')


if __name__ == '__main__':
    main()
