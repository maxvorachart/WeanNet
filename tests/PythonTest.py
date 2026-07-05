"""
drift_test2.py — Concept Drift with a Local Optimum Trap
Testing how well WeanNet retains core knowledge compared to other models.
"""

import sys
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

# Hide harmless nanmean warnings from generation 0
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Basic hyperparameters
lr = 0.02
gamma = 0.95
temperature = 1.0

# Lateral connection settings for WeanNet
lateralInitScale = 0.1
lateralScaleInit = 1.0
lateralScaleMin = 0.05
lateralDecayBatches = 120

# Distillation settings
distillSteps = 150
distillLr = 0.05

# Entropy bonus for exploration (0.0 means no bonus)
entropyCoef = 0.0
emaBeta = 0.1


# ============================================================
# The Trap Environment
# ============================================================

class DriftEnv:
    def __init__(self, sequenceSeed=0, numStates=64, numActions=8, featureDim=24,
                 driftRate=0.1, resetDecoysEachGen=False):
        self.rng = np.random.RandomState(sequenceSeed)
        self.numStates = numStates
        self.numActions = numActions
        self.featureDim = featureDim
        self.driftRate = driftRate
        self.resetDecoysEachGen = resetDecoysEachGen

        # Build random features for the states
        ing = self.rng.normal(0, 1, size=(20, featureDim)).astype(np.float32)
        self.features = np.zeros((numStates, featureDim), dtype=np.float32)
        for s in range(numStates):
            pick = self.rng.choice(20, size=3, replace=False)
            v = ing[pick].mean(axis=0)
            self.features[s] = v / (np.linalg.norm(v) + 1e-8)

        self.targets = self.rng.randint(0, numActions, size=numStates)
        self.decoys = np.full(numStates, -1, dtype=np.int64)

        # Track which states are core vs drifted
        self.coreMask = np.ones(numStates, dtype=bool)
        self.driftMask = np.zeros(numStates, dtype=bool)

        self.resetAttempt()

    @property
    def obsDim(self): return self.featureDim

    def setGeneration(self, genIdx):
        if genIdx == 0: return

        # Pick random states to drift
        numDrift = max(1, int(self.numStates * self.driftRate))
        driftIdx = self.rng.choice(self.numStates, size=numDrift, replace=False)

        self.coreMask[:] = True
        self.coreMask[driftIdx] = False
        self.driftMask[:] = False
        self.driftMask[driftIdx] = True

        # Clear out old decoys if requested
        if self.resetDecoysEachGen:
            self.decoys[:] = -1

        # The old correct answer becomes a trap (decoy)
        for s in driftIdx:
            self.decoys[s] = self.targets[s]
            newA = self.rng.randint(0, self.numActions)
            while newA == self.decoys[s]:
                newA = self.rng.randint(0, self.numActions)
            self.targets[s] = newA

    def resetAttempt(self):
        self.curState = self.rng.randint(0, self.numStates)

    def getObs(self):
        return self.features[self.curState].copy()

    def allStateFeatures(self):
        return self.features.copy()

    def step(self, action):
        t = self.targets[self.curState]
        d = self.decoys[self.curState]

        # Reward logic: 1 for exact hit, 0.5 for falling into the trap, -0.1 for garbage
        if action == t:
            reward = 1.0
        elif action == d and d >= 0:
            reward = 0.5
        else:
            reward = -0.1
        return reward, True, self.curState, False


class VectorizedDriftEnv:
    def __init__(self, nEnvs, sequenceSeed=0, **kw):
        self.envs = [DriftEnv(sequenceSeed, **kw) for _ in range(nEnvs)]
        self.n = nEnvs

    @property
    def obsDim(self): return self.envs[0].obsDim
    @property
    def numActions(self): return self.envs[0].numActions

    def setGeneration(self, genIdx):
        for e in self.envs: e.setGeneration(genIdx)
    def allStateFeatures(self): return self.envs[0].allStateFeatures()
    def getObs(self): return np.stack([e.getObs() for e in self.envs])

    def step(self, actions):
        rewards = np.zeros(self.n, dtype=np.float32)
        dones = np.zeros(self.n, dtype=bool)
        states = np.zeros(self.n, dtype=np.int64)
        for i, e in enumerate(self.envs):
            r, d, s, _ = e.step(int(actions[i]))
            rewards[i], dones[i], states[i] = r, d, s
            e.resetAttempt()
        return rewards, dones, states, np.zeros(self.n, dtype=bool)


# ============================================================
# Networks
# ============================================================

def initUniformBatched(nParallel, outSize, inSize, bound, seed):
    # Setup weights with consistent seeds for parallel batches
    tensors = []
    for i in range(nParallel):
        gen = torch.Generator()
        gen.manual_seed(int(seed * 1000003 + i * 1009 + outSize * 17 + inSize))
        t = torch.empty(outSize, inSize).uniform_(-bound, bound, generator=gen)
        tensors.append(t)
    return torch.stack(tensors)

class BatchedMLP(nn.Module):
    def __init__(self, sizes, nParallel, baseSeed):
        super().__init__()
        self.sizes = sizes; self.nParallel = nParallel
        self.weights = nn.ParameterList(); self.biases = nn.ParameterList()
        for i in range(1, len(sizes)):
            bound = math.sqrt(6 / sizes[i - 1])
            w = initUniformBatched(nParallel, sizes[i], sizes[i - 1], bound, baseSeed + i)
            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(torch.zeros(nParallel, sizes[i])))

    def forward(self, x):
        h = x; length = len(self.weights)
        for i in range(length):
            h = torch.einsum("nij,nj->ni", self.weights[i], h) + self.biases[i]
            if i < length - 1: h = F.relu(h)
        return h

    def resetLayers(self, layerIndices, baseSeed):
        with torch.no_grad():
            for i in layerIndices:
                bound = math.sqrt(6 / self.sizes[i])
                w = initUniformBatched(self.nParallel, self.sizes[i + 1], self.sizes[i], bound, baseSeed + i)
                self.weights[i].copy_(w.to(self.weights[i].device)); self.biases[i].zero_()

class BatchedLateralNet(nn.Module):
    # This is the core net used for WeanNet
    def __init__(self, sizes, nParallel, baseSeed, parents=None, useDecay=True):
        super().__init__()
        self.sizes = sizes; self.nParallel = nParallel; self.useDecay = useDecay
        if parents is None: parents = []
        self.numParents = len(parents)

        # Load in parent weights if they exist
        for c, (pw, pb) in enumerate(parents):
            for li, w in enumerate(pw): self.register_buffer(f"p{c}W{li}", w)
            for li, b in enumerate(pb): self.register_buffer(f"p{c}B{li}", b)

        self.weights = nn.ParameterList(); self.biases = nn.ParameterList()
        self.laterals = nn.ParameterDict(); self.outputExtras = nn.ParameterDict()
        length = len(sizes)

        for i in range(1, length):
            ownIn = sizes[i - 1]; bound = math.sqrt(6 / ownIn)
            w = initUniformBatched(nParallel, sizes[i], ownIn, bound, baseSeed + i * 11)
            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(torch.zeros(nParallel, sizes[i])))

            # Setup lateral connections to the parent network
            if i >= 2 and self.numParents > 0:
                for c in range(self.numParents):
                    lat = initUniformBatched(nParallel, sizes[i], sizes[i - 1], bound * lateralInitScale, baseSeed + i * 23 + c * 101)
                    self.laterals[f"l{i}C{c}"] = nn.Parameter(lat)
            if i == length - 1 and self.numParents > 0:
                for c in range(self.numParents):
                    ex = initUniformBatched(nParallel, sizes[i], sizes[-1], bound * lateralInitScale, baseSeed + i * 37 + c * 103)
                    self.outputExtras[f"oeC{c}"] = nn.Parameter(ex)

    @property
    def hasParents(self): return self.numParents > 0

    def columnForward(self, c, x):
        acts = [x]; length = len(self.sizes) - 1
        for li in range(length):
            w = getattr(self, f"p{c}W{li}"); b = getattr(self, f"p{c}B{li}")
            h = torch.einsum("nij,nj->ni", w, acts[-1]) + b
            if li < length - 1: h = F.relu(h)
            acts.append(h)
        return acts

    def forward(self, x, lateralScale=1.0):
        if not self.useDecay: lateralScale = 1.0
        parentActs = [self.columnForward(c, x) for c in range(self.numParents)]
        h = x; length = len(self.weights)

        for i in range(length):
            out = torch.einsum("nij,nj->ni", self.weights[i], h) + self.biases[i]
            if i + 1 >= 2 and self.numParents > 0:
                for c in range(self.numParents):
                    if f"l{i+1}C{c}" in self.laterals:
                        out = out + lateralScale * torch.einsum("nij,nj->ni", self.laterals[f"l{i+1}C{c}"], parentActs[c][i])
            if i == length - 1 and self.numParents > 0:
                for c in range(self.numParents):
                    if f"oeC{c}" in self.outputExtras:
                        out = out + lateralScale * torch.einsum("nij,nj->ni", self.outputExtras[f"oeC{c}"], parentActs[c][-1])
            if i < length - 1: out = F.relu(out)
            h = out
        return h

    def detachActive(self):
        return ([w.detach().clone() for w in self.weights], [b.detach().clone() for b in self.biases])

class PersistentPNN(nn.Module):
    # Progressive Neural Network baseline
    def __init__(self, sizes, nParallel, baseSeed):
        super().__init__()
        self.sizes = sizes; self.nParallel = nParallel; self.baseSeed = baseSeed
        self.columns = nn.ModuleList(); self.addColumn()

    def addColumn(self):
        c = len(self.columns); col = nn.Module()
        col.weights = nn.ParameterList(); col.biases = nn.ParameterList()
        col.laterals = nn.ParameterDict(); col.gates = nn.ParameterDict()
        length = len(self.sizes)

        for i in range(1, length):
            bound = math.sqrt(6 / self.sizes[i - 1])
            w = initUniformBatched(self.nParallel, self.sizes[i], self.sizes[i - 1], bound, self.baseSeed + c * 7919 + i * 11)
            col.weights.append(nn.Parameter(w))
            col.biases.append(nn.Parameter(torch.zeros(self.nParallel, self.sizes[i])))
            if i >= 2 and c > 0:
                for j in range(c):
                    lat = initUniformBatched(self.nParallel, self.sizes[i], self.sizes[i - 1], bound * 1.0, self.baseSeed + c * 131 + i * 23 + j * 101)
                    col.laterals[f"l{i}C{j}"] = nn.Parameter(lat)
            if i == length - 1 and c > 0:
                for j in range(c):
                    ex = initUniformBatched(self.nParallel, self.sizes[-1], self.sizes[-1], bound * 1.0, self.baseSeed + c * 137 + j * 103)
                    col.laterals[f"oeC{j}"] = nn.Parameter(ex)
        if c > 0:
            for j in range(c): col.gates[f"gC{j}"] = nn.Parameter(torch.full((self.nParallel,), 1.0 / c))
        self.columns.append(col)

    def forward(self, x):
        colActs = []; length = len(self.sizes) - 1
        for c, col in enumerate(self.columns):
            acts = [x]; h = x
            for i in range(length):
                out = torch.einsum("nij,nj->ni", col.weights[i], h) + col.biases[i]
                if i + 1 >= 2 and c > 0:
                    for j in range(c):
                        if f"l{i+1}C{j}" in col.laterals:
                            g = col.gates[f"gC{j}"].unsqueeze(-1)
                            out = out + g * torch.einsum("nij,nj->ni", col.laterals[f"l{i+1}C{j}"], colActs[j][i])
                if i == length - 1 and c > 0:
                    for j in range(c):
                        if f"oeC{j}" in col.laterals:
                            g = col.gates[f"gC{j}"].unsqueeze(-1)
                            out = out + g * torch.einsum("nij,nj->ni", col.laterals[f"oeC{j}"], colActs[j][-1])
                if i < length - 1: out = F.relu(out)
                acts.append(out); h = out
            colActs.append(acts)
        return colActs[-1][-1]

    def freezeAndGrow(self):
        for col in self.columns:
            for p in col.parameters(): p.requires_grad_(False)
        self.addColumn()

    def trainableParameters(self): return [p for p in self.parameters() if p.requires_grad]


# ============================================================
# Trainers
# ============================================================

def sampleAction(logits):
    # Grab an action based on network probabilities
    lp = F.log_softmax(logits / temperature, dim=-1)
    p = lp.exp()
    actions = torch.multinomial(p, 1).squeeze(-1)
    return actions, lp.gather(1, actions.unsqueeze(-1)).squeeze(-1), -(p * lp).sum(-1).mean()

def normalizePerSlot(x, eps=1e-8):
    return (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + eps)

class RegularTrainer:
    name = "Regular"; needsGeneration = False
    def __init__(self, s, n, bs):
        self.sizes = s; self.nParallel = n; self.baseSeed = bs
        self.net = BatchedMLP(s, n, bs).to(device)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        self.lastEnt = None
    def onGeneration(self, genIdx): pass
    def setDistillStates(self, feats): pass

    def act(self, obs):
        a, lp, ent = sampleAction(self.net(obs)); self.lastEnt = ent; return a, lp, ent

    def update(self, buf, adv, entBuf=None):
        loss = -(torch.stack(buf, 0) * adv).sum()
        if entropyCoef and entBuf:
            loss = loss - entropyCoef * torch.stack(entBuf).sum()
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()

class ResetTrainer(RegularTrainer):
    name = "Reset"; needsGeneration = True
    def __init__(self, s, n, bs): super().__init__(s, n, bs); self.gen = 0
    def onGeneration(self, g):
        self.gen += 1
        self.net = BatchedMLP(self.sizes, self.nParallel, self.baseSeed + 10000 * self.gen).to(device)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)

class WarmStartTrainer(RegularTrainer):
    name = "WarmStart"; needsGeneration = True
    def __init__(self, s, n, bs): super().__init__(s, n, bs); self.gen = 0
    def onGeneration(self, g):
        self.gen += 1
        self.net.resetLayers([len(self.net.weights)-1], self.baseSeed + 20000 * self.gen)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)

class DistilledTrainer(RegularTrainer):
    name = "Distilled"; needsGeneration = True
    def __init__(self, s, n, bs): super().__init__(s, n, bs); self.gen = 0; self.distillStates = None
    def setDistillStates(self, feats): self.distillStates = torch.from_numpy(feats).to(device)

    def onGeneration(self, g):
        self.gen += 1
        sMat = self.distillStates
        batched = sMat.unsqueeze(0).expand(self.nParallel, -1, -1)

        with torch.no_grad():
            tLogits = torch.stack([self.net(batched[:, o, :]) for o in range(sMat.shape[0])], dim=1)
            tLogp = F.log_softmax(tLogits / temperature, dim=-1); tP = tLogp.exp()

        self.net = BatchedMLP(self.sizes, self.nParallel, self.baseSeed + 40000 * self.gen).to(device)
        dopt = torch.optim.SGD(self.net.parameters(), lr=distillLr)

        for _ in range(distillSteps):
            sLogits = torch.stack([self.net(batched[:, o, :]) for o in range(sMat.shape[0])], dim=1)
            sLogp = F.log_softmax(sLogits / temperature, dim=-1)
            kl = (tP * (tLogp - sLogp)).sum(-1).mean()
            dopt.zero_grad(); kl.backward(); dopt.step()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)

class WeanNetTrainer(RegularTrainer):
    name = "WeanNet"; needsGeneration = True
    def __init__(self, s, n, bs, decayBatches=None):
        self.sizes = s; self.nParallel = n; self.baseSeed = bs
        self.decayBatches = decayBatches if decayBatches else lateralDecayBatches
        self.net = BatchedLateralNet(s, n, bs, parents=None, useDecay=True).to(device)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        self.gen = 0; self.batchesSinceRebirth = 0; self.lastEnt = None

    def onGeneration(self, g):
        self.gen += 1
        parent = self.net.detachActive()
        self.net = BatchedLateralNet(self.sizes, self.nParallel, self.baseSeed + 30000 * self.gen, parents=[parent], useDecay=True).to(device)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        self.batchesSinceRebirth = 0

    def scale(self):
        # Wean the net off its parent smoothly over time
        if not self.net.hasParents: return 0.0
        prog = min(self.batchesSinceRebirth / self.decayBatches, 1.0)
        return lateralScaleMin + (lateralScaleInit - lateralScaleMin) * (1 - prog)

    def act(self, obs):
        a, lp, ent = sampleAction(self.net(obs, lateralScale=self.scale())); self.lastEnt = ent; return a, lp, ent

    def update(self, buf, adv, entBuf=None):
        loss = -(torch.stack(buf, 0) * adv).sum()
        if entropyCoef and entBuf:
            loss = loss - entropyCoef * torch.stack(entBuf).sum()
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        if self.net.hasParents: self.batchesSinceRebirth += 1

class PNNResetTrainer(WeanNetTrainer):
    name = "PNN-Reset"
    def __init__(self, s, n, bs, decayBatches=None):
        RegularTrainer.__init__(self, s, n, bs)
        self.net = BatchedLateralNet(s, n, bs, parents=None, useDecay=False).to(device)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        self.gen = 0; self.batchesSinceRebirth = 0

    def onGeneration(self, g):
        self.gen += 1
        parent = self.net.detachActive()
        self.net = BatchedLateralNet(self.sizes, self.nParallel, self.baseSeed + 31000 * self.gen, parents=[parent], useDecay=False).to(device)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        self.batchesSinceRebirth = 0

    def scale(self): return 1.0

class PNNTrainer(RegularTrainer):
    name = "PNN"; needsGeneration = True
    def __init__(self, s, n, bs):
        self.sizes = s; self.nParallel = n; self.baseSeed = bs
        self.net = PersistentPNN(s, n, bs).to(device)
        self.optimizer = torch.optim.SGD(self.net.trainableParameters(), lr=lr)
        self.gen = 0; self.lastEnt = None

    def onGeneration(self, g):
        self.gen += 1; self.net.freezeAndGrow()
        self.optimizer = torch.optim.SGD(self.net.trainableParameters(), lr=lr)


# ============================================================
# Probes & Cost Accounting
# ============================================================

metricKeys = ["coreAcc", "driftAcc", "driftTrap", "coreTrap",
               "rewCore", "rewDrift", "rewOverall", "entAll", "entDrift"]

def countParams(net):
    # Count trainable vs total parameters (including frozen buffers)
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    resident = sum(p.numel() for p in net.parameters()) + sum(b.numel() for b in net.buffers())
    return trainable, resident

def probeNetwork(trainer, env):
    # Test the network greedily on all states to log metrics
    e0 = env.envs[0]
    feats = torch.from_numpy(e0.allStateFeatures()).to(device)
    xb = feats.unsqueeze(0).expand(trainer.nParallel, -1, -1)
    scale = trainer.scale() if hasattr(trainer, "scale") else None

    with torch.no_grad():
        if isinstance(trainer.net, BatchedLateralNet):
            logits = torch.stack([trainer.net(xb[:, s, :], lateralScale=scale) for s in range(e0.numStates)], dim=1)
        else:
            logits = torch.stack([trainer.net(xb[:, s, :]) for s in range(e0.numStates)], dim=1)
        probs = F.softmax(logits / temperature, dim=-1)
        ent = -(probs * torch.log(probs + 1e-9)).sum(-1)

    preds = logits.argmax(-1).cpu().numpy()
    ent = ent.cpu().numpy()
    targets, decoys = e0.targets, e0.decoys
    cmask, dmask = e0.coreMask, e0.driftMask
    hasDrift = bool(dmask.any())

    acc = {k: [] for k in metricKeys}
    for p in range(trainer.nParallel):
        pr = preds[p]
        r = np.full(len(targets), -0.1, dtype=np.float32)
        r[pr == targets] = 1.0
        decoyHit = (pr == decoys) & (decoys >= 0)
        r[decoyHit] = 0.5

        acc["coreAcc"].append((pr[cmask] == targets[cmask]).mean() if cmask.any() else np.nan)
        acc["coreTrap"].append(((pr[cmask] == decoys[cmask]) & (decoys[cmask] >= 0)).mean() if cmask.any() else np.nan)
        acc["rewCore"].append(r[cmask].mean() if cmask.any() else np.nan)
        acc["rewOverall"].append(r.mean())
        acc["entAll"].append(ent[p].mean())
        if hasDrift:
            acc["driftAcc"].append((pr[dmask] == targets[dmask]).mean())
            acc["driftTrap"].append(((pr[dmask] == decoys[dmask]) & (decoys[dmask] >= 0)).mean())
            acc["rewDrift"].append(r[dmask].mean())
            acc["entDrift"].append(ent[p][dmask].mean())
        else:
            acc["driftAcc"].append(np.nan); acc["driftTrap"].append(np.nan)
            acc["rewDrift"].append(np.nan); acc["entDrift"].append(np.nan)

    return {k: float(np.nanmean(v)) for k, v in acc.items()}


# ============================================================
# Core Training Loop
# ============================================================

def runMethod(trainerCls, envSeed, trainSeed, generations, genLength, batchSize,
               nParallel, hidden, envKwargs, probeEvery=0, baselineMode="batchZscore"):
    torch.manual_seed(trainSeed)
    np.random.seed(trainSeed)
    env = VectorizedDriftEnv(nParallel, sequenceSeed=envSeed, **envKwargs)
    sizes = [env.obsDim, hidden, hidden, env.numActions]

    if trainerCls in (WeanNetTrainer, PNNResetTrainer):
        trainer = trainerCls(sizes, nParallel, trainSeed, decayBatches=lateralDecayBatches)
    else:
        trainer = trainerCls(sizes, nParallel, trainSeed)
    trainer.setDistillStates(env.allStateFeatures())

    numStates = envKwargs.get("numStates", 64)
    vTable = np.zeros((nParallel, numStates), dtype=np.float32)

    genStats = defaultdict(list)
    paramsTrain, paramsRes = [], []
    curve = []

    obs = torch.from_numpy(env.getObs()).to(device)
    globalBatch = 0
    for g in range(generations):
        env.setGeneration(g)
        if g > 0 and trainer.needsGeneration:
            trainer.onGeneration(g)

        for b in range(genLength):
            rbuf = np.zeros((batchSize, nParallel), dtype=np.float32)
            sbuf = np.zeros((batchSize, nParallel), dtype=np.int64)
            lpBuf, entBuf = [], []

            for step in range(batchSize):
                actions, lp, ent = trainer.act(obs)
                r, d, sIdx, _ = env.step(actions.cpu().numpy())
                rbuf[step] = r
                sbuf[step] = sIdx
                lpBuf.append(lp); entBuf.append(ent)
                obs = torch.from_numpy(env.getObs()).to(device)

            if baselineMode == "stateEma":
                centered = np.zeros_like(rbuf)
                for st in range(batchSize):
                    for p in range(nParallel):
                        s = sbuf[st, p]
                        centered[st, p] = rbuf[st, p] - vTable[p, s]
                        vTable[p, s] = (1 - emaBeta) * vTable[p, s] + emaBeta * rbuf[st, p]
                advNp = centered / (centered.std(0, keepdims=True) + 1e-8)
            else:
                advNp = normalizePerSlot(rbuf)

            adv = torch.from_numpy(advNp).to(device)
            trainer.update(lpBuf, adv, entBuf)
            globalBatch += 1

            if probeEvery and (b % probeEvery == 0):
                pm = probeNetwork(trainer, env)
                pm["globalBatch"] = globalBatch
                curve.append(pm)

        pm = probeNetwork(trainer, env)
        for k, v in pm.items():
            genStats[k].append(v)
        tr, res = countParams(trainer.net)
        paramsTrain.append(tr); paramsRes.append(res)

    out = {k: np.array(v) for k, v in genStats.items()}
    out["paramsTrain"] = np.array(paramsTrain)
    out["paramsRes"] = np.array(paramsRes)
    out["curve"] = curve
    return out


# ============================================================
# Setup & Graph Plotting
# ============================================================

colors = {"Regular": "tab:gray", "Reset": "tab:orange", "WarmStart": "tab:green",
          "Distilled": "tab:purple", "WeanNet": "tab:blue", "PNN-Reset": "tab:cyan", "PNN": "tab:red"}

# Methods with frozen parents get dashed lines
frozen = {"Distilled", "PNN-Reset", "PNN"}
keys = [("driftAcc", "DRIFT FOUND 1.0"), ("driftTrap", "DRIFT TRAP 0.5"),
       ("coreAcc", "CORE (retention)"), ("rewOverall", "REWARD (overall)")]

def sweepLines(ax, results, entValues, labels, names, metric, title, ylim=(-0.05, 1.05)):
    for n in names:
        ys = [np.nanmean(results[l][n][metric][:, -1]) for l in labels]
        es = [np.nanstd(results[l][n][metric][:, -1]) for l in labels]
        ls = "--" if n in frozen else "-"
        ax.errorbar(entValues, ys, yerr=es, marker="o", ls=ls, lw=2, ms=4,
                    capsize=2, label=n, color=colors[n])
    ax.set_xlabel("entropy coefficient"); ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    if ylim: ax.set_ylim(*ylim)

def main():
    global lateralDecayBatches, entropyCoef

    generations = 10
    genLength = 600
    batchSize = 32
    nParallel = 8
    hidden = 64
    envSeeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    nReplicates = 1
    seedBase = 100_000
    entValues = [0.0, 0.05, 0.1, 0.2]

    lateralDecayBatches = int(genLength * 0.5)
    kwargs = dict(numStates=64, numActions=8, featureDim=24, driftRate=0.1)
    classes = [RegularTrainer, ResetTrainer, WarmStartTrainer, DistilledTrainer,
               WeanNetTrainer, PNNResetTrainer, PNNTrainer]
    names = [c.name for c in classes]
    labels = [f"{e:g}" for e in entValues]

    print(f"\n--- Entropy Sweep ---")
    print(f"Coefs: {entValues} | Gens: {generations} | Seeds: {len(envSeeds)} | Device: {device}\n")

    t0 = time.time()
    results = {l: {} for l in labels}

    for e, label in zip(entValues, labels):
        entropyCoef = e
        for cls in classes:
            runs = defaultdict(list)
            for es in envSeeds:
                for rep in range(nReplicates):
                    seed = seedBase + es * 997 + rep * 31
                    res = runMethod(cls, es, seed, generations, genLength, batchSize,
                                     nParallel, hidden, kwargs, probeEvery=0,
                                     baselineMode="batchZscore")
                    for k in metricKeys:
                        runs[k].append(res[k])
            results[label][cls.name] = {k: np.stack(v) for k, v in runs.items()}
            f = lambda k: np.nanmean(results[label][cls.name][k][:, -1])
            print(f"Coef {e:g} | {cls.name:11s} | Core {f('coreAcc'):.2f} | Rew {f('rewOverall'):+.2f} | Drift1.0 {f('driftAcc'):.2f} | Trap0.5 {f('driftTrap'):.2f}")

    entropyCoef = 0.0
    print(f"\nTotal runtime: {time.time()-t0:.0f}s")

    def cell(l, n, k): return np.nanmean(results[l][n][k][:, -1])
    hdr = "  " + f"{'Method':11s} " + " ".join(f"{l:>7s}" for l in labels) + f" {'Net':>7s}"

    print("\n--- Final Generation Stats ---")
    for k, label in keys:
        print(f"\n  {label}\n{hdr}")
        order = sorted(names, key=lambda n: cell(labels[0], n, k), reverse=True)
        for n in order:
            vals = [cell(l, n, k) for l in labels]
            net = vals[-1] - vals[0]
            print("  " + f"{n:11s} " + " ".join(f"{v:7.2f}" for v in vals) + f" {net:+7.2f}")

    print("\n--- Ablation: WeanNet vs PNN-Reset ---")
    print("  " + f"{'Entropy':>8s} {'WeanNet':>9s} {'PNN-Reset':>10s} {'Gap':>7s}")
    for e, l in zip(entValues, labels):
        w = cell(l, "WeanNet", "driftAcc"); p = cell(l, "PNN-Reset", "driftAcc")
        print("  " + f"{e:8g} {w:9.2f} {p:10.2f} {w - p:+7.2f}")

    plastic = [n for n in names if n not in frozen]
    print("\n--- Group Means (Escape) ---")
    print("  " + f"{'Entropy':>8s} {'Plastic':>9s} {'Frozen':>9s}")
    for e, l in zip(entValues, labels):
        pm = np.nanmean([cell(l, n, "driftAcc") for n in plastic])
        fm = np.nanmean([cell(l, n, "driftAcc") for n in frozen])
        print("  " + f"{e:8g} {pm:9.2f} {fm:9.2f}")
    print()

    # Generate the charts
    fig1, ax = plt.subplots(2, 2, figsize=(14, 10))
    sweepLines(ax[0, 0], results, entValues, labels, names, "driftAcc", "DRIFT FOUND 1.0 (escape)")
    sweepLines(ax[0, 1], results, entValues, labels, names, "driftTrap", "DRIFT TRAP 0.5 (stuck)")
    sweepLines(ax[1, 0], results, entValues, labels, names, "coreAcc", "CORE (retention)")
    sweepLines(ax[1, 1], results, entValues, labels, names, "rewOverall", "REWARD (overall)", ylim=None)
    fig1.suptitle("Entropy sweep (solid = plastic, dashed = frozen-parent)", fontsize=12)
    fig1.tight_layout(); fig1.savefig("sweep1_metrics_vs_entropy.png", dpi=110)

    fig2, ax = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5), sharey=True)
    if len(labels) == 1: ax = [ax]
    for a, (e, l) in zip(ax, zip(entValues, labels)):
        for n in names:
            arr = results[l][n]["coreAcc"]; mean = np.nanmean(arr, 0)
            a.plot(np.arange(1, len(mean) + 1), mean, "-o", label=n, color=colors[n], lw=2, ms=4)
        a.set_title(f"core vs generation -- entropy {e:g}", fontsize=10)
        a.set_xlabel("Generation"); a.set_ylabel("Core accuracy")
        a.set_ylim(-0.05, 1.05); a.grid(alpha=0.3); a.legend(fontsize=7)
    fig2.tight_layout(); fig2.savefig("sweep2_core_curves.png", dpi=110)

    print("Charts generated: sweep1_metrics_vs_entropy.png, sweep2_core_curves.png")

if __name__ == "__main__":
    main()
