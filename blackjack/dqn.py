"""Double dueling DQN on the shoe composition, trained on experience from the numba engine.

State: the decision (dealer up card, hand, two-card flag) and the composition features
(experience.features). The network outputs V(s) and the advantages A(s, a); Q = V + A - mean of
A over the legal actions. hidden=() is the linear model: V and A are weighted sums of the
features, with weights per decision. Hidden layers add a nonlinear part on top of it.
Q-learning is off-policy, so the experience comes from a fixed exploring agent (a trained count
table plus random moves) and the targets use the best next action (double DQN: chosen by the
online network, valued by the target network).
"""
import os
import time

import numpy as np
import torch
from torch import nn

from blackjack import engine as E
from blackjack.counts import SYSTEMS, Bucketing, engine_params
from blackjack.experience import collect, features
from blackjack.qlearn import EdgeResult

N_DECISIONS = E.N_UP * E.N_HAND * 2


class DuelingNet(nn.Module):
    def __init__(self, hidden=(), activation="relu"):
        super().__init__()
        assert len(hidden) in (0, 2), "the engine plays the linear model or two hidden layers"
        self.hidden, self.activation = tuple(hidden), activation
        self.WV = nn.Parameter(torch.zeros(N_DECISIONS, E.N_FEAT))
        self.WA = nn.Parameter(torch.zeros(N_DECISIONS, E.N_ACTIONS, E.N_FEAT))
        self.deep = None
        if hidden:
            layers, width = [], E.N_UP + E.N_HAND + 1 + E.N_FEAT - 1
            for h in hidden:
                layers += [nn.Linear(width, h), nn.SiLU() if activation == "silu" else nn.ReLU()]
                width = h
            layers.append(nn.Linear(width, 1 + E.N_ACTIONS))
            nn.init.zeros_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)
            self.deep = nn.Sequential(*layers)

    def forward(self, d, x, mask):
        """d: decision index, x: features, mask: legal actions -> (Q, V, A); Q is -inf where illegal."""
        v = (self.WV[d] * x).sum(-1)
        a = torch.einsum("baf,bf->ba", self.WA[d], x)
        if self.deep is not None:
            u, h, t = d // (E.N_HAND * 2), (d // 2) % E.N_HAND, d % 2
            z = torch.cat([nn.functional.one_hot(u, E.N_UP), nn.functional.one_hot(h, E.N_HAND),
                           t[:, None], x[:, 1:]], 1).float()
            out = self.deep(z)
            v, a = v + out[:, 0], a + out[:, 1:]
        legal = mask.float()
        a = a - (a * legal).sum(1, keepdim=True) / legal.sum(1, keepdim=True)
        q = torch.where(mask, v[:, None] + a, torch.full_like(a, -torch.inf))
        return q, v, a


def _tensors(d, n_decks, device):
    mask_bits = torch.as_tensor(d["M"].astype(np.int64), device=device)
    return dict(
        d=torch.as_tensor((d["U"].astype(np.int64) * E.N_HAND + d["H"]) * 2 + d["T"], device=device),
        x=torch.as_tensor(features(d["X"], n_decks), dtype=torch.float32, device=device),
        mask=((mask_bits[:, None] >> torch.arange(E.N_ACTIONS, device=device)) & 1).bool(),
        a=torch.as_tensor(d["A"].astype(np.int64), device=device),
        n1=torch.as_tensor(d["N1"], device=device), r1=torch.as_tensor(d["R1"], device=device),
        n2=torch.as_tensor(d["N2"], device=device), r2=torch.as_tensor(d["R2"], device=device))


def _next_value(online, target, b, rows, results):
    """Value of the next decision (best action chosen online, valued by the target net), or the result."""
    out = results.clone()
    has = rows >= 0
    if has.any():
        r = rows[has]
        q_on = online(b["d"][r], b["x"][r], b["mask"][r])[0]
        q_tg = target(b["d"][r], b["x"][r], b["mask"][r])[0]
        out[has] = q_tg.gather(1, q_on.argmax(1, keepdim=True)).squeeze(1)
    return out


def train(rules, behavior, rounds=200_000_000, chunk_rounds=4_000_000, hidden=(), activation="relu",
          batch=32768, lr=1e-3, lr_end=1e-5, tau=0.01, eps=0.3, seed=0, device=None, log=print):
    """Train on fresh experience chunks from the behavior count table (with exploration eps)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    online = DuelingNet(hidden, activation).to(device)
    target = DuelingNet(hidden, activation).to(device)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=lr)
    chunks = max(1, rounds // chunk_rounds)
    decay = (lr_end / lr) ** (1 / max(1, chunks - 1))
    for c in range(chunks):
        t0 = time.time()
        b = _tensors(collect(rules, chunk_rounds, seed=seed + 1000 * c, eps=eps, qt=behavior),
                     rules.n_decks, device)
        n = len(b["a"])
        order = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            q = online(b["d"][idx], b["x"][idx], b["mask"][idx])[0].gather(1, b["a"][idx, None]).squeeze(1)
            with torch.no_grad():
                y = _next_value(online, target, b, b["n1"][idx], b["r1"][idx])
                split = b["n2"][idx] > -2
                if split.any():
                    y[split] += _next_value(online, target, b, b["n2"][idx][split], b["r2"][idx][split])
            loss = ((q - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for pt, po in zip(target.parameters(), online.parameters()):
                    pt.lerp_(po, tau)
            total += loss.item() * len(idx)
        for g in opt.param_groups:
            g["lr"] *= decay
        log(f"chunk {c + 1}/{chunks}: {n:,} decisions, loss {total / n:.4f}, {time.time() - t0:.1f}s")
    return online.cpu()


def engine_model(net):
    """(W, mlp, mdims) for acting in the engine. W: linear advantage weights (up, hand, two cards,
    action, feature); the best action is the one with the largest advantage. mlp, mdims: the
    network part flattened (layer weights as input x output, then biases) and its sizes."""
    W = net.WA.detach().numpy().astype(np.float64).reshape(E.N_UP, E.N_HAND, 2, E.N_ACTIONS, E.N_FEAT)
    if net.deep is None:
        return W, None, None
    linears = [m for m in net.deep if isinstance(m, nn.Linear)]
    mlp = np.concatenate([np.concatenate([l.weight.detach().numpy().T.ravel(), l.bias.detach().numpy()])
                          for l in linears]).astype(np.float64)
    mdims = np.array([net.hidden[0], net.hidden[1], 1 if net.activation == "silu" else 0], np.int64)
    return W, mlp, mdims


EDGE_BUCKETS = Bucketing.around(0.0, 0.5, 8)  # the edge the model predicts before the deal, in % per round


class ModelAgent:
    """A trained model playing in the engine: W (plus the network part mlp) picks the actions,
    Wr predicts the edge of a round from the composition before the deal, for betting."""

    def __init__(self, W, Wr, rules, mlp=None, mdims=None):
        self.W, self.Wr, self.rules, self.mlp, self.mdims = W, Wr, rules, mlp, mdims

    @property
    def mode(self):
        return 1 if self.mlp is None else 2

    @classmethod
    def from_net(cls, net, rules, n_rounds=20_000_000, seed=77):
        """Fit Wr by least squares on rounds played by the model itself."""
        W, mlp, mdims = engine_model(net)
        d = collect(rules, n_rounds, seed=seed, eps=0.0, W=W, Wr=np.zeros(E.N_FEAT), mlp=mlp, mdims=mdims)
        Wr, *_ = np.linalg.lstsq(d["P"].astype(np.float64), d["PR"].astype(np.float64), rcond=None)
        return cls(W, Wr, rules, mlp, mdims)

    def evaluate(self, n_rounds=100_000_000, seed=12345, workers=None):
        workers = workers or os.cpu_count() or 1
        r, nb = self.rules, EDGE_BUCKETS.n
        tags, cp = engine_params(SYSTEMS["hi_lo"], EDGE_BUCKETS, r.n_decks)
        stats = np.zeros((workers, r.n_seats, nb, 3))
        mlp = np.zeros(1) if self.mlp is None else self.mlp
        mdims = np.zeros(3, np.int64) if self.mdims is None else self.mdims
        E.run_parallel(np.zeros((workers, 1, 1, 1, 1, E.N_ACTIONS)), np.zeros((workers, 1, 1, 1, 1, E.N_ACTIONS), np.int64),
                       np.zeros((workers, nb)), np.zeros((workers, nb), np.int64), stats, max(1, n_rounds // workers),
                       seed, False, 0.0, tags, cp, r.n_decks, r.penetration, r.h17, r.das, r.double_min,
                       r.max_hands, r.resplit_aces, r.hit_split_aces, r.peek, r.surrender, r.insurance,
                       r.bj_payout, r.n_seats, self.mode, self.W, mlp, mdims, self.Wr)
        s = stats.sum(axis=0)
        labels = [EDGE_BUCKETS.label(k) for k in range(nb)]
        return EdgeResult.from_stats(labels, s.sum(axis=0), [EdgeResult.from_stats(labels, s[i]) for i in range(len(s))])

    def save(self, path):
        extra = {} if self.mlp is None else dict(mlp=self.mlp, mdims=self.mdims)
        np.savez_compressed(path, W=self.W, Wr=self.Wr, **extra)

    @classmethod
    def load(cls, path, rules):
        d = np.load(path)
        return cls(d["W"], d["Wr"], rules, d["mlp"] if "mlp" in d.files else None,
                   d["mdims"] if "mdims" in d.files else None)
