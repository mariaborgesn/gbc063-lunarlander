"""Agentes para o LunarLander em grade: aleatório, Q-learning tabular e DQN.

O DQN é implementado em NumPy puro (sem PyTorch): MLP 5 → 64 → 64 → 4 com
ReLU, otimizador Adam, perda de Huber, memória de replay e rede-alvo.
"""
import numpy as np

from grid_lander import GridLunarLander, N_ACTIONS, N_FEAT, N_STATES

GAMMA = 0.99
FEAT = GridLunarLander.onehot


def evaluate(policy, n_episodes=200, seed=12345):
    """Política gulosa em episódios com sementes fixas (iguais para todos)."""
    env = GridLunarLander(seed)
    rets, outs, lens = [], [], []
    for _ in range(n_episodes):
        s, g, n = env.reset(), 0.0, 0
        while True:
            s, r, te, tr, o = env.step(policy(s))
            g += r
            n += 1
            if te or tr:
                break
        rets.append(g)
        outs.append(o)
        lens.append(n)
    return np.array(rets), outs, np.array(lens)


def landed_rate(outs):
    return float(np.mean([o == "pouso" for o in outs]))


# ---------------------------------------------------------------------------
class TabularQ:
    def __init__(self, seed, alpha=0.1):
        self.Q = np.zeros((N_STATES, N_ACTIONS))
        self.rng = np.random.default_rng(seed)
        self.alpha = alpha

    def act(self, s, eps):
        if self.rng.random() < eps:
            return int(self.rng.integers(N_ACTIONS))
        return int(self.Q[GridLunarLander.index(s)].argmax())

    def greedy(self, s):
        return int(self.Q[GridLunarLander.index(s)].argmax())

    def update(self, s, a, r, s2, done):
        i, j = GridLunarLander.index(s), GridLunarLander.index(s2)
        target = r if done else r + GAMMA * self.Q[j].max()
        self.Q[i, a] += self.alpha * (target - self.Q[i, a])

    def visited(self):
        return int((np.abs(self.Q).sum(1) > 0).sum())


# ---------------------------------------------------------------------------
class MLP:
    def __init__(self, sizes, rng):
        self.W = [rng.normal(0, np.sqrt(2 / m), (m, n)).astype(np.float32)
                  for m, n in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(n, np.float32) for n in sizes[1:]]

    def params(self):
        return self.W + self.b

    def copy_from(self, other):
        for p, q in zip(self.params(), other.params()):
            p[...] = q

    def forward(self, x, keep=False):
        hs = [x]
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if k < len(self.W) - 1:
                x = np.maximum(x, 0)
            hs.append(x)
        return (x, hs) if keep else x

    def backward(self, hs, dout):
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        d = dout
        for k in range(len(self.W) - 1, -1, -1):
            gW[k] = hs[k].T @ d
            gb[k] = d.sum(0)
            if k > 0:
                d = (d @ self.W[k].T) * (hs[k] > 0)
        return gW + gb


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


class DQN:
    def __init__(self, seed, hidden=64, lr=1e-3, buffer_size=50_000,
                 batch_size=64, target_every=500, learning_starts=1_000, double=True):
        self.double = double
        self.rng = np.random.default_rng(seed)
        self.net = MLP([N_FEAT, hidden, hidden, N_ACTIONS], self.rng)
        self.target = MLP([N_FEAT, hidden, hidden, N_ACTIONS], self.rng)
        self.target.copy_from(self.net)
        self.opt = Adam(self.net.params(), lr)
        self.N, self.bs = buffer_size, batch_size
        self.S = np.zeros((buffer_size, N_FEAT), np.float32)
        self.S2 = np.zeros((buffer_size, N_FEAT), np.float32)
        self.A = np.zeros(buffer_size, np.int64)
        self.R = np.zeros(buffer_size, np.float32)
        self.D = np.zeros(buffer_size, np.float32)
        self.n = self.ptr = self.steps = 0
        self.target_every, self.learning_starts = target_every, learning_starts
        self.losses = []

    def q(self, s):
        return self.net.forward(FEAT(s)[None])[0]

    def act(self, s, eps):
        if self.rng.random() < eps:
            return int(self.rng.integers(N_ACTIONS))
        return int(self.q(s).argmax())

    def greedy(self, s):
        return int(self.q(s).argmax())

    def update(self, s, a, r, s2, done):
        k = self.ptr
        self.S[k], self.S2[k] = FEAT(s), FEAT(s2)
        self.A[k], self.R[k], self.D[k] = a, r, float(done)
        self.ptr = (k + 1) % self.N
        self.n = min(self.n + 1, self.N)
        self.steps += 1
        if self.n < self.learning_starts:
            return
        idx = self.rng.integers(0, self.n, self.bs)
        qt = self.target.forward(self.S2[idx])
        if self.double:   # Double DQN: a rede online escolhe, a alvo avalia
            a2 = self.net.forward(self.S2[idx]).argmax(1)
            q2 = qt[np.arange(self.bs), a2]
        else:
            q2 = qt.max(1)
        y = self.R[idx] + GAMMA * (1 - self.D[idx]) * q2
        q, hs = self.net.forward(self.S[idx], keep=True)
        rows = np.arange(self.bs)
        err = q[rows, self.A[idx]] - y
        derr = np.clip(err, -1, 1) / self.bs            # gradiente da Huber
        dout = np.zeros_like(q)
        dout[rows, self.A[idx]] = derr
        self.opt.step(self.net.backward(hs, dout))
        if self.steps % 50 == 0:
            self.losses.append(float(np.mean(np.where(np.abs(err) < 1, 0.5 * err ** 2,
                                                      np.abs(err) - 0.5))))
        if self.steps % self.target_every == 0:
            self.target.copy_from(self.net)


# ---------------------------------------------------------------------------
def train(agent, n_episodes, seed, eval_every=250, n_eval=200,
          eps_start=1.0, eps_end=0.05, eps_frac=0.6):
    env = GridLunarLander(seed)
    ep_returns, curve = [], []
    for ep in range(n_episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / (eps_frac * n_episodes))
        s, g = env.reset(), 0.0
        while True:
            a = agent.act(s, eps)
            s2, r, te, tr, _ = env.step(a)
            agent.update(s, a, r, s2, te)
            s, g = s2, g + r
            if te or tr:
                break
        ep_returns.append(g)
        if (ep + 1) % eval_every == 0:
            rets, outs, _ = evaluate(agent.greedy, n_eval)
            curve.append((ep + 1, float(rets.mean()), landed_rate(outs)))
    return np.array(ep_returns), np.array(curve)
