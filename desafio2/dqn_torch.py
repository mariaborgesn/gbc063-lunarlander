"""DQN em PyTorch para o LunarLander em grade (mesma interface do DQN em NumPy).

Mesma configuração do agents.DQN: estado em one-hot (38 entradas),
MLP 38 → 64 → 64 → 4 com ReLU, Adam (lr 1e-3), perda de Huber,
replay de 50 mil, lote de 64, rede-alvo copiada a cada 500 passos,
Double DQN e γ = 0,99.

    python run_experiment.py --backend torch
"""
import numpy as np
import torch
import torch.nn as nn

from grid_lander import GridLunarLander, N_ACTIONS, N_FEAT

GAMMA = 0.99
FEAT = GridLunarLander.onehot


def make_net(hidden):
    return nn.Sequential(nn.Linear(N_FEAT, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, N_ACTIONS))


class DQNTorch:
    def __init__(self, seed, hidden=64, lr=1e-3, buffer_size=50_000,
                 batch_size=64, target_every=500, learning_starts=1_000, double=True):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.net = make_net(hidden)
        self.target = make_net(hidden)
        self.target.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.N, self.bs, self.double = buffer_size, batch_size, double
        self.S = np.zeros((buffer_size, N_FEAT), np.float32)
        self.S2 = np.zeros((buffer_size, N_FEAT), np.float32)
        self.A = np.zeros(buffer_size, np.int64)
        self.R = np.zeros(buffer_size, np.float32)
        self.D = np.zeros(buffer_size, np.float32)
        self.n = self.ptr = self.steps = 0
        self.target_every, self.learning_starts = target_every, learning_starts
        self.losses = []

    @torch.no_grad()
    def q(self, s):
        return self.net(torch.from_numpy(FEAT(s))[None])[0].numpy()

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
        S, S2 = torch.from_numpy(self.S[idx]), torch.from_numpy(self.S2[idx])
        A = torch.from_numpy(self.A[idx])
        R, D = torch.from_numpy(self.R[idx]), torch.from_numpy(self.D[idx])
        with torch.no_grad():
            qt = self.target(S2)
            if self.double:   # Double DQN: a rede online escolhe, a alvo avalia
                a2 = self.net(S2).argmax(1, keepdim=True)
                q2 = qt.gather(1, a2).squeeze(1)
            else:
                q2 = qt.max(1).values
            y = R + GAMMA * (1 - D) * q2
        q = self.net(S).gather(1, A[:, None]).squeeze(1)
        loss = self.loss_fn(q, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        if self.steps % 50 == 0:
            self.losses.append(float(loss))
        if self.steps % self.target_every == 0:
            self.target.load_state_dict(self.net.state_dict())

    def weights(self):
        """Pesos no mesmo formato do agents.MLP (W como entrada × saída)."""
        lin = [m for m in self.net if isinstance(m, nn.Linear)]
        W = [m.weight.detach().numpy().T.copy() for m in lin]
        b = [m.bias.detach().numpy().copy() for m in lin]
        return W, b
