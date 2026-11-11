"""
LunarLander em grade — versão discreta do LunarLander-v3 (GBC063, Desafio 2).

O LunarLander-v3 do Gymnasium tem estado contínuo (s ∈ ℝ⁸). Aqui cada
variável vira um inteiro, como nos ambientes em grade da turma, e o
espaço de estados passa a ser finito (cabe numa Q-table):

    s = (x, y, vx, vy, θ)
    x  ∈ {-5..5}   coluna (0 = centro da área de pouso)       11 valores
    y  ∈ {0..10}   altura (0 = solo)                          11 valores
    vx ∈ {-2..2}   velocidade horizontal (células/passo)       5 valores
    vy ∈ {-3..2}   velocidade vertical (negativo = caindo)     6 valores
    θ  ∈ {-2..2}   inclinação (0 = reto; ±2 = tombado)         5 valores
    |S| = 11 · 11 · 5 · 6 · 5 = 18.150 estados

Ações (as mesmas 4 do Desafio 1):
    0 nada · 1 motor esquerdo · 2 motor principal · 3 motor direito

Física por passo:
    - gravidade: vy -= 1
    - motor principal: vy += 2 e, se o módulo estiver inclinado, empurra
      para o lado da inclinação (vx -= sinal(θ))
    - motor esquerdo: gira o módulo (θ += 1) e o empurra de lado (vx -= 1)
    - motor direito: o espelho do esquerdo
    - sem vento por padrão (como no Desafio 1): o acaso está só no início,
      que sorteia x, vx e θ (o "empurrão" inicial do v3)
    - com vento (--wind p): em cada passo, com probabilidade p, uma rajada
      muda vx em +1 ou −1 (metade das vezes cada), DEPOIS da ação do agente;
      as rajadas são independentes entre si, então o estado continua markoviano
    - posição: x += vx, y += vy

Fim do episódio:
    - pouso: chega a y = 0 com vy ≥ -1, |vx| ≤ 1 e θ = 0, na área de
      pouso |x| ≤ 1                                         → +100
      (pouso suave fora da área: só os +10 por perna)
    - queda: chega ao solo rápido demais, deslizando ou inclinado → -100
    - sair da tela (|x| > 5)                                   → -100
    - limite de 100 passos (truncamento)

Recompensa a cada passo (mesma estrutura do v3):
    r = Φ(s') − Φ(s) − 0,3·[principal] − 0,03·[lateral]  (+10 por perna no solo)
    Φ(s) = −100·dist − 100·vel − 100·|θ|/2
"""

import numpy as np

X_MIN, X_MAX = -5, 5
Y_MAX = 10
VX_MAX = 2
VY_MIN, VY_MAX = -3, 2
TH_MAX = 2
MAX_STEPS = 100
PAD = 1  # |x| <= PAD é a área de pouso

N_X = X_MAX - X_MIN + 1
N_Y = Y_MAX + 1
N_VX = 2 * VX_MAX + 1
N_VY = VY_MAX - VY_MIN + 1
N_TH = 2 * TH_MAX + 1
N_STATES = N_X * N_Y * N_VX * N_VY * N_TH
N_ACTIONS = 4
N_FEAT = N_X + N_Y + N_VX + N_VY + N_TH
DEFAULT_WIND = 0.0  # probabilidade de rajada por passo; run_experiment.py --wind muda
ACTION_NAMES = ["nada", "motor esq.", "motor principal", "motor dir."]


def clip(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def sign(v):
    return (v > 0) - (v < 0)


class GridLunarLander:
    def __init__(self, seed=None, wind_prob=None):
        self.rng = np.random.default_rng(seed)
        self.wind_prob = DEFAULT_WIND if wind_prob is None else wind_prob

    # ---------------------------------------------------------------- estado
    def obs(self):
        return (self.x, self.y, self.vx, self.vy, self.th)

    @staticmethod
    def index(s):
        x, y, vx, vy, th = s
        i = x - X_MIN
        i = i * N_Y + y
        i = i * N_VX + (vx + VX_MAX)
        i = i * N_VY + (vy - VY_MIN)
        i = i * N_TH + (th + TH_MAX)
        return i

    @staticmethod
    def features(s):
        """Vetor normalizado em [-1, 1] para a rede do DQN."""
        x, y, vx, vy, th = s
        return np.array([x / X_MAX, 2 * y / Y_MAX - 1, vx / VX_MAX,
                         (2 * vy - VY_MIN - VY_MAX) / (VY_MAX - VY_MIN),
                         th / TH_MAX], dtype=np.float32)

    @staticmethod
    def onehot(s):
        """Codificação one-hot por variável (11+11+5+6+5 = 38 entradas)."""
        x, y, vx, vy, th = s
        v = np.zeros(N_FEAT, dtype=np.float32)
        v[x - X_MIN] = 1
        v[N_X + y] = 1
        v[N_X + N_Y + vx + VX_MAX] = 1
        v[N_X + N_Y + N_VX + vy - VY_MIN] = 1
        v[N_X + N_Y + N_VX + N_VY + th + TH_MAX] = 1
        return v

    @staticmethod
    def potential(s, on_ground=False):
        x, y, vx, vy, th = s
        dist = np.hypot(x / X_MAX, y / Y_MAX)
        vel = np.hypot(vx / VX_MAX, vy / abs(VY_MIN))
        return -100 * dist - 100 * vel - 100 * abs(th) / TH_MAX + (20 if on_ground else 0)

    # ---------------------------------------------------------------- API
    def reset(self):
        self.x = int(self.rng.integers(-3, 4))
        self.y = Y_MAX
        self.vx = int(self.rng.integers(-1, 2))
        self.vy = 0
        self.th = int(self.rng.integers(-1, 2))
        self.t = 0
        return self.obs()

    def step(self, a):
        s = self.obs()
        phi0 = self.potential(s)
        fuel = 0.0
        vx, vy, th = self.vx, self.vy - 1, self.th          # gravidade
        if a == 2:                                           # principal
            vy += 2
            vx -= sign(th)
            fuel = 0.3
        elif a == 1:                                         # motor esquerdo
            th += 1
            vx -= 1
            fuel = 0.03
        elif a == 3:                                         # motor direito
            th -= 1
            vx += 1
            fuel = 0.03
        if self.rng.random() < self.wind_prob:               # rajada de vento
            vx += 1 if self.rng.random() < 0.5 else -1
        self.vx = clip(vx, -VX_MAX, VX_MAX)
        self.vy = clip(vy, VY_MIN, VY_MAX)
        self.th = clip(th, -TH_MAX, TH_MAX)
        self.x += self.vx
        self.y = min(self.y + self.vy, Y_MAX)
        self.t += 1

        terminated, outcome = False, None
        if abs(self.x) > X_MAX:
            terminated, outcome = True, "saiu"
            self.x = clip(self.x, X_MIN, X_MAX)
        elif self.y <= 0:
            self.y = 0
            terminated = True
            soft = self.vy >= -1 and abs(self.vx) <= 1 and self.th == 0
            outcome = ("pouso" if abs(self.x) <= PAD else "pouso fora") if soft else "queda"

        s2 = self.obs()
        if terminated and outcome in ("pouso", "pouso fora"):
            # repouso: velocidade zerada, duas pernas no solo
            self.vx = self.vy = 0
            s2 = self.obs()
            bonus = 100 if outcome == "pouso" else 0
            r = self.potential(s2, on_ground=True) - phi0 - fuel + bonus
        elif terminated:
            r = self.potential(s2) - phi0 - fuel - 100
        else:
            r = self.potential(s2) - phi0 - fuel
        truncated = (not terminated) and self.t >= MAX_STEPS
        if truncated:
            outcome = "tempo"
        return s2, float(r), terminated, truncated, outcome
