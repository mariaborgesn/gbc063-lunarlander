"""Política ótima do LunarLander em grade por iteração de valor (aula 2).

Conhece o modelo: para cada estado e ação, enumera os próximos estados
possíveis e suas probabilidades (sem vento: um só; com vento: três).

    python value_iteration.py               # sem vento  -> pi_opt.npy
    python value_iteration.py --wind 0.1    # vento 10%  -> pi_opt_vento10.npy
"""
import argparse
import collections
import itertools

import numpy as np

import grid_lander as G

p = argparse.ArgumentParser()
p.add_argument("--wind", type=float, default=0.0, help="probabilidade de rajada por passo")
p.add_argument("--out", default=None, help="arquivo da política (padrão depende do vento)")
args = p.parse_args()
out = args.out or ("pi_opt.npy" if args.wind == 0 else f"pi_opt_vento{round(100 * args.wind)}.npy")
GAMMA = 0.99


class FixedRng:
    """Substitui o sorteio do ambiente para forçar cada desfecho possível."""
    def __init__(self, vals):
        self.vals = list(vals)

    def random(self):
        return self.vals.pop(0)


# desfechos de um passo: (probabilidade, números que o ambiente vai "sortear")
w = args.wind
branches = [(1.0, [0.999])] if w == 0 else [
    (1 - w, [0.999]),        # sem rajada
    (w / 2, [0.0, 0.2]),     # rajada: vx + 1
    (w / 2, [0.0, 0.8]),     # rajada: vx - 1
]

env = G.GridLunarLander(0, wind_prob=w)
states = list(itertools.product(range(G.X_MIN, G.X_MAX + 1), range(1, G.Y_MAX + 1),
                                range(-G.VX_MAX, G.VX_MAX + 1), range(G.VY_MIN, G.VY_MAX + 1),
                                range(-G.TH_MAX, G.TH_MAX + 1)))
T = {}
for s in states:
    i = G.GridLunarLander.index(s)
    for a in range(G.N_ACTIONS):
        outs = []
        for prob, vals in branches:
            env.x, env.y, env.vx, env.vy, env.th = s
            env.t = 0
            env.rng = FixedRng(vals)
            s2, r, te, tr, o = env.step(a)
            outs.append((prob, r, G.GridLunarLander.index(s2), te))
        T[(i, a)] = outs


def q_values(i, V):
    return [sum(prob * (r + (0 if te else GAMMA * V[j])) for prob, r, j, te in T[(i, a)])
            for a in range(G.N_ACTIONS)]


V = np.zeros(G.N_STATES)
for it in range(1000):
    V2 = V.copy()
    for s in states:
        i = G.GridLunarLander.index(s)
        V2[i] = max(q_values(i, V))
    delta = np.abs(V2 - V).max()
    V = V2
    if delta < 1e-4:
        break

pi = np.zeros(G.N_STATES, dtype=np.int64)
for s in states:
    i = G.GridLunarLander.index(s)
    pi[i] = int(np.argmax(q_values(i, V)))
np.save(out, pi)

env = G.GridLunarLander(5, wind_prob=w)
outcomes, rets = collections.Counter(), []
for _ in range(2000):
    s, g = env.reset(), 0.0
    while True:
        s, r, te, tr, o = env.step(int(pi[env.index(s)]))
        g += r
        if te or tr:
            break
    outcomes[o] += 1
    rets.append(g)
print(f"vento {w:.0%}: {it + 1} iterações · retorno médio {np.mean(rets):.1f} · "
      f"{dict(outcomes)} -> {out}")
