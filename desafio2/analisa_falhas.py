"""Onde a política ótima falha com vento: rajadas nos últimos passos antes do fim."""
import collections, copy, json
import numpy as np
import grid_lander as G

pi = np.load("pi_opt_vento10.npy")
env = G.GridLunarLander(5, wind_prob=0.1)
fails = collections.Counter()
gust_before_end = collections.Counter()
n_eps = 5000
for _ in range(n_eps):
    s = env.reset()
    gust_steps, t = [], 0
    while True:
        a = int(pi[env.index(s)])
        ghost = copy.copy(env); ghost.wind_prob = 0.0; ghost.rng = np.random.default_rng(0)
        pred, *_ = ghost.step(a)
        s, r, te, tr, o = env.step(a)
        t += 1
        if (s[0], s[2]) != (pred[0], pred[2]):   # x ou vx diferente do previsto = houve rajada
            gust_steps.append(t)
        if te or tr:
            break
    if o != "pouso":
        fails[o] += 1
        last = t - gust_steps[-1] if gust_steps else None
        gust_before_end[last if last is not None and last <= 2 else ("nenhuma" if last is None else "3+")] += 1
tot = sum(fails.values())
res = dict(episodios=n_eps, falhas=tot, taxa_falha=tot / n_eps, por_tipo=dict(fails),
           ultima_rajada_passos_antes_do_fim={str(k): v for k, v in gust_before_end.items()})
print(json.dumps(res, ensure_ascii=False, indent=1))
json.dump(res, open("results_vento10_falhas_otimo.json", "w"), ensure_ascii=False, indent=1)
