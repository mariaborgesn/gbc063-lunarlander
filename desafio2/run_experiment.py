"""Roda o experimento do Desafio 2 e salva tudo em results/.

    python run_experiment.py                    # DQN em PyTorch
    python run_experiment.py --backend numpy    # DQN em NumPy (sem PyTorch)
    python run_experiment.py --wind 0.1 --out results_vento10   # com vento de 10%
      (antes: python value_iteration.py --wind 0.1)
"""
import argparse
import collections
import json
import os
import time

import numpy as np

from agents import DQN, TabularQ, evaluate, landed_rate, train
from grid_lander import N_STATES, GridLunarLander

p = argparse.ArgumentParser()
p.add_argument("--episodes", type=int, default=20_000)
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
p.add_argument("--n-final", type=int, default=1000)
p.add_argument("--out", default="results")
p.add_argument("--wind", type=float, default=0.0,
               help="probabilidade de rajada de vento por passo (ex.: 0.1)")
p.add_argument("--pi", default=None, help="política ótima (padrão: pi_opt.npy ou pi_opt_ventoXX.npy)")
p.add_argument("--backend", choices=["torch", "numpy"], default="torch",
               help="DQN em PyTorch (padrão) ou em NumPy puro")
args = p.parse_args()
import grid_lander
grid_lander.DEFAULT_WIND = args.wind
pi_file = args.pi or ("pi_opt.npy" if args.wind == 0 else f"pi_opt_vento{round(100 * args.wind)}.npy")
os.makedirs(args.out, exist_ok=True)


def final_stats(policy):
    rets, outs, lens = evaluate(policy, args.n_final, seed=999)
    c = collections.Counter(outs)
    return rets, dict(mean=float(rets.mean()), std=float(rets.std()),
                      landed=landed_rate(outs), outcomes=dict(c),
                      mean_len=float(lens.mean()))


summary = dict(episodes=args.episodes, seeds=args.seeds, n_states=N_STATES, wind=args.wind)

rng = np.random.default_rng(0)
rets, summary["random"] = final_stats(lambda s: int(rng.integers(4)))
np.save(f"{args.out}/random_final.npy", rets)

pi = np.load(pi_file)
rets, summary["optimal"] = final_stats(lambda s: int(pi[GridLunarLander.index(s)]))
np.save(f"{args.out}/optimal_final.npy", rets)

if args.backend == "torch":
    from dqn_torch import DQNTorch as DQNImpl
else:
    DQNImpl = DQN
summary["dqn_backend"] = args.backend

for name, cls in [("tabular", TabularQ), ("dqn", DQNImpl)]:
    runs = []
    for seed in args.seeds:
        t0 = time.time()
        agent = cls(seed)
        ep_ret, curve = train(agent, args.episodes, seed, eps_frac=0.5)
        el = time.time() - t0
        rets, st = final_stats(agent.greedy)
        st.update(seed=seed, train_seconds=el)
        if name == "tabular":
            st["visited_states"] = agent.visited()
        else:
            st["env_steps"] = agent.steps
            np.save(f"{args.out}/dqn_seed{seed}_loss.npy", np.array(agent.losses))
        np.save(f"{args.out}/{name}_seed{seed}_episodes.npy", ep_ret)
        np.save(f"{args.out}/{name}_seed{seed}_curve.npy", curve)
        np.save(f"{args.out}/{name}_seed{seed}_final.npy", rets)
        runs.append(st)
        print(f"{name} seed {seed}: média {st['mean']:.1f}  pouso {100*st['landed']:.1f}%  ({el:.0f}s)",
              flush=True)
        if name == "dqn":
            W, b = agent.weights() if hasattr(agent, "weights") else (agent.net.W, agent.net.b)
            np.savez(f"{args.out}/dqn_seed{seed}_weights.npz", *W, *b)
    summary[name] = runs

with open(f"{args.out}/summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(json.dumps({k: v for k, v in summary.items() if k in ("random", "optimal")}, indent=1))
