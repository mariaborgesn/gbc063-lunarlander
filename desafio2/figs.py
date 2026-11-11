"""Figuras dos slides do Desafio 2 (tema escuro, igual ao deck do Desafio 1)."""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from agents import FEAT, MLP
from grid_lander import N_ACTIONS, N_FEAT, GridLunarLander, X_MAX, X_MIN, Y_MAX

BG = "#0d1323"
INK, INK2, GRID = "#f0efec", "#c3c2b7", "#2a3550"
C_DQN, C_TAB, C_AQ, C_YL, C_GR = "#3987e5", "#d95926", "#199e70", "#c98500", "#6b7280"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "figure.facecolor": BG,
    "axes.facecolor": BG, "savefig.facecolor": BG, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "text.color": INK,
})
R = "results"
S = json.load(open(f"{R}/summary.json"))


def smooth(y, k=5):
    return np.convolve(np.pad(y, (k // 2, k // 2), mode="edge"), np.ones(k) / k, "valid")


def band(ax, files, color, label, xscale=1000, k=5, ls="-", ytxt=None):
    C = [np.load(f) for f in sorted(glob.glob(files))]
    n = min(len(c) for c in C)
    x = C[0][:n, 0] / xscale
    Y = np.array([smooth(c[:n, 1], k) for c in C])
    ax.fill_between(x, Y.min(0), Y.max(0), color=color, alpha=0.18, lw=0)
    ax.plot(x, Y.mean(0), color=color, lw=2.2, ls=ls)
    return x, Y.mean(0)


# --------------------------------------------------------------- curva
fig, ax = plt.subplots(figsize=(7.4, 4.3))
x, y = band(ax, f"{R}/dqn_seed*_curve.npy", C_DQN, "DQN")
ax.text(x[-1], y[-1] - 34, "DQN", color=INK, ha="right", fontsize=12, fontweight="bold")
x, y = band(ax, f"{R}/tabular_seed*_curve.npy", C_TAB, "Q-tabular")
ax.text(x[-1], y[-1] - 30, "Q-tabular", color=INK, ha="right", fontsize=12, fontweight="bold")
opt, rnd = S["optimal"]["mean"], S["random"]["mean"]
ax.axhline(opt, color=INK2, lw=1.2, ls=(0, (1, 2)))
ax.text(0.2, opt + 8, f"ótimo (iteração de valor) · {opt:.0f}", color=INK2, fontsize=10)
ax.axhline(rnd, color=C_GR, lw=1.4, ls=(0, (4, 3)))
ax.text(0.2, rnd + 8, f"aleatório · {rnd:.0f}", color=INK2, fontsize=10)
ax.set_xlabel("episódios de treino (mil)")
ax.set_ylabel("retorno médio (política gulosa)")
ax.set_xlim(0, 20)
ax.set_ylim(-215, 330)
fig.tight_layout()
fig.savefig(f"{R}/fig_curva.png", dpi=220)
plt.close(fig)

# --------------------------------------------------------------- desfechos
rows = [("Aleatório", [S["random"]["outcomes"]]),
        ("Q-tabular", [r["outcomes"] for r in S["tabular"]]),
        ("DQN", [r["outcomes"] for r in S["dqn"]]),
        ("Ótimo*", [S["optimal"]["outcomes"]])]
cats = [("pouso", "pouso na área", C_DQN), ("pouso fora", "pouso fora", C_AQ),
        ("queda", "queda", C_TAB), ("saiu", "saiu da tela", C_YL), ("tempo", "tempo", C_GR)]
fig, ax = plt.subplots(figsize=(7.4, 3.3))
for i, (name, outs) in enumerate(rows[::-1]):
    tot = sum(sum(o.values()) for o in outs)
    left = 0
    for key, lab, c in cats:
        v = 100 * sum(o.get(key, 0) for o in outs) / tot
        if v <= 0:
            continue
        ax.barh(i, v - 0.4, left=left + 0.2, color=c, height=0.62)
        if v >= 7:
            ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center",
                    color="#ffffff", fontsize=10, fontweight="bold")
        left += v
ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], color=INK, fontsize=12)
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
ax.grid(axis="y", visible=False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
handles = [Rectangle((0, 0), 1, 1, color=c) for _, _, c in cats]
ax.legend(handles, [l for _, l, _ in cats], ncol=5, loc="lower center",
          bbox_to_anchor=(0.45, 1.0), frameon=False, fontsize=10, handlelength=1,
          columnspacing=1.0, labelcolor=INK2)
fig.tight_layout()
fig.savefig(f"{R}/fig_desfechos.png", dpi=220)
plt.close(fig)

# --------------------------------------------------------------- trajetórias
best = max(S["dqn"], key=lambda r: r["mean"])["seed"]
w = np.load(f"{R}/dqn_seed{best}_weights.npz")
arrs = [w[f"arr_{i}"] for i in range(len(w.files))]
net = MLP([N_FEAT, 64, 64, N_ACTIONS], np.random.default_rng(0))
net.W, net.b = arrs[:3], arrs[3:]
pol = lambda s: int(net.forward(FEAT(s)[None])[0].argmax())

fig, ax = plt.subplots(figsize=(4.6, 4.3))
ax.add_patch(Rectangle((-1.5, -0.5), 3, 0.5, color=C_DQN, alpha=0.35, lw=0))
ax.text(1.7, -0.25, "área de pouso", ha="left", va="center", fontsize=9, color=INK)
env = GridLunarLander(7)
best_path = None
for _ in range(400):
    s = env.reset()
    if s[0] > -3:
        continue
    path, acts = [s], []
    while True:
        a = pol(s)
        s, r, te, tr, o = env.step(a)
        path.append(s); acts.append(a)
        if te or tr:
            break
    if o == "pouso" and (best_path is None or len(path) < len(best_path[0])):
        best_path = (path, acts)
path, acts = best_path
P = np.array(path)
ax.plot(P[:, 0], P[:, 1], color=C_YL, lw=2.2, zorder=3)
mk = {0: ("o", "nada"), 1: ("<", "motor esq."), 2: ("^", "motor principal"), 3: (">", "motor dir.")}
for a, (m, lab) in mk.items():
    idx = [i for i, aa in enumerate(acts) if aa == a]
    if idx:
        ax.scatter(P[idx, 0], P[idx, 1], marker=m, s=110, color=INK, edgecolor=BG, lw=1.5,
                   zorder=4, label=lab)
ax.scatter([P[-1, 0]], [P[-1, 1]], marker="*", s=260, color=C_DQN, edgecolor=INK, zorder=5,
           label="pousou")
lab = {}
for i, p in enumerate(path):
    lab.setdefault((p[0], p[1]), []).append(str(i))
for (px, py), ids in lab.items():
    ax.text(px + 0.28, py + 0.15, ",".join(ids), fontsize=8, color=INK2)
ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK2,
          title="ação em cada passo", title_fontsize=9)
traj_info = dict(steps=len(acts), start=list(map(int, path[0])))
ax.set_xlim(X_MIN - 0.5, X_MAX + 0.5)
ax.set_ylim(-0.6, Y_MAX + 0.6)
ax.set_xticks(range(X_MIN, X_MAX + 1, 1))
ax.set_yticks(range(0, Y_MAX + 1, 1))
ax.tick_params(labelsize=9)
ax.set_xlabel("x (coluna)")
ax.set_ylabel("y (altura)")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(f"{R}/fig_trajetorias.png", dpi=220)
plt.close(fig)
print("ok", best, traj_info)

# --------------------------------------------------------------- vento (slide 3)
import os
import sys

def landed_pct(summary, key):
    v = summary[key]
    if isinstance(v, list):
        return 100 * float(np.mean([r["landed"] for r in v]))
    return 100 * v["landed"]

no_wind_path = sys.argv[1] if len(sys.argv) > 1 else f"{R}/summary.json"
wind_path = sys.argv[2] if len(sys.argv) > 2 else "results_vento10/summary.json"
if os.path.exists(wind_path):
    S0, S1 = json.load(open(no_wind_path)), json.load(open(wind_path))
    names = [("Aleatório", "random"), ("Q-tabular", "tabular"), ("DQN", "dqn"), ("Ótimo*", "optimal")]
    v0 = [landed_pct(S0, k) for _, k in names]
    v1 = [landed_pct(S1, k) for _, k in names]
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    xs = np.arange(len(names))
    wdt = 0.36
    C0, C1 = "#6b8fc9", C_TAB
    b0 = ax.bar(xs - wdt / 2 - 0.01, v0, wdt, color=C_DQN, label="sem vento")
    b1 = ax.bar(xs + wdt / 2 + 0.01, v1, wdt, color=C_TAB, label=f"vento {S1['wind']:.0%}".replace(".", ","),
                hatch="//", edgecolor=BG, lw=0)
    for b, v in list(zip(b0, v0)) + list(zip(b1, v1)):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, (f"{v:.1f}".rstrip("0").rstrip(".").replace(".", ",") + "%"), ha="center", va="bottom",
                color=INK, fontsize=11, fontweight="bold")
    ax.set_xticks(xs, [n for n, _ in names], color=INK, fontsize=12)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.set_ylabel("pousos na área")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", frameon=False, labelcolor=INK2, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(wind_path), "fig_vento.png"), dpi=220)
    plt.close(fig)
    print("fig_vento ok")
