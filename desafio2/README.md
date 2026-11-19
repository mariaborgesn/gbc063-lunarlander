# Desafio 2 — DQN no LunarLander em grade

Versão discreta do LunarLander-v3: s = (x, y, vx, vy, θ) inteiros, |S| = 18.150 estados,
as mesmas 4 ações e recompensas do Desafio 1.

- **Cenário principal (slides 1 e 2): sem vento.** Dinâmica determinística; o acaso está
  só no estado inicial (x, vx, θ sorteados), como no v3 sem vento do Desafio 1.
- **Extensão (slide 3): vento de 10%.** Em cada passo, depois da ação, com 10% de chance uma
  rajada muda vx em ±1. A transição fica estocástica; o estado continua markoviano.

## Como rodar

    pip install -r requirements.txt

    # sem vento
    python value_iteration.py                   # política ótima -> pi_opt.npy
    python run_experiment.py                    # aleatório, Q-tabular e DQN (PyTorch) -> results/
    python tab_long.py 0                        # opcional: Q-tabular com 200 mil episódios

    # com vento de 10%
    python value_iteration.py --wind 0.1        # -> pi_opt_vento10.npy
    python run_experiment.py --wind 0.1 --out results_vento10
    python analisa_falhas.py                    # onde o ótimo falha com vento

    python figs.py                              # figuras dos 3 slides

`--backend numpy` roda o mesmo DQN sem PyTorch.

## Resultados (pouso na área; 1.000 episódios de avaliação por semente, 3 sementes)

| Agente | Sem vento | Vento 10% |
|---|---|---|
| Aleatório | 0% | 0% |
| Q-tabular (20 mil ep.) | 67,8% | 47,4% |
| DQN (20 mil ep.) | 99,1% | 93,3% |
| Ótimo (iteração de valor) | 100% | 94,5% |

Com vento, 98% das falhas do ótimo têm uma rajada nos 3 últimos passos: não sobra tempo
para corrigir. O teto vem do ambiente; a distância até ele, do treino.

file:///C:/Users/maria/AppData/Local/Packages/5319275A.WhatsAppDesktop_cv1g1gvanyjgm/LocalState/sessions/32BF0B7D322956AE4D676B5A4ACF00F29EC6829F/transfers/2026-47/Desafio%202%20%C2%B7%20LunarLander%20em%20grade%20rodando.html
