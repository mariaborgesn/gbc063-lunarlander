# Desafio 1: LunarLander

- **Disciplina:** GBC063 – Inteligência Artificial (UFU/FACOM)
- **Dupla:** Maria Eduarda Borges e Felipe Morotti
- **Apresentação:** quinta-feira, 29/10/2026

---

## 1. O que o desafio pede

> Fazer uma apresentação de até 3 slides com a descrição do problema da dupla, contendo:
> 1. o ambiente e a descrição do estado markoviano, ou seja, quais informações do ambiente estão disponíveis para o agente;
> 2. as ações do agente;
> 3. as recompensas.
>
> A apresentação será feita em sala na quinta, 29/10/2026. É necessário criar um repositório git próprio e dar acesso ao professor, para enviar a apresentação e as próximas entregas.

O problema escolhido pela dupla é o **LunarLander-v3**, da biblioteca **Gymnasium**.

---

## 2. A biblioteca Gymnasium

- **O que é:** uma biblioteca em Python com ambientes prontos para aprendizagem por reforço (jogos e simulações em que um agente aprende por tentativa e erro).
- **Quem mantém:** a Farama Foundation. É a continuação da antiga biblioteca *Gym*, da OpenAI.
- **Por que usamos:** é a mesma biblioteca usada nas atividades da disciplina (FrozenLake, Taxi e Space Invaders). Ela já traz o jogo pronto: a física, a tela, os estados, as ações e a pontuação.
- **Versão:** o LunarLander-v3 existe desde o **Gymnasium 1.0**, lançado em 08/10/2024. A versão mais recente, em outubro de 2026, é a **1.4.0** (05/10/2026).
- **Física:** o LunarLander usa o motor de física **Box2D**.

---

## 3. A versão do LunarLander

| Versão | O que mudou |
|---|---|
| v0 | Versão inicial. |
| v1 | Contato das pernas com o solo entrou no estado, junto com a recompensa por tocar o chão. |
| v2 | Contagem de combustível gasto e parâmetro de turbulência. |
| **v3** (atual, a que usamos) | Correção de um erro que deixava a simulação não determinística; o vento passou a ser sorteado de novo a cada partida; limites das observações ficaram mais largos. |

### Parâmetros do ambiente (valores padrão)

| Parâmetro | Padrão | O que faz |
|---|---|---|
| `continuous` | `False` | `False` = 4 ações discretas (botões). `True` = motores com intensidade contínua. |
| `gravity` | `-10.0` | Gravidade (entre 0 e −12). |
| `enable_wind` | `False` | Liga ou desliga o vento. **Usamos desligado.** |
| `wind_power` | `15.0` | Força máxima do vento, se ligado. |
| `turbulence_power` | `1.5` | Força máxima da turbulência (giro), se ligado. |

- **Limite de tempo:** 1000 passos por partida.
- **Critério de sucesso:** uma partida com **200 pontos ou mais** é considerada resolvida.

---

## 4. O problema (resumo dos 3 slides)

### 4.1 Ambiente e estado markoviano

O módulo lunar começa no alto da tela, com um empurrão aleatório, e precisa pousar devagar e nivelado na área de pouso, no ponto (0, 0).

A cada passo, o agente recebe um **estado com 8 números** (s ∈ ℝ⁸):

| # | Variável | Faixa | Significado |
|---|---|---|---|
| 0 | Posição x | −2,5 a 2,5 | Para os lados; 0 = centro, ±1 = bordas da tela |
| 1 | Posição y | −2,5 a 2,5 | Altura; 0 = altura da área de pouso |
| 2 | Velocidade x | −10 a 10 | Movimento para os lados |
| 3 | Velocidade y | −10 a 10 | Movimento vertical; negativo = caindo |
| 4 | Ângulo | −2π a 2π | Inclinação, em radianos; 0 = reta |
| 5 | Velocidade angular | −10 a 10 | Quão rápido está girando |
| 6 | Perna esquerda no solo | 0 ou 1 | 1 = tocando o chão |
| 7 | Perna direita no solo | 0 ou 1 | 1 = tocando o chão |

**Por que é markoviano:** posição, velocidade e ângulo de agora bastam para prever o próximo passo da física, sem precisar lembrar do passado. Por isso o LunarLander é uma escolha de ambiente ideal para o experimento.

**Fim da partida (episódio):** queda, sair da tela (|x| > 1), pousar e ficar parado, ou chegar a 1000 passos.

### 4.2 Ações (Discrete(4))

| Ação | Nome | O que faz | Custo |
|---|---|---|---|
| 0 | Nada | Deixa a gravidade agir | 0 |
| 1 | Motor esquerdo | Gira e desloca o módulo para um lado | −0,03 por passo |
| 2 | Motor principal | Empurra para cima e freia a queda | −0,3 por passo |
| 3 | Motor direito | Gira e desloca para o lado oposto | −0,03 por passo |

A cada passo, o agente escolhe exatamente uma dessas 4 ações.

### 4.3 Recompensas

| Evento | Recompensa |
|---|---|
| Aproximar-se da área de pouso | + (afastar-se: −) |
| Diminuir a velocidade | + (acelerar: −) |
| Inclinar o módulo | − |
| Cada perna tocando o solo | +10 |
| Motor principal ligado | −0,3 por passo |
| Motor lateral ligado | −0,03 por passo |
| Pouso seguro (parado) | +100 (fim) |
| Queda ou sair da tela | −100 (fim) |

- As três primeiras linhas são a variação de um "potencial": r = Φ(s′) − Φ(s). O agente ganha pontos quando a situação melhora em relação ao passo anterior e perde quando piora.
- **Fator de desconto:** γ = 0,99 (escolha da dupla), porque o prêmio maior só vem no fim da partida.
- **Meta:** 200 pontos ou mais por episódio.
- **Risco de recompensa mal projetada:** para fugir do −100, o agente poderia aprender a ficar pairando sem pousar. O custo de combustível e o limite de 1000 passos tiram a vantagem disso.

---

## 5. Fontes

- [Gymnasium – Lunar Lander (documentação oficial)](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [Gymnasium – Notas de versão](https://gymnasium.farama.org/gymnasium_release_notes/)
- [Repositório da disciplina GBC063](https://github.com/albertiniufu/gbc063)
