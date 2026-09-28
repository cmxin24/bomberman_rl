# Money-First Phase-Aware DQN Agent

This directory contains the strongest verified DQN variant developed for the
Machine Learning Essentials Bomberman project.

## Contents

- `callbacks.py`: feature extraction, action selection, safety filtering, and
  phase-aware economy/endgame logic.
- `train.py`: Double DQN training with a frozen base network, a residual head,
  five-step returns, replay memory, and a target network.
- `dqn_model_task4_phase.pt`: trained checkpoint used for evaluation.

## Policy overview

The agent uses a 43-dimensional feature representation. It combines learned
Q-values with deterministic legality and survival checks. While crates remain,
it prioritizes safe opportunistic kills, winnable visible coins, useful bombs,
and routes to productive bomb positions. Once all crates are gone, it protects
a score lead and otherwise retains the learned combat policy.

The checkpoint was evaluated against three `rule_based_agent` opponents in the
`classic` scenario. Across seeds 123, 456, 789, and 2026, with 100 rounds per
seed, it achieved:

- strict first place: 170/400 (42.50%);
- first or tied first: 209/400 (52.25%).

## Requirements

- Python 3.11
- NumPy
- PyTorch

## Usage

Copy this directory into the framework's `agent_code` directory, then run:

```bash
python main.py play \
  --my-agent lsk_task4_vnext_agent \
  --scenario classic \
  --seed 123 \
  --n-rounds 1
```

The trained checkpoint is included. Retraining is not required for evaluation.
