# evodm

<!-- badges -->
[![unit tests](https://github.com/DavisWeaver/evo_dm/actions/workflows/tests.yml/badge.svg)](https://github.com/DavisWeaver/evo_dm/actions)
[![codecov](https://codecov.io/gh/DavisWeaver/evo_dm/branch/main/graph/badge.svg?token=ET8DJP3FI7)](https://codecov.io/gh/DavisWeaver/evo_dm)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
<!-- badges end -->

**evodm** (Evolutionary Drug Management) is an RL-based framework for discovering optimal drug scheduling policies that control bacterial and carcinomic populations. It bundles classical MDP solvers, deep RL learners, landscape generators, and experiment utilities so you can reproduce published benchmarks or plug in your own models.

> Original Authors: Davis Weaver and Jeff Maltas  
> V2 Maintainer: Chaaranath Badrinath

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Running Experiments](#running-experiments)
- [Testing](#testing)
- [FAQ](#faq)
- [Citation](#citation)
- [License](#license)

## Features

- **Deterministic & stochastic environments** via `evodm.evol_game` and `evodm.dpsolve`
- **Classical RL**: dynamic programming solvers, policy iteration, and heuristics
- **Deep RL**: Tiankou + Tianshou integrations for modern policy learning
- **Landscape utilities**: generation, normalization, selectivity estimation, and visualization helpers
- **Experiment harnesses**: reproducible sweeps, Mira benchmarks, seascape simulations, and logging utilities
- **Research-friendly tooling**: notebooks, scripts, and data directories scaffolded for typical ML experimentation

## Project Structure

```
evo_dm/
├── evodm/               # Main package (environments, learners, utilities)
├── tests/               # Pytest test suite with shared fixtures
├── docs/                # Documentation entry point
├── scripts/             # Helper scripts (e.g., import organization)
├── notebooks/           # Jupyter notebooks for exploration
├── data/                # data/raw & data/processed (kept empty via .gitkeep)
├── results/             # Experiment outputs (gitignored except .gitkeep)
├── pyproject.toml       # Packaging + tooling configuration
├── uv.lock              # uv dependency lockfile
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9+  
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

Install uv (macOS/Linux):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Dependencies

```bash
# Clone the repo
git clone https://github.com/DavisWeaver/evo_dm.git
cd evo_dm

# Create virtual environment (optional but recommended)
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install project in editable mode with dev deps
uv pip install -e ".[dev]"
```

## Development Workflow

### Format and Organize Imports

```bash
# Sort imports across evodm/, tests/, scripts/
uv run ruff check --select I --fix evodm tests scripts
```

### Run Static Checks (optional)

```bash
uv run ruff check evodm tests
uv run mypy evodm
```

## Running Experiments

Example commands (see `evodm/examples/` and `scripts/` for more):

```bash
# Mira MDP example
uv run python -m evodm.examples.mira_mdp

# Tianshou training harness
uv run python -m evodm.tianshou_learner
```

You can store logs in `results/` and raw datasets in `data/raw/`. Processed artifacts can live in `data/processed/`.

## Testing

```bash
uv run pytest
```

## FAQ

**Q: Why uv?**  
uv provides deterministic dependency resolution and fast installs. The generated `uv.lock` keeps builds reproducible.

**Q: Are `__pycache__` directories safe to delete?**  
Yes. They are regenerated automatically and are gitignored.

**Q: Where do notebooks live?**  
Put exploratory notebooks in `notebooks/`. They are ignored by default except for documentation.

## Citation

If you use this project in academic work, please cite the original authors:

```
Davis Weaver, Jeff Maltas. "evodm: Evolutionary Drug Management."
```

## License

Distributed under the GNU GPL v3. See `LICENSE.md` for details.

