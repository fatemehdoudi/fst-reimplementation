# FST Reimplementation

Unofficial implementation of the FST paper.

This repository contains an independent implementation based on the
paper description. It is **not** the official implementation released by
the authors.

## Overview

This project implements the FST approach for improving reasoning
performance on graph path-finding tasks.

The repository includes:

-   Star-graph benchmark generation
-   Training pipeline
-   Reinforcement learning components
-   Prompt optimization / evolution components
-   Evaluation utilities
-   Experiment scripts

## Repository Structure

    .
    ├── data/
    │   └── star_graph.py        # Star-graph benchmark generation
    ├── fast/
    │   └── gepa_wrapper.py      # Prompt optimization utilities
    ├── rl/
    │   ├── advantages.py        # Advantage computation
    │   ├── cispo.py             # RL optimization component
    │   └── rollout.py           # Rollout utilities
    ├── tests/                   # Unit tests
    ├── slurm/                   # Cluster execution scripts
    ├── config.py                # Configuration
    ├── trainer.py               # Main training entry point
    ├── scoring.py               # Evaluation/scoring functions
    └── requirements.txt         # Dependencies

## Installation

Create an environment and install dependencies:

``` bash
pip install -r requirements.txt
```

## Dataset

The implementation includes the star-graph benchmark described in
Appendix C.

The default configuration:

-   Source node degree: `d = 25`
-   Path length: `p = 20`
-   Node pool size: `n = 500`
-   Training examples: `10,000`
-   Test examples: `200`

The dataset is generated procedurally using:

    data/star_graph.py

## Running Experiments

Example:

``` bash
python trainer.py
```

For cluster experiments, SLURM scripts are provided:

``` bash
sbatch slurm/env.sh
sbatch slurm/smoke.sh
```

## Testing

Run unit tests with:

``` bash
pytest tests/
```

## Notes

-   This is a research reimplementation and may differ from the original
    authors' implementation.
-   Results may not exactly match the paper due to differences in
    training infrastructure, randomness, and implementation details.

## Citation

If you use this repository, please cite the original FST paper:

    [Add paper citation here]

## License

This repository is intended for research and educational purposes.
