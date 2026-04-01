# Flow Matching

COMP4680/8650: Advanced Topics in Machine Learning

See the assignment PDF for the full specification.

## Repository Structure

```
.
├── src/
│   └── dataloader.py     # Loads projected toy data
├── pyproject.toml        # Dependencies
├── README.md             # This file
└── data/                 # Created after download (not tracked)
```

Everything else (model, training loop, diffusion logic, sampling, evaluation scripts) will be implemented by yourself. See the assignment PDF for the specification.

## Quick Start

```bash
# install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# install dependencies
uv sync

# download data
uv run hf download xingjianleng/toy-data --local-dir data --repo-type dataset
```

## References

- [Back to Basics (JiT)](https://arxiv.org/abs/2511.13720): prediction parameterization in flow matching
- [RAE](https://arxiv.org/abs/2510.11690): parameterization and dimension
- [MeanFlow](https://arxiv.org/abs/2505.13447): one-step generation
- [DiT](https://github.com/facebookresearch/DiT): sinusoidal time embedding reference
- [SiT](https://github.com/willisma/SiT): Euler ODE sampling reference
