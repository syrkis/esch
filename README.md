# esch

Create beautiful presentations and visualizations inspired by Escher.

## Components

1. Typst presentation template
2. CLI tool for generating presentations from Typst files
3. Python package for Escher-inspired numpy array visualizations

## Installation

```bash
npm install -g esch
pip install esch
```

## Usage

### Generate presentation

```bash
esch presentation input.typ
```

### Create visualization

```python
import esch
from jax import random


rng = random.PRNGKey(0)
data = random.normal(rng, (100, 37, 37))
esch.hinton(data).saveas('hinton.svg')  # save a svg animation with 100 frames (each frame is a 37x37 hinton-like plot)
```

## Development

1. Clone the repo
2. Install dependencies: `npm install` and `poetry install`

## License

MIT

For detailed documentation, visit [our docs](https://github.com/syrkis/esch/wiki).
