# MachineLearning

A personal open-source learning repository for machine learning, deep learning, signal processing, applied math, and computational modeling.

This project is organized as a practical notebook-style codebase with multiple independent topic folders.

## Repository Scope

This repository includes:
- learning notes and runnable scripts
- algorithm practice and experiments
- small project demos across ML/DL and related engineering topics

This repository does **not** include private datasets.

## Dataset & Privacy Policy

To protect private data, dataset-like files are not published in this open-source repository.

Restricted private data file types:
- `.jpg`, `.jpeg`, `.png`, `.wav`, `.mat`, `.csv`

If you run code that depends on data, prepare your own local dataset paths and keep them out of Git tracking.

## Main Directory Layout

- `Acoustic/` - speech/audio feature and model experiments
- `Computational_Physics/` - ODE/PDE and numerical simulation code
- `DeepLearning/` - neural network experiments (MLP/CNN/RNN/Transformer/CV)
- `Digital Signal Processing/` - signal analysis and DSP practice
- `Math/` - probability, statistics, and algorithm notes
- `Practicle_MachineLearning/` - practical ML exercises
- `python_elementary/` - Python/Numpy/Pandas/Matplotlib fundamentals
- `The-common-algorithm-in-MCM-ICM/` - common algorithm implementations
- `z_scripts/` - utility scripts and helper tools

## Quick Start

1. Clone the repository.
2. Create and activate a Python environment.
3. Install required packages for the specific subproject you want to run.
4. Run scripts from the corresponding module directory.

Example:

```bash
git clone https://github.com/PlankRiver/MachineLearning.git
cd MachineLearning
python -m venv .venv
source .venv/bin/activate
```

## Notes

- Different folders may have different dependency versions.
- This is a learning-oriented monorepo, so code style and structure can vary by topic.

## License

This project is released under the MIT License. See [LICENSE](./LICENSE).
