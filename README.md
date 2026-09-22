# Machine Learning Evolution Equations

Python code and computational experiments accompanying **_Machine Learning Evolution Equations_** by Joseph Shomberg (De Gruyter).

This repository is the computational companion to the book. It connects the mathematical development of evolution equations with reproducible numerical simulation, dataset generation, inverse learning, physics-informed WGAN-GP models, and statistical model evaluation.

The examples progress from elementary time stepping to nonlinear PDE solvers and learned inverse maps. They include the Chafee-Infante reaction-diffusion equation, ternary Cahn-Hilliard dynamics, Allen-Cahn, viscous Burgers, Navier-Stokes, Kuramoto-Sivashinsky, nonlinear Schrodinger, and thin-film equations.

> **Companion guide:** [`MLEE_Code_Descriptions.pdf`](MLEE_Code_Descriptions.pdf) gives the book section, mathematical model, numerical method, and purpose of each major program.

## What is included

- Explicit, implicit, semi-implicit, finite-difference, and spectral solvers
- Smooth, random, and spectrally engineered initial conditions
- Dataset pipelines for supervised PDE inversion
- U-Net generators and PatchGAN critics trained with WGAN-GP
- Physics-informed losses based on energy, short-horizon dynamics, mass, statistics, and interfacial structure
- Full-test evaluation with MAE, residuals, empirical probabilities, Wilson confidence intervals, and qualitative comparisons
- Publication-oriented figures and diagnostic output

## Repository map

### Chapter 3 - Numerical evolution and data generation

| Directory | Model or task | Main method |
| --- | --- | --- |
| [`3.1-Euler_Introduction`](3.1-Euler_Introduction/) | Nonlinear scalar ODE near a singularity | Forward Euler |
| [`3.2-Euler_Cancer-Immune_I`](3.2-Euler_Cancer-Immune_I/) | Cancer-immune system, first example | Forward Euler |
| [`3.3-Euler_Cancer-Immune_II`](3.3-Euler_Cancer-Immune_II/) | Cancer-immune parameter sweep | Forward Euler and reference integration |
| [`3.4-Poisson-smooth-initial-condition`](3.4-Poisson-smooth-initial-condition/) | Smooth Poisson initial data | Sparse finite differences |
| [`3.5.a-CIRDE-simulation-smooth-initial-data`](3.5.a-CIRDE-simulation-smooth-initial-data/) | Chafee-Infante equation with smooth data | Semi-implicit Eyre splitting and Newton iteration |
| [`3.5.b-CIRDE-smooth-dataset-generation`](3.5.b-CIRDE-smooth-dataset-generation/) | Legacy batched CIRDE dataset tools | Poisson data, forward evolution, NPZ merge/check |
| [`3.5.c-CIRDE-smooth-dataset`](3.5.c-CIRDE-smooth-dataset/) | Sample smooth initial condition | Reference image |
| [`3.6-random-initial-condition`](3.6-random-initial-condition/) | Uniformly random initial data | Random field with Dirichlet boundary data |
| [`3.7-CIRDE-simulation-uniformly-random-initial-condition`](3.7-CIRDE-simulation-uniformly-random-initial-condition/) | Chafee-Infante equation with random data | Semi-implicit finite differences |
| [`3.8-CIRDE-smooth-dataset-generation`](3.8-CIRDE-smooth-dataset-generation/) | Unified CIRDE inverse-pair dataset pipeline | Batched generation, merge, and validation |
| [`3.9-CIRDE-simulation-high-frequence-suppression`](3.9-CIRDE-simulation-high-frequence-suppression/) | Spectral suppression experiment | Mixed Laplacian eigenmodes and forward evolution |

For the inverse CIRDE datasets, the stored pair follows

```text
src = S^N(u0)   evolved/terminal state
tar = u0        initial state
```

### Chapter 4 - Learned inverse solutions

| Directory | Purpose |
| --- | --- |
| [`4.8.a-CIRDE-WGANGP-physics-informed-smooth-initial-data`](4.8.a-CIRDE-WGANGP-physics-informed-smooth-initial-data/) | Train a physics-informed conditional WGAN-GP for the inverse Chafee-Infante problem |
| [`4.8.b-CIRDE-WGANGP-smooth-testing`](4.8.b-CIRDE-WGANGP-smooth-testing/) | Evaluate saved inverse models on the full test set and generate reports, plots, and probabilistic solution metrics |

The generator maps a near-equilibrium observation back to a candidate initial state. The training objective combines adversarial learning with reconstruction and physically motivated penalties. The evaluation code compares models selected by reconstruction error and by dynamical residual.

### Chapter 5 - Additional evolution equations

| Directory | Model or task | Main method |
| --- | --- | --- |
| [`5.4-AC-simulation`](5.4-AC-simulation/) | Two-dimensional Allen-Cahn equation | Periodic finite differences and IMEX Euler |
| [`5.6.a-tCHE-simulation-Eyre-noisy-initial-data`](5.6.a-tCHE-simulation-Eyre-noisy-initial-data/) | Ternary Cahn-Hilliard simulation | Eyre splitting, fixed-point solve, and simplex projection |
| [`5.6.b-tCHE-Eyre-noisy-dataset-generation`](5.6.b-tCHE-Eyre-noisy-dataset-generation/) | Complete tCHE dataset pipeline | Projected forward evolution, batching, merge, validation |
| [`5.6.c-WGAN-tCHE-physics-informed-Eyer-initial-data`](5.6.c-WGAN-tCHE-physics-informed-Eyer-initial-data/) | Physics-informed tCHE inverse training | Conditional WGAN-GP with physical and statistical losses |
| [`5.6.d-Evaluation-tCHE-models`](5.6.d-Evaluation-tCHE-models/) | Compare tCHE inverse models | Full-test error distributions and ML-solution statistics |
| [`5.7-VBE-simulation`](5.7-VBE-simulation/) | One-dimensional viscous Burgers equation | Rusanov flux and exact Fourier diffusion |
| [`5.9-NSE-simulation`](5.9-NSE-simulation/) | Two-dimensional incompressible Navier-Stokes | Fourier pseudo-spectral projection method |
| [`5.10-KSE-simulation`](5.10-KSE-simulation/) | Kuramoto-Sivashinsky equation | Spectral space discretization and semi-implicit stepping |
| [`5.13-NLS-simulation`](5.13-NLS-simulation/) | Cubic nonlinear Schrodinger equation | Strang split-step Fourier method |
| [`5.14-Thin-film-simulation`](5.14-Thin-film-simulation/) | Two-dimensional thin-film/dewetting dynamics | De-aliased Fourier pseudo-spectral method |

The tCHE datasets use channel-first arrays and the inverse convention

```text
src = c(T)   evolved ternary state
tar = c(0)   initial ternary state
shape = (samples, 3, M, M)
```

### Appendix

[`A.2.2-Activation-functions`](A.2.2-Activation-functions/) generates the ReLU, sigmoid, tanh, and custom activation-function figures used in the text.

## Installation

Python 3.10 or newer is recommended. Create a virtual environment and install the common numerical dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib
```

The machine-learning programs additionally require TensorFlow/Keras:

```bash
python -m pip install tensorflow
```

Some training scripts call `tensorflow.keras.utils.plot_model`; installing `pydot` and Graphviz enables model-diagram output:

```bash
python -m pip install pydot
```

TensorFlow installation differs across operating systems and accelerators. Consult the official TensorFlow instructions if GPU or Apple-silicon acceleration is required.

## Running an example

Run each program from its own directory. Many scripts use relative paths and write images, CSV files, checkpoints, or datasets into the current working directory.

```bash
cd 3.1-Euler_Introduction
python 3.1-Euler_Introduction.py
```

For a PDE example:

```bash
cd 5.7-VBE-simulation
python 5.7-VBE-simulation.py
```

Most simulation parameters are grouped near the top of each file. Full-resolution simulations and dataset generators can be computationally expensive, so a reduced grid, sample count, or number of time steps is advisable for an initial test.

## CIRDE inverse-learning workflow

The intended Chafee-Infante workflow is:

1. Inspect the forward dynamics with the Chapter 3 simulation programs.
2. Generate training and testing `.npz` files with `3.8-CIRDE-smooth-dataset-generation.py`.
3. Set `TRAIN_NPZ` and `TEST_NPZ` near the top of the Chapter 4 training script.
4. Train the WGAN-GP model from its directory.
5. Point the evaluation script to the datasets and saved models, then run it to produce the full-test report and figures.

```bash
cd 4.8.a-CIRDE-WGANGP-physics-informed-smooth-initial-data
python 4.8.a-CIRDE-WGANGP-physics-informed-smooth-initial-data.py
```

The training script currently contains placeholder dataset paths by design. Set those paths before running it. Its outputs are organized under `training_output/`, including logs, checkpoints, and saved best-MAE and best-residual models.

The CIRDE evaluation script also contains configurable dataset and model paths near the beginning of the file. Update them for the local experiment before execution.

## Ternary Cahn-Hilliard inverse-learning workflow

The corresponding tCHE workflow is:

1. Run `5.6.a-tCHE-simulation-Eyre-noisy-initial-data.py` to inspect the forward solver and diagnostics.
2. Configure `START`, `END`, `BATCH_SIZE`, `NSTEPS`, and the physical parameters in `5.6.b-tCHE-Eyre-noisy-dataset-generation.py`.
3. Generate separate training and testing datasets.
4. Set the dataset paths in `5.6.c-WGAN-tCHE-physics-informed-Eyer-initial-data.py` and train the inverse model.
5. Evaluate the saved best-MAE and best-residual generators with the command-line evaluation program.

Example evaluation command:

```bash
cd 5.6.d-Evaluation-tCHE-models
python evaluate_two_inverse_models_tche_BOOK_with_ML_solution_scatter.py \
  --train-npz /path/to/train-dataset.npz \
  --test-npz /path/to/test-dataset.npz \
  --model-dir /path/to/saved-models \
  --outdir full_model_evaluation_tche \
  --residual-steps 200 \
  --max-test-samples 10000
```

Use `python evaluate_two_inverse_models_tche_BOOK_with_ML_solution_scatter.py --help` for all available arguments.

## Data and trained models

Large generated datasets, complete training runs, and model checkpoints are not stored in this source repository. They are produced locally by the dataset and training programs. Before running a machine-learning script, verify:

- the training and testing `.npz` paths;
- the `src` and `tar` keys and expected array layout;
- the grid size and physical parameters;
- the residual horizon used for training and evaluation;
- the output and checkpoint directories.

Do not mix datasets or checkpoints created with different grid sizes, time steps, physical parameters, or forward horizons unless the experiment is explicitly designed to study that mismatch.

## Reproducibility notes

- Scripts preserve the parameter choices used for the corresponding book experiments where practical.
- Random seeds are fixed in many programs; confirm the seed before comparing independent runs.
- Numerical experiments may create many high-resolution images and large intermediate arrays.
- The reported machine-learning results require the same datasets, hyperparameters, stopping rules, residual horizon, and model-selection criterion used in the book.
- The programs favor transparency and correspondence with the mathematics over packaging as a general-purpose solver library.

## Citation

If these programs contribute to published work, please cite the accompanying book:

```bibtex
@book{shomberg_mlee,
  author    = {Joseph L. Shomberg},
  title     = {Machine Learning Evolution Equations},
  publisher = {De Gruyter},
  note      = {Computational companion available at
               https://github.com/josephshomberg/Machine_Learning_Evolution_Equations}
}
```

Update the entry with the final publication year, edition, and DOI when available.

## License

This repository is distributed under the terms of the [`LICENSE`](LICENSE) file.
