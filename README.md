# SSM-DNN-JAX

**A JAX implementation of SSM-DNN: a deep-network decoder with an embedded state-space layer, trained with particle filtering and EM, for small and noisy biobehavioral time series.**

[![Paper](https://img.shields.io/badge/IEEE%20EMBC-2025-blue)](https://doi.org/10.1109/EMBC58623.2025.11254746)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](License)
![JAX](https://img.shields.io/badge/JAX-Flax%20%7C%20Optax-orange)

Deep networks struggle on neuroscience and clinical data: datasets are small, and the signals are noisy and non-stationary. SSM-DNN addresses this by placing a **state-space model (SSM)** in front of a **deep neural network (DNN)**. The SSM infers a smooth latent trajectory from noisy observations, and the DNN decodes the participant's label from that trajectory. The two parts are trained jointly with an **Expectation–Maximization (EM)** loop driven by **sequential Monte Carlo (particle filtering)**.

This repository is a JAX rewrite of the method from our IEEE EMBC 2025 paper ([original repository](https://github.com/Pedram-Rajaei/SSM-DNN)).

---

## How it works

```
noisy reaction times y_1:T
          │
          ▼
┌───────────────────────────┐   E-step: particle filter
│ State-space model         │     (jax.lax.scan over time,
│ x_t = A x_{t-1} + B + w_t │      jax.vmap over particles and trials)
│ y_t = C x_t + D + v_t     │   M-step: closed-form update of noise Q
└───────────────────────────┘
          │  particle trajectories x_1:T
          ▼
┌───────────────────────────┐   multitaper spectrum of each trajectory,
│ Spectral features         │   restricted to the low-frequency band
└───────────────────────────┘   where task rhythms live
          │
          ▼
┌───────────────────────────┐   Flax 1D-CNN (2 conv layers + max-pooling),
│ 1D-CNN decoder            │   trained with Optax Adam; its class
└───────────────────────────┘   probabilities re-weight the particles
          │                     at every EM iteration
          ▼
   label prediction
```

**Implementation highlights**

- **Particle filter in pure JAX:** time recursion with `jax.lax.scan`, batching over particles and trials with `jax.vmap`, and data-dependent branches with `jax.lax.cond`.
- **Label-guided resampling:** the CNN's predicted class probabilities are used as particle weights, coupling the latent-state estimate to the decoding task.
- **Flax + Optax decoder:** a 1D-CNN with a `jax.jit`-compiled training step.
- **Evaluation:** k-fold cross-validation with per-fold and aggregated ROC/AUC, accuracy and confusion matrices.

---

## Application: Brief Death Implicit Association Test (BDIAT)

<img src="Images/AIT_Task.png" alt="BDIAT task" width="700">

Participants complete 360 trials (18 blocks of 20) of a Death Implicit Association Test while reaction times are recorded. Each participant's 360-trial reaction-time series is one sequence, labeled by group. The SSM acts as an adaptive smoother over the reaction times, and the CNN decodes the group label. See [Data_Description.md](Data_Description.md) for the full protocol.

> **Data availability:** the clinical data were collected under IRB approval and are not included in this repository.

| | |
|---|---|
| ![Particle filtering](Images/SSM.png) | ![Classification](Images/CNN.png) |
| Particle trajectories recover the underlying pattern in noisy reaction times. | Output of the CNN decoder trained on particle-filtered trajectories. |

---

## Repository structure

```
SSM-DNN-JAX/
├── Main.py                                    # Cross-validated training and evaluation
├── Latent_State_Inference.py                  # State dataclass, particle filter, EM (E- and M-steps)
├── Neural_Network_for_Classification_Task.py  # Flax 1D-CNN, jitted train step, CNN-guided resampling
├── Evaluate_Metric.py                         # Test-set evaluation (probabilities, confusion matrix)
├── Utiles.py                                  # Data splitting, class balancing, multitaper spectra, plotting
├── Data_Description.md                        # Task and dataset description
└── Images/                                    # Figures
```

## Code status

This is a research snapshot of the modeling pipeline, exported from the Colab notebook used in the study. The scripts show the full method, but the clinical data are not public, so they are not meant to run end-to-end out of the box. The [Colab notebook](https://colab.research.google.com/drive/10Y5nb0CLcVjm37g2VrAfuHTNNgykV53w?usp=sharing) is the best place to follow the workflow step by step.

Dependencies: `jax`, `flax`, `optax`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`.

```bash
git clone https://github.com/Pedram-Rajaei/SSM-DNN-JAX.git
pip install jax flax optax numpy scipy scikit-learn matplotlib
```

The multitaper spectrogram routine in `Utiles.py` is adapted from the [Prerau Lab multitaper toolbox](https://github.com/preraulab/multitaper_toolbox).

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{rajaei2025ssmdnn,
  title     = {Robust {DNN}-based Decoder Model with an Embedded State-Space Model Layer},
  author    = {Rajaei, Pedram and Kallam, Pavan and Garcia, Benito and Yadav, Shreya and McInnes, Aaron and Brown, Blair and Freedman, Miriam and Bronstein, Michael and Goodman Keiser, Melanie D. and Widge, Alik S. and Yousefi, Ali},
  booktitle = {2025 47th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)},
  pages     = {1--6},
  year      = {2025},
  doi       = {10.1109/EMBC58623.2025.11254746}
}
```

## Acknowledgments

This work was partially supported by the Defense Advanced Research Projects Agency (DARPA) under cooperative agreement #N660012324016. The content does not necessarily reflect the position or policy of the Government, and no official endorsement should be inferred. We thank our collaborators at the University of Minnesota and Intheon. This work was also supported by startup funds from the University of Houston.

## License

MIT. See [License](License).

## Contact

Pedram Rajaei, Department of Biomedical Engineering, University of Houston · prajaei@cougarnet.uh.edu
