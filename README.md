# JEPA Model Views Experimentation

This repository is dedicated to experimenting with **Joint-Embedding Predictive Architecture (JEPA)** model views in Image Prediction tasks. It leverages advancements from **VICRegularization** (Variance-Invariance-Covariance Regularization) and the [**LeJEPA**](https://github.com/galilai-group/lejepa) framework to explore how different view selection strategies influence representation learning efficiency and quality. <br>

#### [Weights and Biases Link](https://wandb.ai/aho13-duke-university/VIT_JEPA_Views?nw=nwuseraho13)

## Goals

The primary goals of this codebase are:
1.  **JEPA & View Selection**: To investigate the impact of various view selection strategies (e.g., Random, Laplacian, LID-based) on the performance of JEPA models.
2.  **Regularization**: To apply and test **SIGReg** (part of the VICReg family) for regularizing the embedding space, ensuring variance and decorrelation of features.
3.  **Efficiency**: To benchmark and optimize the training pipeline for Vision Transformers (ViT) in a self-supervised learning setting.

## Key Frameworks

### LeJEPA & VICReg
The core training loop utilizes the **LeJEPA** loss function, which combines:
-   **Prediction Loss**: Minimizing the distance between the predicted representations of global context views and other views (local or global).
-   **SIGReg Loss**: A regularization term inspired by VICReg that encourages:
    -   **Variance**: Preventing collapse by ensuring embedding dimensions have variance.
    -   **Invariance**: Ensuring similar views map to similar embeddings.
    -   **Covariance**: Decorrelating feature dimensions to maximize information content.

### View Selection Strategies
The repository implements novel ways to select "views" (crops/transformations) of an image to maximize learning signal:
-   **Random**: Standard random cropping and augmentation.
-   **Mixed / Cross-Instance**: Mixing views from different instances or using specific "mixed" data augmentations.
-   **Laplacian**: Selecting views based on Laplacian gradients to focus on high-frequency/informative regions.
-   **LID (Local Intrinsic Dimensionality)**: Using LID estimation to select views that are "hard" or diverse in the embedding space.

## Main Experiment: `scripts/vit_views.sh`

The primary entry point for running experiments is `scripts/vit_views.sh`. This script configures and launches the `eval/views_vit.py` training loop using Slurm (or can be run locally).

### Key Parameters
The experiment is highly configurable via Hydra. Important parameters in `scripts/vit_views.sh` include:

*   `+view_selection`: The strategy to use (`mixed`, `Laplacian`, `random`, `lid`).
*   `+model_name`: The backbone architecture (default: `vit_base_patch16_224.dino`).
*   `+V_global` / `+V_local`: Number of global (large) and local (small) views per image.
*   `+lamb`: The weight of the regularization loss.
*   `+proj_dim`: Dimension of the projection head.

### Training Flow
1.  **Data Loading**: Images are loaded and processed into multiple views (global and local) based on the `view_selection` strategy.
2.  **Encoder**: A Vision Transformer (ViT) processes these views to generate embeddings.
3.  **Projection**: Embeddings are projected into a higher-dimensional space for loss calculation.
4.  **Loss Calculation**: The LeJEPA loss (Prediction + SIGReg) is computed.
5.  **Evaluation**: A linear probe is trained online to monitor classification accuracy. Additional representation metrics (Effective Rank, LID, Cluster Quality) are logged to WandB.

## Usage

To run the main experiment on a Slurm cluster:

```bash
sbatch scripts/vit_views.sh
```

To run locally (ensure you have the environment set up):

```bash
./run_inf.sh python eval/views_vit.py \
  +lamb=0.05 \
  +V_global=2 \
  +V_local=5 \
  +view_selection=mixed \
  +model_name=vit_base_patch16_224.dino
```

## Directory Structure
-   `eval/`: Main training and evaluation scripts (`views_vit.py`, `selection.py`).
-   `scripts/`: Shell scripts for job submission and experimental configurations.
-   `jepa/`: Core JEPA model definitions (loss functions, datasets).
-   `stats/`: utilities for computing representation metrics.
