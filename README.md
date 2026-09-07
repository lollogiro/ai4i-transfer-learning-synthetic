# Synthetic Data Generation for Transfer Learning

AI for Industry group project: **transfer learning (TL) with synthetic data for detecting rare, undesirable events in industrial systems**. The working hypothesis is that TL efficacy depends on the source-target domain distance, the larger the distance, so the larger the performance loss when a model trained on one domain is applied to another, and that synthetic data can bridge the gap where transfer fails.

## Overview

The project follows the same three-stage architecture for every dataset:

1. **Stage 1 - Characterization (EDA):** load and understand the data, its sensors, temporal structure, labels, quality issues, and the domains it contains.
2. **Stage 2 - Domain distance + transfer penalty:** define quantitative source-target domain pairs, measure the domain distance with **SWD / MMD / Fréchet Distance**, and evaluate whether that distance predicts the **transfer penalty** (Q1).
3. **Stage 3 - Synthetic generation:** generate synthetic data with conditional generative models (CVAE / CGAN) and check whether the resulting distance reduction translates into a transfer-penalty reduction (Q2).

Four industrial datasets were selected. **3W**, **MetroPT-3**, **SCANIA Component X**, **Tennessee Eastman Process**.

Two questions frame every experiment:

- **Q1**: How well does source-target domain distance predict the transfer penalty?
- **Q2**: When synthetic data is introduced, does the resulting distance reduction translate into a reduced transfer penalty?

## Report

**`final_report.md`** is the single project report. It contains the datasets' description, the shared methodology, the domain definitions and source-target pairs, the
main results and the limitations of the study.

## Datasets

| Dataset | System / task | Signals | Classes | Units |
|---|---|---|---|---|
| **3W** (Vargas et al., 2019) | Offshore oil & gas wells - rare-event classification | 8 process variables | 9 (normal + 8 failure modes) | 1,984 instances (whole events), 1 Hz |
| **MetroPT-3** | Urban metro Air Production Unit - predictive maintenance | 15 (7 analog + 8 digital) | raw unlabeled, annotated *Air Leak* faults | continuous time series (1 row/s) |
| **SCANIA Component X** | TODO | TODO | TODO | TODO |
| **Tennessee Eastman Process** | TODO | TODO | TODO | TODO |

### 3W

Public dataset of rare undesirable events in oil and gas wells (Vargas et al., 2019). It contains **1,984 instances** (whole events) recorded at **1 Hz over 8 process variables** and labeled into **9 classes** (1 normal + 8 failure modes). Instances come from three sources: **real wells (1,025)**, an OLGA **simulator (939)**, and expert **hand-drawn curves (20)**. Labels are two-level (instance class + per-observation period code), and the data carries quality issues: **31 % missing variable readings** overall and **10 % frozen sensors** (only in the real source), with severe class imbalance (the rarest real class holds 3 instances).

### MetroPT-3

Real-world predictive-maintenance data collected from an Air Production Unit (APU) operating on an urban metro train. It captures continuous multivariate sensor readings (pressure, temperature, motor current, etc.) during actual operation. The raw data is **entirely unlabeled**. We manually annotated it with the maintenance logs in the
official data card, revealing four distinct **Air Leak** failure events (April-July 2020). Normal operating data accounts for **> 98 %** of the dataset, so the methodology isolates the **anomalous datapoints** for the distance calculations (global metrics are overwhelmed by the massive class imbalance).

### SCANIA Component X

TODO

### Tennessee Eastman Process

TODO

## Repository structure

The project is organized by **stage folders** (one folder per pipeline stage). Each developed dataset contributes **one notebook per stage folder**.

```
README.md                          # This file
final_report.md                    # Project report (methodology, results, limitations)

data/                              # Raw data, git-ignored
  ...

stage1-data_exploration/           # Stage 1 - EDA / characterization (one notebook per dataset)
  3w_eda.ipynb                     #   3W EDA
  metropt3.ipynb                   #   MetroPT-3 EDA
  TODO.ipynb                     #   SCANIA Component X EDA
  TODO.ipynb                  #   Tennessee Eastman EDA

stage2-transfer_learning/          # Stage 2 - domain distance + transfer penalty
  3w_dom_dist_tl.ipynb             #   3W: 3-pair qualitative picture + 20-pair distance $\longleftrightarrow$ penalty correlation
  metropt3.ipynb                   #   MetroPT-3: domain-distance analysis over 3 domain pairs + 27-pair seasonal correlation
  TODO.ipynb                     #   SCANIA Component X distance + TL
  TODO.ipynb                  #   Tennessee Eastman distance + TL

stage3-synthetic_generation/       # Stage 3 - synthetic data generation (CVAE and/or CGAN)
  3w_generation.ipynb              #   3W: CVAE generation, from scratch/fine-tuning, and transfer learning tests with synthetic data over 2 source-target pairs
  metropt3_budget_sweep.ipynb      #   MetroPT-3: few-shot budget sweep over the seasonal (Spring vs Summer) gap
  TODO.ipynb                     #   SCANIA Component X generation
  TODO.ipynb                  #   Tennessee Eastman generation
```

## Data

Raw data is **git-ignored** and lives in the `data/` folder at the repository root.

- **3W**: `data/Data for A Realistic and Public Dataset with Rare Undesirable Real Events in Oil Wells/` (original 300 MB zip + extracted multi-part `data.7z.001-004`). The stage notebooks unpack the multi-part 7z archives automatically via `py7zr`. Download from Zenodo.
- **MetroPT-3**: expected at `data/metropt3/MetroPT3(AirCompressor).csv`. Donwload from Kaggle.
- **SCANIA Component X**: TODO
- **Tennessee Eastman**: TODO

## Environment and notebooks

- **Python 3.12** with the project virtual environment `.venv/` at the repository root (git-ignored). All packages are pre-installed, see `requirements.txt`.
- The notebooks are self-contained.

## Methodology (common to all datasets)

- **Distances.** The **conditional shift** (per-class distance, averaged over the pair's classes) in the classifier's feature space:
  - **SWD** (Sliced Wasserstein): mean of the 1-D Wasserstein distance over 50 random projections.
  - **MMD** (Maximum Mean Discrepancy): RBF kernel, gamma = 0.1.
  - **FD** (Fréchet Distance): distance between the multivariate Gaussians fitted to the two sets (covariance-regularized).
- **Transfer penalty** = source CV macro-F1 - target macro-F1 (train on the source, test on the target), positive means loss.
- **Correlation** of distance $\times$ penalty is quantified with Spearman / Kendall plus block-bootstrap confidence intervals.

See `final_report.md` for the domain definitions, the source-target pairs, and the main results of each dataset.
