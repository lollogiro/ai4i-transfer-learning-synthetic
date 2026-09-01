# 3W dataset: Synthetic Data Generation for Transfer Learning

## Dataset Description

3W (Vargas et al., 2019) is a public dataset of rare undesirable events in offshore oil and gas wells. It contains **1,984 instances** (whole events) recorded at **1 Hz over 8 process variables** and labeled into **9 classes** (1 normal + 8 failure modes). Instances come from three sources: **real wells (1,025)**, an OLGA **simulator (939)**, and expert **hand-drawn curves (20)**. Labels are two-level (instance class + per-observation period code), and the data carries quality issues, such as **31% missing variable readings** overall and **10% frozen sensors** (only in the real source), with severe class imbalance (the rarest real class has 3 instances).

## Stage 1: Exploratory Data Analysis

The EDA characterized composition, sensors, temporal structure, labels, and data quality. Only class 0 (normal) is real-only, while class 1 is the only event present in all three sources. Sensor prevalence varies by well/class (e.g. T-JUS-CKGL is absent from all class-0 real instances), event durations range from $\sim$ 0.5 h (DHSV closure) to 96 h (scaling), and the correlation structure of the 8 sensors differs by source, KS tests confirm real vs simulated distributions differ significantly.

## Domain Definition

In the context of 3W, a domain is defined along three axes: the **data source**, the producing **well**, and the **recording time period**. The quantitative analysis builds source-target pairs along these axes from the real instances of classes $\{0, 4\}$ (the only classes with enough occurence across wells to form many pairs): **12 single-well pairs**, **4 well-group pairs**, and **4 year pairs**. Only one **cross-source** pair (Simulated$\to$Real) is kept as a reference on the full target, outside the correlation computation, showing a very high penalty. Every pair trains on the full source and tests on the **eval** fold of a global 50/50 eval/reserve split (the reserve is never tested, it is used to train the CVAE in the stage 3). Instead of using raw time series, each event is summarized by 7 statistics per sensor (mean, std, min, p25, p50, p75, max) giving a 56-dim feature vector (8 sensors $\times$ 7 statistics). Raw recordings have variable durations (from under an hour to over a day), so a fixed-size representation is required for the Random Forest classifiers, the summary discards temporal structure, an accepted limitation for instance-level TL. The 56-dim feature space is further reduced to the **active subset 21/56** so constant features cannot pollute the distances through the scaler's variance floor.

## Stage 2: Domain Distance and Transfer Penalty

We computed the conditional **SWD / MMD / FD** distances (per class, active feature space, train-only imputation) and the **transfer penalty** (source CV macro-F1 - target macro-F1, across 5 seeds) over the 20 real pairs. The penalty correlates with the **mean-shift** distances but not with the shape metric: **SWD Spearman = +0.46**, **FD Spearman = +0.42** (block-bootstrap CIs), **MMD Spearman = -0.31** (robust to the RBF kernel's bandwidth). Penalties span 0.06-0.74. From these results we can observe that the mean-shift distances, specifically SWD and FD, predict the transfer penalty within real 3W pairs, while MMD is not a good predictor.

## Stage 3: Synthetic Data Generation

The experiment asks **how much labeled target data a CVAE generator needs** and whether synthetic data beats simply using the few real target examples. It runs on two pairs:
- W1 $\to$ W5 (baseline penalty 0.726) 
- {W4,W5} $\to$ {W1,W2} (baseline penalty 0.598)

Both on classes {0, 4}. Comparing per budget N: source-only, +N real, +N budget-only CVAE (trained only on N target instances), +N synthetic (source-pretrained + budget-finetuned CVAE), and +5N/+10N amplification control for the +N synthetic. Then, penalty is plotted against training set size, and then measuring the conditional SWD/FD/MMD to the target test set. The results show:

### Discussion

- **A small labeled target budget is very effective**: 4 real instances collapse W1 $\to$ W5's penalty from 0.726 to -0.031 (group pair: 0.598 $\to$ 0.082 at N = 4, down to 0.033 at N = 128).
- **Synthetic data is not always necessary**: forests trained with synthetic data never beat those trained on a few real target examples. The transferred CVAE reaches parity with 5N real from N = 32 on W1 $\to$ W5, but at the sometime does not give this promising results on the well-group pair, where the source-pretrained CVAE is not able to generate good synthetic data even if the conditional distance to the target test decrease with the increase of N.
- **Penalty tracks the distance**: as N grows, the conditional SWD/FD to the target test decreases and the penalty partially follows. On the other hand, MMD again fails to reflect transferability, confirming stage 2 results.

## Limitations

- The correlation is restricted to classes $\{0, 4\}$. with the rare classes are concentrated in some specific single wells and cannot form classification pairs.
- Only the mean-shift metrics correlate (SWD/FD), the effect is real but not strong, highlighted by a wide block-bootstrap confidence intervals.
- Only 21 of 56 features are active, so the distances describe the gap in the common-variance subspace only.
- Year-pair sources are small (the 2013 source has 31 instances), so the correlation is not robust to the source size.
- The 56-dim summary discards temporal structure, losing the temporal dynamics of the events.
- In stage 3, synthetic data never exceeds a few real target examples (128 for the well group and 64 for the single well), also the generator is limited by the small amount of available data.
