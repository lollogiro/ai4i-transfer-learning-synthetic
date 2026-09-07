# Synthetic Data Generation for Transfer Learning

## Datasets Description

### 3W

3W (Vargas et al., 2019) is a public dataset of rare undesirable events in offshore oil and gas wells. It contains **1,984 instances** (whole events) recorded at **1 Hz over 8 process variables** and labeled into **9 classes** (1 normal + 8 failure modes). Instances come from three sources: **real wells (1,025)**, an OLGA **simulator (939)**, and expert **hand-drawn curves (20)**. Labels are two-level (instance class + per-observation period code), and the data carries quality issues, such as **31% missing variable readings** overall and **10% frozen sensors** (only in the real source), with severe class imbalance (the rarest real class has 3 instances).

### MetroPT-3

The MetroPT-3 dataset provides real-world predictive maintenance data collected from an Air Production Unit (APU) operating on an urban metro train. It captures continuous multivariate sensor readings (including pressure, temperature, and motor current) during actual operation, offering a practical testbed for industrial challenges like failure prediction and anomaly detection.

### SCANIA

The SCANIA dataset consists of operational readouts and Time-To-Event (TTE) records for a fleet of heavy-duty commercial vehicles, targeting the predictive maintenance of a specific component (Component X). The dataset is highly heterogeneous, comprising both continuous sensor readings and complex multivariate histograms (binned operational conditions accumulated over time) alongside technical vehicle specifications (e.g., chassis or weight configurations categorized as Cat0, Cat1, Cat2). The primary challenge lies in predicting the `in_study_repair` target variable within an inherently noisy, heavily imbalanced industrial environment where normal operating conditions vastly outnumber actual failure events.

## Stage 1: Exploratory Data Analysis

### 3W

The EDA characterized composition, sensors, temporal structure, labels, and data quality. Only class 0 (normal) is real-only, while class 1 is the only event present in all three sources. Sensor prevalence varies by well/class (e.g. T-JUS-CKGL is absent from all class-0 real instances), event durations range from $\sim$ 0.5 h (DHSV closure) to 96 h (scaling), and the correlation structure of the 8 sensors differs by source, KS tests confirm real vs simulated distributions differ significantly.

### MetroPT-3

The raw MetroPT-3 dataset is entirely unlabeled. To perform supervised classification and transfer learning, we manually annotated the continuous sensor readings using the maintenance logs provided in the official data card. This process revealed four distinct "Air Leak" failure events spanning from April to July 2020. Our exploratory data analysis showed that normal operating data accounts for >98% of the dataset. Because this massive class imbalance overwhelms global distance metrics and masks the underlying physical concept drift, our methodology explicitly isolates and focuses the distance calculations exclusively on the anomalous datapoints.

### SCANIA

Our initial exploratory data analysis (EDA) characterized the severe class imbalance and complex temporal dynamics of the dataset. Using Mutual Information (MI), we identified the most informative sensors (e.g., features from families `158`, `167`, and `459`). However, correlation matrices revealed weak linear relationships with the target, indicating that physical degradation follows non-linear patterns.

By inverting the temporal axis to represent a "countdown to failure" (Time-To-Event = 0), we visualized a clear global degradation trajectory: aggregated histogram bins (e.g., `total_exposure_167`) systematically accumulate as the component approaches the end of its useful life. Crucially, Kernel Density Estimation (KDE) plots stratified by vehicle specifications provided the first visual proof of severe Covariate Shift. The underlying statistical distribution of identical sensors drastically mutated depending on the vehicle's physical configuration (`Cat0` vs. `Cat1` vs. `Cat2`), validating the hypothesis that a single, static global model would inevitably fail.

## Domains Definition

### 3W

In the context of 3W, a domain is defined along three axes: the **data source**, the producing **well**, and the **recording time period**. The quantitative analysis builds source-target pairs along these axes from the real instances of classes $\{0, 4\}$ (the only classes with enough occurrence across wells to form many pairs): **12 single-well pairs**, **4 well-group pairs**, and **4 year pairs**. Only one **cross-source** pair (Simulated$\to$Real) is kept as a reference on the full target, outside the correlation computation, showing a very high penalty. Every pair trains on the full source and tests on the **eval** fold of a global 50/50 eval/reserve split (the reserve is never tested; it is used to train the CVAE in stage 3). Instead of using raw time series, each event is summarized by 7 statistics per sensor (mean, std, min, p25, p50, p75, max) giving a 56-dim feature vector (8 sensors $\times$ 7 statistics). Raw recordings have variable durations (from under an hour to over a day), so a fixed-size representation is required for the Random Forest classifiers; the summary discards temporal structure, an accepted limitation for instance-level TL. The 56-dim feature space is further reduced to the **active subset 21/56** so constant features cannot pollute the distances through the scaler's variance floor.

### MetroPT-3

To evaluate transfer learning under different conditions, we initially defined three distinct domains: 
1. **Component Shift:** Partitioned using the `Towers` digital sensor to compare data generated while Tower 1 is active (Source) vs. Tower 2 (Target).
2. **Load Shift:** Partitioned using the `Motor_current` analog sensor to compare data generated while the compressor is operating 'Offloaded' (Source) vs. 'Under Load' (Target), based on the exact current thresholds provided in the data card.
3. **Seasonal Shift:** Partitioned using timestamps to compare anomalies occurring in the Spring (Source) vs. the Summer (Target).

### SCANIA

To move beyond a qualitative assessment and ensure a statistically robust evaluation of Transfer Learning, we programmatically generated **19 distinct Source-Target pairs** across three main axes:
1. **Technical Specifications (Cross-Specs):** Permutations among the vehicle hardware configurations (e.g., Cat0 $\rightarrow$ Cat1, Cat2 $\rightarrow$ Cat0) to quantify how physical chassis or load differences induce drift.
2. **Temporal Lifecycle Drift (Wear Level):** A quantile-based segmentation of the vehicles' maximum lifespan (Q1-Short, Q2-Medium Short, Q3-Medium, Q4-Long). Permuting these groups allowed us to simulate the physical drift between early (infant) failures and long-term wear-and-tear degradation.
3. **Sanity Check (Placebo):** A randomized 50/50 split within a single, homogenous category (Cat0) to act as a rigorous control baseline, ensuring our geometric metrics report near-zero distances in the absence of true physical domain shift.

## Methodology (common across datasets)

- **Base classifier**: Random Forest with trials over several seeds, reported as mean ± std.
- **Distances**: SWD / MMD / FD, computed on the **conditional shift**, per class, over the class(es) each dataset's task focuses on — never on the whole population, whose heavy imbalance would mask the drift.
- **Transfer penalty** = source CV macro-F1 - target macro-F1.
- **Leakage control**: test/target never shares the same entity (well / vehicle / window) with training.
- **Correlation distance $\times$ penalty** via Spearman/Kendall with 95% block-bootstrap CIs.

## Stage 2: Domain Distance and Transfer Penalty

### 3W

Across the previously defined 20 source-target pairs, we observed a strong correlation between the geometric distance and the transfer penalty. The Spearman correlation coefficients were as follows: **SWD = +0.46**, **FD = +0.42**, and **MMD = -0.31**. The transfer penalties ranged from 0.06 to 0.74, and pairing these results with corresponding domain distances, we found that the performance loss when transferring models, in most cases, also increases with the distance between source and target domains. The correlation was statistically significant for SWD and FD, while MMD failed to show a significant relationship, confirming that not all distance metrics are equally effective in predicting transferability.

### MetroPT-3

For each of the three initial domain configurations (Component, Load, Seasonal), distances were computed on the anomalies only, in the raw 7-D analog sensor space. To formally quantify the correlation between distance and penalty, we leveraged the 4-month seasonal data. Since the seasonal shift naturally produced the largest distance and penalty, it provided an ideal testbed. We split the 4 months into smaller temporal windows and generated a total of 27 unique Source-Target permutations. Across these 27 pairs, we rigorously evaluated the rank correlation using **Spearman's $\rho$** and **Kendall's $\tau$** (with 95% Block-Bootstrap CIs, grouped by undirected pair families, directly matching the 3W methodology). We found strong, statistically significant correlations for **SWD** ($\rho=+0.751$ [+0.478, +0.893]; $\tau=+0.572$ [+0.348, +0.746]) and **FD** ($\rho=+0.760$ [+0.502, +0.894]; $\tau=+0.582$ [+0.380, +0.742]). Conversely, MMD exhibited a weaker correlation ($\rho=+0.359$ [-0.132, +0.702]), confirming that SWD and FD are much more reliable predictors of transfer degradation.

Among the three primary, broad domain configurations, the **Spring $\rightarrow$ Summer** transition exhibited by far the most severe geometric distance and the highest transfer penalty (~54%), proving that environmental concept drift severely degrades predictive maintenance models. While some of the smaller, single-month permutations exhibited even more extreme distances and near-total transfer failure, we explicitly selected the broader Spring $\rightarrow$ Summer seasonal split as the primary focus for our generative synthetic experiments as the broader seasonal split guarantees a richer variance of normal states and provides a much larger dataset of anomalies to work with, establishing a rigorous and realistic industrial baseline.

### SCANIA

Distances were computed on class 1 (the anomalous readings) in the aggregated feature space. To evaluate the transfer penalty, we used a Random Forest with majority-class undersampling (3 seeds) under a strict `GroupShuffleSplit` by `vehicle_id`, so no vehicle ever appears in both training and test/target.

After computing the block-bootstrap CIs, we observed the following results:
- *FD demonstrated the strongest predictive power* with a statistically significant Spearman of **+0.607 [95% CI: 0.016, 0.935]** ($p = 0.016$). SWD also showed a solid positive correlation of **+0.532** ($p = 0.041$).
- *Transfer learning failed asymmetrically*, indeed transferring knowledge from early-failure vehicles (Q1) to long-life vehicles (Q4) triggered massive penalties. In contrast, the reverse path (Q4 $\rightarrow$ Q1) showed higher tolerance, suggesting long-term degradation signatures subsume early-failure patterns.

Unlike standard global metrics, we also integrated a 1D Wasserstein feature-sensitivity heatmap to provide explainability. When analyzing the severe drift between standard vehicles (`Cat0/1`) and `Cat2`, this technique mathematically isolated the aggregated histogram `total_exposure_291` (1D distance 7.24) and the continuous counters `370_0` and `100_0` as the absolute primary drivers of the concept drift, pinpointing exactly which sensors fail under configuration changes.

## Stage 3: Synthetic Data Generation

### 3W

For the 3W dataset, we check **how much labeled target data a CVAE generator needs** and whether **synthetic data beats simply using the few real target examples**. It runs on two pairs:
- W1 $\to$ W5 (baseline penalty 0.726), only 130 samples in the source and 59 in the target
- {W4,W5} $\to$ {W1,W2} (baseline penalty 0.598), larger source pool with 174 samples in the source and 226 in the target.

Both on classes {0, 4}. Comparing per budget N: source-only, +N real, +N budget-only CVAE (trained only on N target instances), +N synthetic (source-pretrained + budget-finetuned CVAE), and +5N/+10N amplification control for the +N synthetic. Then, penalty is plotted against training set size, and the conditional SWD/FD/MMD to the target test is measured.

A handful of real target examples might be already enough, synthetic data helps only where the source prior fits. A few real instances drive the penalty to almost 0 on both pairs, and the source-pretrained CVAE slightly outperforms results obtained with only the real budget on the homogeneous W1 $\to$ W5 pair, but underperforms a budget-only CVAE on the heterogeneous {W4,W5} $\to$ {W1,W2} pair.

### MetroPT-3

To bridge the Spring $\rightarrow$ Summer domain gap under realistic industrial constraints, we evaluated our generative models (CVAE and CGAN) in a strict "Few-Shot" Budget Sweep. We limited access to the target domain data to tiny budgets (1%, 5%, 10%, and 25%) to answer two core questions: can the models survive data starvation, and can they successfully amplify a tiny budget into a robust synthetic dataset?

We performed a rigorous Multi-Seed Evaluation (`N_RUNS=3`) for every budget to guarantee statistical significance, generating $1N$, $5N$, and $10N$ synthetic anomalies to evaluate amplification. 

The CVAE bridges the seasonal gap from a tiny budget, while the CGAN doesn't. Amplifying a 5% real-anomaly budget (35 samples) with the CVAE cuts the Spring $\to$ Summer penalty from 75% to 13.6%, close to the only-real-target ceiling, while the CGAN collapses under data starvation. Across budgets the penalty tracks the conditional SWD to the target, mirroring the stage-2 distance-penalty relationship.


### Discussion

#### 3W

- **A small labeled target budget can be effective**: 4 real instances collapse W1 $\to$ W5's penalty from 0.726 to -0.031 (group pair: 0.598 $\to$ 0.082 at N = 4, down to 0.033 at N = 128).
- **Synthetic data is not always necessary**: forests trained with synthetic data never beat those trained on a few real target examples. The transferred CVAE reaches parity with 5N real from N = 32 on W1 $\to$ W5, but at the same time it does not give these promising results on the well-group pair, where the source-pretrained CVAE cannot generate good synthetic data even though the conditional distance to the target test decreases as N grows.
- **Penalty tracks the distance**: as N grows, the conditional SWD/FD to the target test decreases and the penalty partially follows. On the other hand, MMD again fails to reflect transferability, confirming stage 2 results.

#### MetroPT-3

- **Data starvation vs. amplification:** The `CVAE (10x)` proved to be very robust, tracking closely behind the theoretical ceiling of using pure real target data. It successfully amplified a tiny 5% target budget (just 35 samples) to drop the Transfer Penalty from ~75% (averaged across our 3 random multi-seed trials) down to 13.6%. The CGAN, conversely, suffered from severe adversarial instability and mode collapse when starved of data.
- **Distance vs. penalty:** The multi-seed trajectory plot revealed a clear, overarching correlation, as the mathematical distance (SWD) increases, so does the transfer penalty. While the relationship is not perfectly monotonic (for example, even the Source + Real scenario exhibits a drop in penalty while its distance continues to increase, reflecting the natural statistical variance of drawing random samples) the macro trend across all scenarios strongly validates our stage 2 hypothesis, bridging the mathematical domain distance successfully closes the transfer learning gap.

#### SCANIA

- **Baseline weak results:** Due to the severe imbalance and extreme noise inherent to heavy-duty automotive telemetry, the absolute F1-Scores achieved on the source baselines remain modest (between 0.15 and 0.40). While this is sufficient to calculate a reliable transfer penalty, it highlights the limits of relying solely on standard Random Forests without advanced temporal modeling.
- **MMD instability:** MMD failed to exhibit a statistically significant correlation with the transfer penalty, confirming results obtained also in the 3W dataset. This suggests that MMD may not be a reliable metric for predicting transferability in complex, real-world industrial datasets.
