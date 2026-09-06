# SCANIA Dataset: Domain Shift Quantification and Transfer Learning

## Dataset Description
The SCANIA dataset consists of operational readouts and Time-To-Event (TTE) records for a fleet of heavy-duty commercial vehicles, targeting the predictive maintenance of a specific component (Component X). The dataset is highly heterogeneous, comprising both continuous sensor readings and complex multivariate histograms (binned operational conditions accumulated over time) alongside technical vehicle specifications (e.g., chassis or weight configurations categorized as Cat0, Cat1, Cat2). The primary challenge lies in predicting the `in_study_repair` target variable within an inherently noisy, heavily imbalanced industrial environment where normal operating conditions vastly outnumber actual failure events.

## Stage 1: Exploratory Data Analysis
Our initial exploratory data analysis (EDA) characterized the severe class imbalance and complex temporal dynamics of the dataset. Using Mutual Information (MI), we identified the most informative sensors (e.g., features from families `158`, `167`, and `459`). However, correlation matrices revealed weak linear relationships with the target, indicating that physical degradation follows non-linear patterns.

By inverting the temporal axis to represent a "countdown to failure" (Time-To-Event = 0), we visualized a clear global degradation trajectory: aggregated histogram bins (e.g., `total_exposure_167`) systematically accumulate as the component approaches the end of its useful life. Crucially, Kernel Density Estimation (KDE) plots stratified by vehicle specifications provided the first visual proof of severe Covariate Shift. The underlying statistical distribution of identical sensors drastically mutated depending on the vehicle's physical configuration (Cat0 vs. Cat1 vs. Cat2), validating the hypothesis that a single, static global model would inevitably fail.

## Domain Definition
To move beyond a qualitative assessment and ensure a statistically robust evaluation of Transfer Learning, we programmatically generated **19 distinct Source-Target pairs** across three main axes:
1. **Technical Specifications (Cross-Specs):** Permutations among the vehicle hardware configurations (e.g., Cat0 $\rightarrow$ Cat1, Cat2 $\rightarrow$ Cat0) to quantify how physical chassis or load differences induce drift.
2. **Temporal Lifecycle Drift (Wear Level):** A quantile-based segmentation of the vehicles' maximum lifespan (Q1-Short, Q2-Medium Short, Q3-Medium, Q4-Long). Permuting these groups allowed us to simulate the physical drift between early (infant) failures and long-term wear-and-tear degradation.
3. **Sanity Check (Placebo):** A randomized 50/50 split within a single, homogenous category (Cat0) to act as a rigorous control baseline, ensuring our geometric metrics report near-zero distances in the absence of true physical domain shift.

## Stage 2: Domain Distance & Transfer Penalty

### Methodology
To prevent the overwhelming volume of nominal data from masking the actual degradation shift, spatial distances were calculated **exclusively on the anomalous readings (class 1)**. We computed the conditional **Sliced Wasserstein Distance (SWD)**, **Maximum Mean Discrepancy (MMD)**, and **Fréchet Distance (FD)** in the active feature space. 

To evaluate the Transfer Penalty (the drop in macro-F1 score), we implemented a strict `GroupShuffleSplit` based on the `vehicle_id`. This architectural constraint guaranteed zero data leakage, ensuring no readings from a training vehicle ever leaked into the test or target evaluation sets. A Random Forest classifier (with undersampling of the majority class) was trained across 3 independent random seeds, extracting the mean and standard deviation of the penalty to isolate true geometric impact from stochastic algorithmic variance.

### Quantitative Results & Correlation
A Global PCA projection confirmed heavy Covariate Shift across the dataset, while a Conditional PCA (anomalies only) visually proved Concept Drift, showing the "failure signature" migrating across the vector space depending on the domain.

To answer precisely how well domain distance predicts the loss of transfer performance, we calculated correlations across the 19 pairs using 2000-iteration block-bootstrapping to extract 95% Confidence Intervals (CI). 
* **Strong Predictors:** The Fréchet Distance (FD) demonstrated the strongest predictive power with a statistically significant Spearman Rho of **+0.607 [95% CI: 0.016, 0.935]** ($p = 0.016$). SWD also showed a solid positive correlation of **+0.532** ($p = 0.041$).
* **Asymmetric Degradation:** Transfer learning failed asymmetrically. Transferring knowledge from early-failure vehicles (Q1) to long-life vehicles (Q4) triggered massive penalties. Conversely, the reverse path (Q4 $\rightarrow$ Q1) showed higher tolerance, suggesting long-term degradation signatures subsume early-failure patterns.
* **Control Validation:** The Sanity Check pair correctly returned near-zero spatial distances and a null transfer penalty, certifying the pipeline's immunity to random statistical noise.

### Explainability and Concept Drift Isolation
Unlike standard global metrics, we integrated a 1D Wasserstein Feature Sensitivity Heatmap to provide actionable engineering explainability. When analyzing the severe drift between standard vehicles (Cat0/1) and Cat2, this technique mathematically isolated the aggregated histogram `total_exposure_291` (1D distance 7.24) and the continuous counters `370_0` and `100_0` as the absolute primary drivers of the concept drift, pinpointing exactly which sensors buckle under configuration changes.

## Limitations
* **Absence of Synthetic Mitigation:** Unlike other evaluated datasets, Stage 3 (Synthetic Data Generation) was explicitly omitted for the SCANIA dataset. Our pipeline robustly *diagnoses* and quantifies the Transfer Learning failure but does not attempt to bridge the gap via generative modeling (e.g., CVAE/CGAN).
* **Baseline Modesty:** Due to the severe imbalance and extreme noise inherent to heavy-duty automotive telemetry, the absolute F1-Scores achieved on the source baselines remain modest (between 0.15 and 0.40). While this is sufficient to calculate a reliable delta (Transfer Penalty), it highlights the limits of relying solely on standard Random Forests without advanced temporal modeling.
* **MMD Instability:** MMD failed to exhibit a statistically significant correlation with the Transfer Penalty, confirming theoretical expectations that uncalibrated RBF kernels struggle to capture drift accurately in high-dimensional, aggregated sensor spaces.