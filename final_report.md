# Synthetic Data Generation for Transfer Learning

## Datasets Description

### 3W

3W (Vargas et al., 2019) is a public dataset of rare undesirable events in offshore oil and gas wells. It contains **1,984 instances** (whole events) recorded at **1 Hz over 8 process variables** and labeled into **9 classes** (1 normal + 8 failure modes). Instances come from three sources: **real wells (1,025)**, an OLGA **simulator (939)**, and expert **hand-drawn curves (20)**. Labels are two-level (instance class + per-observation period code), and the data carries quality issues, such as **31% missing variable readings** overall and **10% frozen sensors** (only in the real source), with severe class imbalance (the rarest real class has 3 instances).

### MetroPT-3

The MetroPT-3 dataset provides real-world predictive maintenance data collected from an Air Production Unit (APU) operating on an urban metro train. It captures continuous multivariate sensor readings (including pressure, temperature, and motor current) during actual operation, offering a practical testbed for industrial challenges like failure prediction and anomaly detection.

### SCANIA

The SCANIA dataset consists of operational readouts and Time-To-Event (TTE) records for a fleet of heavy-duty commercial vehicles, targeting the predictive maintenance of a specific component (Component X). The dataset is highly heterogeneous, comprising both continuous sensor readings and complex multivariate histograms (binned operational conditions accumulated over time) alongside technical vehicle specifications (e.g., chassis or weight configurations categorized as Cat0, Cat1, Cat2). The primary challenge lies in predicting the `in_study_repair` target variable within an inherently noisy, heavily imbalanced industrial environment where normal operating conditions vastly outnumber actual failure events.

### Tennessee Eastman Process (TEP)

The Tennessee Eastman Process (TEP) dataset (Downs & Vogel, 1993; Rieth et al., 2017) is a widely established simulation benchmark of a complex chemical production plant. It tracks **52 continuous process variables** (pressures, temperatures, flow rates, and valve positions) sampled across **21 distinct operational classes** (1 normal state + 20 fault mechanisms). The dataset comprises 500 independent simulation runs per fault, featuring complete temporal sequences without missing values or frozen sensors. It is originally distributed across four RData files (fault_free_train/test and faulty_train/test), which differ in run length and fault injection timing: 500 samples (25h) with injection at sample 20 in training, 960 samples (48h) with injection at sample 160 in testing.

## Stage 1: Exploratory Data Analysis

### 3W

The EDA characterized composition, sensors, temporal structure, labels, and data quality. Only class 0 (normal) is real-only, while class 1 is the only event present in all three sources. Sensor prevalence varies by well/class (e.g. T-JUS-CKGL is absent from all class-0 real instances), event durations range from $\sim$ 0.5 h (DHSV closure) to 96 h (scaling), and the correlation structure of the 8 sensors differs by source, KS tests confirm real vs simulated distributions differ significantly.

### MetroPT-3

The raw MetroPT-3 dataset is entirely unlabeled. To perform supervised classification and transfer learning, we manually annotated the continuous sensor readings using the maintenance logs provided in the official data card. This process revealed four distinct "Air Leak" failure events spanning from April to July 2020. Our exploratory data analysis showed that normal operating data accounts for >98% of the dataset. Because this massive class imbalance overwhelms global distance metrics and masks the underlying physical concept drift, our methodology explicitly isolates and focuses the distance calculations exclusively on the anomalous datapoints.

### SCANIA

Our initial exploratory data analysis (EDA) characterized the severe class imbalance and complex temporal dynamics of the dataset. Using Mutual Information (MI), we identified the most informative sensors (e.g., features from families `158`, `167`, and `459`). However, correlation matrices revealed weak linear relationships with the target, indicating that physical degradation follows non-linear patterns.

By inverting the temporal axis to represent a "countdown to failure" (Time-To-Event = 0), we visualized a clear global degradation trajectory: aggregated histogram bins (e.g., `total_exposure_167`) systematically accumulate as the component approaches the end of its useful life. Crucially, Kernel Density Estimation (KDE) plots stratified by vehicle specifications provided the first visual proof of severe Covariate Shift. The underlying statistical distribution of identical sensors drastically mutated depending on the vehicle's physical configuration (`Cat0` vs. `Cat1` vs. `Cat2`), validating the hypothesis that a single, static global model would inevitably fail.

### Tennessee Eastman Process (TEP)

Our exploratory analysis evaluated feature distributions, variable sensitivities, and temporal fault progression across all 52 process variables. By computing Z-score deviations against the normal baseline ($Z_{f,v} = (\mu_{f,v} - \mu_{\text{normal},v})/\sigma_{\text{normal},v}$), we identified key discriminative features characterizing the 20 failure modes across their physical mechanisms (step changes, random variations, kinetic drift, valve stiction, and unmodeled perturbations).
Inspecting temporal trajectories of the training dataset revealed distinct onset behaviors across fault classes. Across 500 simulation runs, post-injection behavior varies significantly by fault mechanism: some variables settle into new steady states, others exhibit persistent oscillations, and key control parameters show sharp cross-run variance surges during plant stabilization. Cross-run consistency heatmaps revealed that nearly all tested faults show a sharp shift in process variability at the fault injection and remain highly reproducible across all 500 simulation runs. Finally, we revealed distinct static signatures across fault types, ranging from narrow interquartile ranges with localized outliers to heavily skewed distributions. To capture these distribution profiles, each time series was summarized using 4 sliding-window statistics (mean, std, min, max), yielding a 208-dimensional feature representation.

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

### Tennessee Eastman Process (TEP)

Domains in TEP are defined along physical disturbance mechanisms, cross-campaign splits, and diagnostic task regimes:

1. **Binary Fault Detection Pairs (20 Domain Pairs):** Partitioned into 5 categories based on physical alignment: _Similar mechanisms_ (e.g., feed temperature shifts $F3 \to F9$), _Same subsystem_ (e.g., condenser thermal/valve $F5 \to F15$), _Unrelated mechanisms_ (e.g., chemical kinetics vs. actuator stiction $F4 \to F13$, $F6 \to F15$), _Coherent groups_ (pooled, physically aligned sources sharing target dynamics, such as $\{F3, F4\} \to F9$), and _Control group_ (orthogonal multi-source inputs transferred to an unrelated target, specifically $\{F1, F6\} \to F15$).
2. **Multiclass Diagnostic Taxonomy (6 Physical Classes):** Grouped into 6 operational mechanism categories: Normal ($F0$), Step Disturbances ($F1 \dots F7$), Random Variations ($F8 \dots F12$), Slow Dynamic Drift ($F13$), Actuator Stiction ($F14 \dots F15$), and Unknown Perturbations ($F16 \dots F20$). Structured cross-mechanism pairs (8 pairs) simulate diagnostic transfer across distinct physical failure modes.

## Methodology (common across datasets)

- **Base classifier**: Random Forest with trials over several seeds, reported as mean ± std.
- **Distances**: SWD / MMD / FD, computed on the **conditional shift**, per class, over the classes each dataset's task focuses on.
- **Transfer penalty** = source CV macro-F1 - target macro-F1.
- **Leakage control**: test/target never shares the same entity (well / vehicle / window) with training.
- **Correlation distance $\times$ penalty** via Spearman $\tau$/Kendall $\rho$ with 95% block-bootstrap CIs.

## Stage 2: Domain Distance and Transfer Penalty

### 3W

Across the previously defined 20 source-target pairs, we observed a strong correlation between the geometric distance and the transfer penalty. The Spearman correlation coefficients were as follows: **SWD = +0.46**, **FD = +0.42**, and **MMD = -0.31**. The transfer penalties ranged from 0.06 to 0.74, and pairing these results with corresponding domain distances, we found that the performance loss when transferring models, in most cases, also increases with the distance between source and target domains. The correlation was statistically significant for SWD and FD, while MMD failed to show a significant relationship, confirming that not all distance metrics are equally effective in predicting transferability.

### MetroPT-3

For each of the three initial domain configurations (Component, Load, Seasonal), geometric distances were computed on the anomalies in the 7-D continuous space. This preliminary analysis strongly suggested that Transfer Learning effectiveness is heavily dependent on the mathematical distance between the Source and Target domains. The results indicated that the Transfer Learning penalty scales exponentially with geometric domain distance, with macroscopic temporal shifts (changing seasons) inducing a significantly larger spatial displacement and transfer penalty than varying the physical load conditions or components on the train.

To rigorously confirm this relationship, we expanded our sample size to 24 strictly disjoint temporal permutations (1-Month and 1-to-2-Month shifts) from the seasonal data, completely avoiding data leakage. We performed a dual-normalization analysis to evaluate the distance metrics under different operational assumptions:
- **Robust Prediction under Real-World Conditions**: Under strict Asymmetric Normalization (which simulates evaluating a frozen model on shifted data by scaling the target with the source's parameters), both **SWD** ($\rho \approx +0.78$ [+0.59, +0.88]) and **FD** ($\rho \approx +0.76$ [+0.57, +0.87]) proved to be highly robust predictors. They successfully modeled the exponential degradation of the classifier as the geometric distance expanded. MMD, however, completely failed to capture this shift ($\rho \approx -0.07$) due to the loss of localized kernel overlap in the displaced feature space.
- **Symmetric Normalization**: Aligning the domains globally via a common Symmetric Normalization restored the strict mathematical symmetry of the distances and successfully rescued MMD. In this shared feature space, all three metrics showed a strong, observable linear trend with the Transfer Penalty: **SWD** ($\rho = +0.68$, $\tau = +0.46$), **MMD** ($\rho = +0.62$, $\tau = +0.44$), and **FD** ($\rho = +0.65$, $\tau = +0.47$).

Ultimately, isolating the geometric distance of the **Conditional Shift** provides a highly accurate predictor of Transfer Learning success. While SWD and FD are highly resilient regardless of the normalization strategy, MMD requires symmetric feature alignment to function correctly.

**Note on Split Methodology:** We rigorously investigated avoiding the random train/test split in favor of separating at the temporal event block level to avoid adjacent row leakage. However, because MetroPT-3 contains only a single non-stationary anomaly event per month, a strict chronological block split forces the model to extrapolate across changing physical signatures (e.g., training on the onset of a leak, and testing on the stabilization phase). Because Random Forests are row-independent classifiers without temporal memory, this caused catastrophic extrapolation failure on the Source test set, collapsing the baseline F1-Score and rendering the Transfer Penalty undefined. We empirically concluded that for this specific dataset and model architecture, uniform random row splitting is mathematically required to construct a valid and stable source baseline against which transfer degradation can be measured.

As the Spring $\rightarrow$ Summer transition exhibited the most severe transfer penalty, we explicitly selected this broader seasonal split as the primary focus for our generative synthetic experiments in Stage 3, establishing a rigorous industrial baseline.

### SCANIA

Distances were computed on class 1 (the anomalous readings) in the aggregated feature space. To evaluate the transfer penalty, we used a Random Forest with majority-class undersampling (3 seeds) under a strict `GroupShuffleSplit` by `vehicle_id`, so no vehicle ever appears in both training and test/target.

After computing the block-bootstrap CIs, we observed the following results:
- *FD demonstrated the strongest predictive power* with a statistically significant Spearman of **+0.607 [95% CI: 0.016, 0.935]** ($p = 0.016$). SWD also showed a solid positive correlation of **+0.532** ($p = 0.041$).
- *Transfer learning failed asymmetrically*, indeed transferring knowledge from early-failure vehicles (Q1) to long-life vehicles (Q4) triggered massive penalties. In contrast, the reverse path (Q4 $\rightarrow$ Q1) showed higher tolerance, suggesting long-term degradation signatures subsume early-failure patterns.

Unlike standard global metrics, we also integrated a 1D Wasserstein feature-sensitivity heatmap to provide explainability. When analyzing the severe drift between standard vehicles (`Cat0/1`) and `Cat2`, this technique mathematically isolated the aggregated histogram `total_exposure_291` (1D distance 7.24) and the continuous counters `370_0` and `100_0` as the absolute primary drivers of the concept drift, pinpointing exactly which sensors fail under configuration changes.

### Tennessee Eastman Process (TEP)

Geometric distances were computed on anomalous states in the 208-dimensional sliding-window feature space. Across the 20 binary source-target pairs, transfer evaluation using the primary Random Forest classifier confirmed a strong positive relationship between domain distance and transfer penalty. Correlation analysis with block-bootstrap 95% CIs yielded statistically significant positive predictive performance across all distance metrics: **SWD = +0.549 [95% CI: 0.141, 0.703]**, **MMD = +0.517 [95% CI: 0.143, 0.703]**, and **FD = +0.533 [95% CI: 0.094, 0.680]**.

Transfer performance varied significantly by physical domain alignment: _Similar mechanism_ pairs (e.g., $F3 \to F9$) exhibited minimal distance ($\text{SWD} = 0.0668$, $\text{MMD} = 0.0019$) and low transfer penalty ($\text{penalty} = 0.2941$), with some pairs like $F9 \to F15$ and $F11 \to F14$ achieving near-zero transfer penalties ($\text{penalty} \approx 0.00$). In contrast, transfer across _Unrelated mechanisms_ (e.g., $F6 \to F15$, $\text{SWD} = 1.1761$, $\text{FD} = 518.48$) resulted in total classifier breakdown ($\text{penalty} = 1.0000$).

Evaluating multiclass transfer across structured physical perturbation groups reveals a severe breakdown in cross-domain generalization. Across all evaluated domain pairs, despite strong in-domain Baseline Macro-F1, cross-domain transfer triggers massive performance degradation, with transfer penalties frequently exceeding $0.75$ and target Macro-F1 consistently dropping below $0.25$. This confirms that direct cross-domain diagnosis fails across physical mechanism boundaries without conditional alignment.

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

### Tennessee Eastman Process (TEP)

To evaluate domain bridging across complex chemical plant perturbations, we evaluated CVAE and CGAN models in a budget-controlled sweep ($2\%$, $6\%$, $12\%$, and $20\%$ target data with $1\times$, $5\times$, and $10\times$ amplification) across both binary detection and multiclass diagnosis tracks. The generative results directly confirmed the Stage 2 distance-penalty relationship, though performance across domain categories frequently exhibited a performance plateau.

For physically aligned pairs (e.g., binary $F3 \to F9$), adding CVAE synthetic data with just a 2% target budget boosted performance from $\text{F1} = 0.6171$ to $\text{F1} = 0.6667$. The generated samples brought the source and target data closer together, cutting the transfer penalty and nearing the score of real target data. However, across many domain categories, the addition of synthetic data reached a stalemate, keeping the gap wide, showing no real improvement over the baseline, and failing to capture the target fault patterns.

In the multiclass diagnostic track, synthetic generation similarly faced steep transfer limits when bridging non-aligned physical perturbation groups: transferring synthetic samples across disjoint operational categories (such as _Step Disturbances vs. Actuator Stiction_) resulted in minimal performance recovery, with target Macro-F1 remaining constrained near baseline levels ($\text{Target F1} \approx 0.2009$) due to non-overlapping feature manifolds.

### Discussion

#### 3W

- **A small labeled target budget can be effective**: 4 real instances collapse W1 $\to$ W5's penalty from 0.726 to -0.031 (group pair: 0.598 $\to$ 0.082 at N = 4, down to 0.033 at N = 128).
- **Synthetic data is not always necessary**: forests trained with synthetic data never beat those trained on a few real target examples. The transferred CVAE reaches parity with 5N real from N = 32 on W1 $\to$ W5, but at the same time it does not give these promising results on the well-group pair, where the source-pretrained CVAE cannot generate good synthetic data even though the conditional distance to the target test decreases as N grows.
- **Penalty tracks the distance**: as N grows, the conditional SWD/FD to the target test decreases and the penalty partially follows. On the other hand, MMD again fails to reflect transferability, confirming stage 2 results.

#### MetroPT-3

- **Data starvation vs. amplification:** The `CVAE (10x)` proved highly effective in extreme data starvation scenarios. In highly restricted few-shot settings (e.g., amplifying a tiny 5% target budget of just 35 samples), the CVAE effectively bridged the gap, dropping the Transfer Penalty from ~75% down to 13.6%. However, as the available target budget grows, the direct use of real target data predictably becomes preferable and overtakes the synthetic augmentation. The CGAN, conversely, suffered from severe adversarial instability and mode collapse across all starvation budgets.
- **Distance vs. penalty:** The multi-seed trajectory plot revealed a clear, overarching correlation: as the mathematical distance decreases, so does the transfer penalty. While the relationship is not perfectly monotonic (reflecting the inherent statistical variance of drawing random samples), the aggregate macro trend across all augmentation strategies strongly validates our Stage 2 hypothesis: bridging the mathematical domain distance successfully closes the transfer learning gap.

#### SCANIA

- **Baseline weak results:** Due to the severe imbalance and extreme noise inherent to heavy-duty automotive telemetry, the absolute F1-Scores achieved on the source baselines remain modest (between 0.15 and 0.40). While this is sufficient to calculate a reliable transfer penalty, it highlights the limits of relying solely on standard Random Forests without advanced temporal modeling.
- **MMD instability:** MMD failed to exhibit a statistically significant correlation with the transfer penalty, confirming results obtained also in the 3W dataset. This suggests that MMD may not be a reliable metric for predicting transferability in complex, real-world industrial datasets.

#### Tennessee Eastman Process (TEP)

- **Mechanism alignment vs. transfer limits:** For physically aligned failure modes (e.g., $F3 \to F9$), CVAE synthetic augmentation helped cut transfer penalties toward the real-target ceiling. Conversely, for completely disjoint mechanisms (e.g., $F6 \to F15$ with $\text{SWD} = 1.1761$), source-only transfer breaks down completely ($\text{penalty} = 1.0000$), and synthetic generation alone fails to bridge the gap, resulting in a performance stalemate near baseline levels.
- **Multiclass diagnostic transfer:** Across TEP's physical perturbation groups, supervised in-domain models achieved strong baseline performance. However, cross-mechanism transfer triggered severe performance drops, proving that distinct physical perturbations occupy disjoint feature manifolds where synthetic data provides limited cross-domain recovery.
- **Generative stability & distance metrics:** CGAN models suffered from adversarial mode collapse on high-dimensional 208-D continuous features, whereas CVAE benefited from latent sampling temperature scaling ($T = 2.0$). Furthermore, Sliced Wasserstein Distance ($\rho = +0.549$), Fréchet Distance ($\rho = +0.533$), and MMD ($\rho = +0.517$) reliably predicted transfer penalties across TEP domain pairs, with all 95% bootstrap confidence intervals strictly excluding zero.