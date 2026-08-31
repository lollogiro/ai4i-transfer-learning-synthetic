# MetroPT-3: Transfer Learning & Synthetic Domain Bridging

## Dataset Description:
The MetroPT-3 dataset provides real-world predictive maintenance data collected from an Air Production Unit (APU) operating on an urban metro train. It captures continuous multivariate sensor readings (including pressure, temperature, and motor current) during actual operation, offering a practical testbed for industrial challenges like failure prediction and anomaly detection. 

## Stage 1: Exploratory Data Analysis
The raw MetroPT-3 dataset is entirely unlabeled. To perform supervised classification and transfer learning, we manually annotated the continuous sensor readings using the maintenance logs provided in the official data card. This process revealed four distinct "Air Leak" failure events spanning from April to July 2020. Our exploratory data analysis showed that normal operating data accounts for >98% of the dataset. Because this massive class imbalance overwhelms global distance metrics and masks the underlying physical concept drift, our methodology explicitly isolates and focuses the distance calculations exclusively on the anomalous datapoints.

## Domain Definition:
To evaluate transfer learning under different conditions, we initially defined three distinct domains: 
1. **Component Shift:** Partitioned using the `Towers` digital sensor to compare data generated while Tower 1 is active (Source) vs. Tower 2 (Target).
2. **Load Shift:** Partitioned using the `Motor_current` analog sensor to compare data generated while the compressor is operating 'Offloaded' (Source) vs. 'Under Load' (Target), based on the exact current thresholds provided in the data card.
3. **Seasonal Shift:** Partitioned using timestamps to compare anomalies occurring in the Spring (Source) vs. the Summer (Target).


## Stage 2: Domain Distance & Transfer Penalty
- For each of the three domain configurations, we calculated the geometric domain distance on the raw 7D sensor space using three optimal transport and kernel metrics: **Sliced Wasserstein Distance (SWD)**, **Maximum Mean Discrepancy (MMD)**, and **Fréchet Distance (FD)**. We also computed the Transfer Penalty (the drop in F1 score) using a Random Forest classifier.
- We observed a clear correlation: as the mathematical distance (SWD, MMD, FD) between the Source and Target domains increased, the Transfer Penalty worsened. 
- The **Spring $\rightarrow$ Summer** transition exhibited by far the most severe geometric distance across all metrics and the highest transfer penalty (~50%), proving that environmental concept drift severely degrades predictive maintenance models. Consequently, we selected this specific cross-season pair as the primary focus for our generative synthetic experiments.

## Stage 3: Synthetic Data Generation
- To bridge the Spring $\rightarrow$ Summer domain gap, we deployed a Conditional Variational Autoencoder (CVAE) and a Conditional GAN (CGAN). 
- We performed a domain interpolation sweep ($\alpha$ from 0.0 to 1.0) to generate synthetic anomalies. By injecting these synthetic anomalies into the Spring training set, we mathematically shifted the training distribution closer to the Summer target.
- **Result:** Both the CVAE and CGAN successfully learned to interpolate across the domain gap. As the mathematical distance (SWD, MMD, FD) between the augmented training set and the target set dropped, the Transfer Penalty plummeted. The adversarial training of the CGAN proved exceptionally effective, dropping the Transfer Penalty from ~50% down to a negligible **4.4%**.

## Limitations

- **Single Failure Mode:** The anomalies in MetroPT-3 are exclusively "Air Leak" failures. While we successfully proved domain bridging across different environmental conditions, this dataset does not allow us to test cross-failure transfer.
- **Limited Domain Pairs:** Currently, we only evaluated 3 Source-Target pairs based on broad seasonal splits. While this qualitatively demonstrates the relationship between distance and transfer degradation, a larger number of pairs is required to compute a rigorous quantitative correlation (e.g., Spearman/Kendall).
- **Target Data Usage:** The generative models currently utilize all available target anomalies during training to map the latent space. Future experiments are needed to evaluate their efficiency in a "few-shot" regime where only a tiny fraction of target data is available.
