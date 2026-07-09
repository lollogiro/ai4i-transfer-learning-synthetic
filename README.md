# Synthetic Data Generation for Transfer Learning

## 0. Dataset discovery

Different datasets have been identified, finally selecting the following ones:
* **SCANIA Component X Dataset**
* **MetroPT-3 Dataset**
* **3W Dataset**
* **Tennesse Eastman Process Dataset**

## 1. Exploratory Data Analysis (EDA)
All the notebook files in the `stage1-data_exploration` folder contain the EDA for each dataset, including:
* Data loading and preprocessing
* Data visualization
* Data analysis and statistics
* Data comparison between real and synthetic data
* Correlation analysis between features
* Final summary of the dataset and its characteristics

## 2. Methods for quantitative distance between distributions

Still in `stage1-data_exploration`, we have explored several methods to compute the distance between distributions, which can be used to compare real and synthetic data. These methods can be divided into two main categories:

1. Raw sample space distances:
   * **Earth Mover's Distance (EMD)**,
     * **Sliced Wasserstein Distance (SWD)**, better EMD approach for multivariate time-series
     * **Sinkhorn Divergence**, regularized version of EMD, faster to compute and differentiable, can be used as a loss function for training generative models
   * **Maximum Mean Discrepancy (MMD)**, distance between means of the two distributions in a reproducing kernel Hilbert space (RKHS), using the kernel trick 
     * **Multi-kernel MMD (MK-MMD)**, weighted sum of MMDs with different kernels, to capture different aspects of the distributions
     * **CORAL (Correlation Alignment)**, distance between the second-order statistics (covariance matrices) of the two distributions
   * **HoMM (Higher-order Moment Matching)**, distance between the higher-order moments of the two distributions, using tensor decomposition to compute the moments
   * **Hellinger Distance (HD)**, distance between the square root of the two distributions, can be used for discrete distributions
   * **Rényi Divergence (RD)**, generalization of Kullback-Leibler divergence, can be used for discrete and continuous distributions
2. Feature space distances:
   Possibly using pre-trained networks to extract features from the raw data, and then computing distances between the feature distributions, such as MOMENT, Chronos (Amazon), TimesFM (Google), TabFM (Google new)
   * **Proxy A-distance (PAD)**, distance between two distributions based on the error of a classifier trained to distinguish them, if accuracy is high, distributions are far apart
   * **Fréchet Distance (FD)**, distance between two multivariate Gaussians fitted to the feature space of a pre-trained network
     * **Kernel Distance (KD)**, unbiased estimator of MMD in the feature space of a pre-trained network

## 3. Domain adaptation and transfer learning scenarios

Some scenarios have been explored to use transfer learning for synthetic data generation, including:
* **Simulated source -> Real target**: using a simulated dataset as source and a real dataset as target, to generate synthetic data that is closer to the real data distribution
* **Real source (but far) -> Real target (specific)**: using a real dataset as source, but far from the target distribution, to generate synthetic data that is closer to the target distribution
* **Mixed source -> Real target**: using a mixed dataset as source, including real and simulated data, to generate synthetic data that is closer to the target distribution

## 4. Synthetic data generation

Possible generative models:
* **TimeVAE**, a variational autoencoder for time-series data, can be used to generate synthetic data that is similar to the real data distribution
* **Conditional GANs**, generative adversarial networks that can generate synthetic data conditioned on some input, such as class labels (most common).