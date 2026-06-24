# Cloud Workload Prediction

**B.Tech Capstone Project — Indian Institute of Technology Ropar**
**Authors:** Harsh Srova (2018CSB1095) · Guzzarlapudi Stephen Sugun (2018CSB1225)
**Supervisor:** Dr. Shweta Jain, Department of Computer Science & Engineering
**April 2022**

---

## Overview

This project addresses cloud server workload prediction using the **Transformer model** (vanilla architecture from "Attention Is All You Need"), applied to the NASA HTTP server log dataset. The goal is to enable **dynamic resource scaling** and **efficient energy management** in cloud systems by accurately forecasting future workloads.

We compare Transformer-based models against Long Short-Term Memory (LSTM) networks, demonstrating that Transformers achieve lower prediction error, faster training, and faster inference — making them better suited for this time-series forecasting task.

**Best result: MSE of 2.95 × 10⁻³** (Transformer on seconds dataset, PWS = 60)

---

## Motivation

Cloud demand is growing at a CAGR of 16.3%, projected to reach $947 billion by 2026. As user counts scale, cloud systems must dynamically provision resources to maintain Quality of Service. Accurate workload prediction enables:

- **Dynamic resource scaling** — allocate resources proportionally to real-time demand
- **Efficient energy management** — reduce unnecessary power consumption and operating costs

---

## Approach

### Model: Vanilla Transformer

The model follows the encoder-decoder architecture from Vaswani et al. (2017):

```
Input Workload
      │
      ▼
Input Embedding + Positional Encoding
      │
      ▼
 ┌─────────────┐
 │   Encoder   │  (Multi-Head Self-Attention → Feed Forward) × N layers
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │   Decoder   │  (Masked MHA → Multi-Head Attention → Feed Forward) × N layers
 └──────┬──────┘
        │
        ▼
  Linear + Softmax
        │
        ▼
 Predicted Workload (next timestep)
```

The decoder is auto-regressively fed previously predicted outputs alongside encoder context to generate future workload probabilities.

### Key Improvements Over Prior Work

- **Cyclic temporal modelling** applied to capture inherent cyclical/periodic patterns in the dataset
- **Dataset normalization** for more stable and efficient training
- **Hyperparameter tuning** (batch size, learning rate) to find the optimal configuration
- **Reduced layer depth** — empirically found that a single encoder/decoder layer minimizes error due to the vanishing gradient problem on this dataset size

---

## Technology Stack

| Language | Frameworks / Libraries |
|----------|------------------------|
| Python | PyTorch, NumPy, Pandas, Matplotlib |

---

## Dataset

**NASA HTTP Server Log** — publicly available server access log dataset.

- **Raw size:** 3.26 million data points
- **Attributes:** Host, Time, Method, Response, URL, Bytes
- **Preprocessing:** Extracted a `load` attribute (number of user requests per unit time), then split into two temporal granularities:
  - **Seconds dataset** — unit time = 1 second
  - **Minutes dataset** — unit time = 1 minute
- **Train/test split:** 60% training / 40% testing

---

## Experiments & Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 50 |
| Learning rate | 0.005 |
| Training Window Size (TWS) | 100 |
| Prediction Window Sizes (PWS) tested | 1, 5, 10, 20, 30, 60 |
| Loss function | Mean Squared Error (MSE) |
| Training hardware | Google Colab — Tesla K80 GPU (12 GB GDDR5), Xeon CPU @ 2.3 GHz |

**PWS** = number of previous timesteps used as input to predict the next workload value.
**TWS** = number of previous timesteps used during training.

---

## Results

### Transformer vs. LSTM (MSE × 10⁻³)

| PWS | Transformer (seconds) | Transformer (minutes) | LSTM |
|-----|----------------------|-----------------------|------|
| 1   | 3.68 | 4.47 | 13.06 |
| 5   | 2.66 | 3.50 | 4.79  |
| 10  | 2.29 | 3.52 | 6.66  |
| 20  | 2.15 | 3.12 | 7.01  |
| 30  | 2.05 | 3.01 | 6.43  |
| 60  | **1.89** | **2.95** | 5.59  |

Transformers consistently outperform LSTMs across all PWS values. The seconds dataset outperforms the minutes dataset due to its larger size enabling better generalization.

### Effect of Number of Layers (MSE × 10⁻³, minutes dataset)

| PWS | Layers = 1 | Layers = 2 | Layers = 3 | Layers = 4 | Layers = 5 |
|-----|-----------|-----------|-----------|-----------|-----------|
| 1   | **3.30**  | 68.58 | 68.72 | 68.55 | 68.62 |
| 60  | **2.62**  | 63.68 | 63.78 | 63.66 | 63.71 |

A single-layer transformer dramatically outperforms deeper variants. Increasing depth causes severe vanishing gradient degradation on this dataset size, pushing error ~20× higher.

---

## Conclusions

- Transformer models outperform LSTMs for cloud workload time-series prediction in terms of accuracy, training speed, and inference speed.
- A **single-layer transformer** is optimal for this dataset — deeper models suffer from vanishing gradients.
- Longer prediction window sizes (higher PWS) consistently reduce error as the model captures more temporal context.
- The approach can meaningfully improve cloud resource efficiency and reduce energy costs.

---

## Repository Structure

```
.
├── data/
│   ├── raw/                  # NASA HTTP server log (raw)
│   └── processed/            # Seconds and minutes datasets after preprocessing
│
├── src/
│   ├── preprocess.py         # Data loading, cleaning, load attribute extraction
│   ├── model.py              # Vanilla Transformer architecture (PyTorch)
│   ├── train.py              # Training loop, hyperparameter configuration
│   └── evaluate.py           # MSE evaluation and result plots
│
├── results/
│   ├── mse_comparison.png    # Transformer vs LSTM MSE chart
│   └── layers_comparison.png # MSE by number of layers
│
├── Cloud_Workload_Prediction.pdf   # Full project report
└── README.md
```

---

## References

- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS 2017.
- Kumar, J. et al. (2017). *LSTM-RNN Based Workload Forecasting Model for Cloud Datacenters.* ICSCC 2017.
- Guhr, O. (2021). [Transformer Time-Series Prediction](https://bit.ly/3M2mZGs). GitHub.
- Klingenbrunn, N. (2021). [Transformers for Time-Series Forecasting](https://bit.ly/3yo1uMq). Medium.

---

## Citation

```bibtex
@misc{srova2022cloudworkload,
  title     = {Cloud Workload Prediction},
  author    = {Srova, Harsh and Guzzarlapudi, Stephen Sugun},
  year      = {2022},
  school    = {Indian Institute of Technology Ropar},
  note      = {B.Tech Capstone Project (CP303), supervised by Dr. Shweta Jain}
}
```
