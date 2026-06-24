# Cloud Workload Prediction

**B.Tech Capstone Project — Indian Institute of Technology Ropar**
**Authors:** Harsh Srova (2018CSB1095) · Guzzarlapudi Stephen Sugun (2018CSB1225)
**Supervisor:** Dr. Shweta Jain, Department of Computer Science & Engineering
**Submitted:** April 2022

---

## Overview

Cloud infrastructure demand is growing at an unprecedented rate, with the global market projected to nearly double from $445B to $947B by 2026 (CAGR 16.3%). As user bases surge — exemplified by Zoom's 326% user growth and Microsoft Teams adding 44 million users during COVID-19 — cloud providers need reliable, low-error workload prediction to enable **dynamic resource scaling** and **efficient energy management**.

This project addresses cloud server workload prediction using the **Transformer architecture**, benchmarked against the widely-used LSTM model on the NASA HTTP server log dataset. The Transformer consistently outperforms LSTM across all tested prediction window sizes, achieving a best MSE of **2.95 × 10⁻³**.

---

## Motivation

Two core problems drive the need for accurate workload prediction:

**Dynamic Resource Scaling** — Cloud servers must maintain consistent Quality of Service (QoS) across varying user loads and times of day. Accurate prediction enables proactive resource allocation rather than reactive scaling.

**Efficient Energy Management** — Over-provisioned servers waste energy and cost. Accurate forecasts allow servers to run lean, reducing operational costs and environmental footprint.

---

## Approach

### Model: Vanilla Transformer (Encoder-Decoder)

The project applies the vanilla Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017) to time-series workload forecasting. Key properties:

- **Self-attention** captures long-range temporal dependencies that LSTMs struggle with due to the vanishing gradient problem.
- **Multi-head attention** runs parallel attention blocks across multiple cores, enabling faster training and inference compared to sequential LSTM processing.
- **Sin-cos positional encoding** is applied to the input before feeding it into the encoder, preserving temporal order information.

The encoder processes the historical workload sequence; the decoder autoregressively predicts the next timestep's workload using previously predicted outputs.

### Why Transformers Beat LSTMs Here

LSTMs suffer from the **vanishing gradient problem** — as network depth increases, early-sequence information degrades, harming long-range predictions. Transformers avoid this entirely through direct attention over all timesteps, and their parallelism means they train faster on the same hardware.

---

## Dataset

**NASA HTTP Server Log Dataset** — 3.26 million data points with fields: Host, Time, Method, Response, URL, Bytes.

Preprocessing steps:
- Extracted a `load` attribute: number of user requests arriving per unit time.
- Split into two temporal granularities:
  - **Seconds dataset** — finer-grained, larger, trains a more accurate model.
  - **Minutes dataset** — coarser-grained, smaller, slightly higher error.

**Train/test split:** 60% training / 40% testing.

---

## Technology Stack

| Component | Tool |
|-----------|------|
| Language | Python |
| Deep Learning | PyTorch |
| Visualization | Matplotlib |
| Data Processing | Pandas, NumPy |
| Training Hardware | Google Colab (1× Tesla K80, 12 GB GDDR5, 2496 CUDA cores) |

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 50 |
| Learning rate | 0.005 |
| Training Window Size (TWS) | 100 |
| Prediction Window Size (PWS) | 1, 5, 10, 20, 30, 60 |
| Optimal number of layers | 1 |
| Loss function | Mean Squared Error (MSE) |

**Key finding on layers:** A single encoder-decoder layer dramatically outperforms deeper configurations. Adding layers (2–5) causes MSE to jump from ~3 × 10⁻³ to ~68 × 10⁻³ due to the vanishing gradient problem compounding with the relatively small dataset size.

---

## Results

### Transformer vs. LSTM — MSE by Prediction Window (×10⁻³)

| PWS | Transformer (seconds) | Transformer (minutes) | LSTM |
|-----|----------------------|-----------------------|------|
| 1   | 3.68 | 4.47 | 13.06 |
| 5   | 2.66 | 3.50 | 4.79  |
| 10  | 2.29 | 3.52 | 6.66  |
| 20  | 2.15 | 3.12 | 7.01  |
| 30  | 2.05 | 3.01 | 6.43  |
| 60  | **1.89** | **2.95** | 5.59 |

The Transformer (seconds) achieves the best result at **MSE = 1.89 × 10⁻³** (PWS=60), compared to LSTM's best of 4.79 × 10⁻³ — roughly **2.5× lower error**.

The seconds dataset consistently outperforms the minutes dataset because its larger size gives the model more training signal.

### MSE by Number of Transformer Layers (×10⁻³, minutes dataset, PWS=60)

| Layers | MSE |
|--------|-----|
| 1 | **2.62** |
| 2 | 63.68 |
| 3 | 63.78 |
| 4 | 63.66 |
| 5 | 63.71 |

Deeper models collapse in performance — a clear signal to keep the architecture shallow for this dataset size.

---

## Key Improvements Over Prior Work

**Accuracy:** Hyperparameter tuning (batch size, learning rate), dataset normalization, and cyclic temporal modelling to capture inherent periodic patterns in server load. This helps the model identify long-term trends more effectively.

**Speed:** Reduced the number of layers after observing that deeper models both increase error and slow training due to vanishing gradients. Transformer's native parallelism in multi-head attention further speeds up computation relative to sequential LSTM processing.

---

## Conclusions

Transformers outperform LSTMs for cloud workload time-series prediction across all tested prediction window sizes, delivering lower prediction error, faster training, and faster inference. A single-layer Transformer trained on second-level granularity data achieves the best results. These findings support the adoption of Transformer-based forecasting for dynamic cloud resource scaling and energy-efficient infrastructure management.

---

## Repository Structure

```
.
├── data/
│   ├── nasa_http_log/          # Raw NASA HTTP server log dataset
│   ├── seconds_dataset.csv     # Preprocessed per-second workload
│   └── minutes_dataset.csv     # Preprocessed per-minute workload
│
├── src/
│   ├── preprocess.py           # Data loading, load feature extraction, normalization
│   ├── model.py                # Vanilla Transformer (encoder-decoder) in PyTorch
│   ├── train.py                # Training loop, hyperparameter configuration
│   └── evaluate.py             # MSE computation and comparison vs. LSTM baseline
│
├── notebooks/
│   └── experiments.ipynb       # Google Colab notebook with full experiment runs
│
├── results/
│   ├── mse_comparison.png      # Transformer vs. LSTM MSE plot
│   └── mse_by_layers.png       # MSE vs. number of layers plots
│
├── report/
│   └── Cloud_Workload_Prediction.pdf
│
└── README.md
```

---

## References

- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS 2017*.
- Kumar, J., Goomer, R., & Singh, A.K. (2017). LSTM-RNN Based Workload Forecasting Model for Cloud Datacenters. *ICSCC 2017*.
- Guhr, O. (2021). Transformer Time-Series Prediction. GitHub.
- Klingenbrunn, N. (2021). Transformers for Time-Series Forecasting. Medium / GitHub.
- Svensson, L. (2020). Self-Attention — An Introduction. Chalmers University.

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
