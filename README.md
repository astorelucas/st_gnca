
# Graph Neural Cellular Automata (GNCA)

Official implementation of:

**Spatiotemporal Modeling with Graph Neural Cellular Automata for Modular Traffic Forecasting**

Published in *Expert Systems With Applications (2026)*.

---

## Overview

Graph Neural Cellular Automata (GNCA) is a modular framework for spatiotemporal forecasting that combines:

* Graph Neural Networks (GNNs)
* Graph Attention Networks (GATs)
* Cellular Automata principles
* Long Short-Term Memory (LSTM) forecasting

The framework was designed for traffic flow forecasting but can be adapted to other spatiotemporal domains such as:

* Environmental monitoring
* Air quality prediction
* Weather forecasting
* Epidemiological modeling
* Land-use and land-cover change analysis

---

## Key Features

✓ Modular forecasting architecture

✓ Graph Neural Cellular Automata formulation

✓ Multi-embedding representation learning

✓ Graph Attention based spatial encoding

✓ Neighborhood-aware tokenization

✓ Localized subgraph processing

✓ Fine-tuning on unseen graph regions

✓ Competitive performance on PEMS benchmarks

---

## Architecture

GNCA is composed of three main stages:

### 1. Multi-Embedding Representation

Each sensor observation is decomposed into:

#### Temporal Embedding (TE)

Encodes:

* Hour of day
* Day of week
* Day of year

using sinusoidal positional encoding.

#### Value Embedding (VE)

Traffic measurements are normalized using Z-score normalization and projected into a latent space.

#### Spatial Embedding (SE)

Graph topology is encoded through Laplacian Eigenmaps computed from the adjacency matrix.

---

### 2. Graph Attention Encoding

For each target node:

1. Extract a k-hop neighborhood subgraph.
2. Apply Graph Attention Networks (GAT).
3. Generate spatially-aware node representations.

Unlike conventional GNNs, GNCA processes local subgraphs independently, reducing computational overhead and reinforcing the cellular automata paradigm.

---

### 3. Cell Model

Each node is represented by a forecasting cell.

The cell consists of:

#### Neighborhood Tokenizer

Builds a spatiotemporal token using:

* Temporal features
* Central node embedding
* Aggregated neighborhood embedding

#### Forecasting Module

An LSTM network predicts future traffic flow values from the generated token sequence.

---

## Datasets

Experiments were conducted on four benchmark traffic forecasting datasets:

| Dataset | Nodes |
| ------- | ----- |
| PEMS03  | 358   |
| PEMS04  | 307   |
| PEMS07  | 883   |
| PEMS08  | 170   |

Each sample uses:

* Input sequence length: 12 timesteps
* Forecast horizon: 3–12 timesteps
* Train/Validation/Test split: 60/20/20

---

## Installation

### Clone repository

```bash
git clone https://github.com/yourusername/GNCA.git
cd GNCA
```

### Create environment

```bash
conda create -n gnca python=3.11
conda activate gnca
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training

Train GNCA on a selected dataset:

```bash
python training/train.py \
    --dataset PEMS08 \
    --epochs 50
```

---

## Evaluation

```bash
python training/evaluate.py \
    --checkpoint checkpoints/best_model.pt
```

---

## Fine-Tuning on New Communities

GNCA supports modular adaptation to unseen graph regions.

```bash
python training/finetune.py \
    --community C3 \
    --checkpoint checkpoints/pretrained.pt
```

This procedure allows the model to adapt to:

* Newly added sensors
* Evolving road networks
* Local traffic patterns

without retraining the entire graph model.

---

## Hyperparameters

Best-performing configuration:

| Parameter                    | Value |
| ---------------------------- | ----- |
| GAT Heads                    | 3     |
| Hidden Dimension             | 256   |
| LSTM Layers                  | 1     |
| Dropout                      | 0.10  |
| Temporal Embedding Dimension | 12    |
| Laplacian Components         | 20    |
| Learning Rate                | 1e-4  |

---

## Citation

```bibtex
@article{astore2026gnca,
  title={Spatiotemporal Modeling with Graph Neural Cellular Automata for Modular Traffic Forecasting},
  author={Astore, Lucas Malacarne and Silva, Gustavo Henrique Pinheiro da and Neto, Cayro Teixeira de Siqueira and Ayala, Daniel de Araujo and Bastos, Allana Tavares and Silva, Petronio Candido de Lima e and Orang, Omid and Guimaraes, Frederico Gadelha},
  journal={Expert Systems With Applications},
  volume={331},
  pages={133166},
  year={2026},
  publisher={Elsevier}
}
```

---

## License

This repository is released under the MIT License.

---

## Acknowledgements

Federal University of Minas Gerais (UFMG)

FutureLab, 

Kunumi,

Expert Systems With Applications (Elsevier)
