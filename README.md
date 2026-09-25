# Fake News Detection using Graph Attention Networks and QGWO

A graph-based fake news detection approach using **Graph Attention Networks (GAT)** combined with **Quantum-inspired Grey Wolf Optimization (QGWO)** for model optimization and misinformation classification.

## Overview

This project explores graph neural network approaches for fake news detection by representing textual information as graphs and applying attention-based graph learning.

The pipeline:
Text Data
↓
Graph Construction
↓
Node Feature Extraction
↓
Graph Attention Network (GAT)
↓
QGWO Optimization
↓
Fake / Real Classification

## Technologies

- Python
- PyTorch
- PyTorch Geometric
- Graph Neural Networks (GNN)
- Graph Attention Networks (GAT)
- Natural Language Processing (NLP)
- Scikit-learn
- TF-IDF
- Hyperparameter Optimization

## Key Features

✓ Graph-based text representation  
✓ Multi-head Graph Attention Network architecture  
✓ Quantum-inspired Grey Wolf Optimization (QGWO)  
✓ Evaluation on multiple fake news datasets  
✓ Reproducible experimental pipeline  

## Results

The model was evaluated using held-out test splits across multiple datasets.

| Dataset | Accuracy | F1-score (Macro) | ROC-AUC |
|---|---:|---:|---:|
| Synthetic PolitiFact | 91.01% | 86.07% | 92.49% |
| GossipCop++ | 80.28% | 80.24% | 85.58% |
| PolitiFact (New) | 79.05% | 78.94% | 86.89% |
| PolitiFact++ | 78.67% | 78.36% | 93.40% |
| GossipCop | 68.61% | 68.60% | 76.28% |
| All Text Sources | 64.17% | 64.15% | 70.93% |
| FactCheck Binary | 61.39% | 61.36% | 66.49% |

## Methodology

### Graph Attention Network (GAT)
Used for learning graph-based representations through attention mechanisms, allowing the model to capture relationships between textual samples.

### Quantum-inspired Grey Wolf Optimization (QGWO)
Applied as a metaheuristic optimization technique for selecting improved model configurations.

### Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Project Structure
Fake-News-Detection-GAT-QGWO/
│
├── fake_news_detection_gat_qgwo.ipynb
├── README.md
├── requirements.txt
└── LICENSE


## Future Improvements

- Compare against baseline ML and deep learning models
- Add explainability methods (XAI)
- Experiment with transformer-based graph representations
- Deploy inference API
