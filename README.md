# Fake News Detection using Graph Attention Networks and QGWO
Fake news detection using Graph Attention Networks (GAT) with Quantum-inspired Grey Wolf Optimization (QGWO) for model optimization and improved classification performance.

## Overview

This project explores graph-based learning methods for misinformation detection.

The pipeline:
Text → Graph Construction → GAT Model → QGWO Optimization → Classification

## Technologies

- Python
- PyTorch
- Graph Neural Networks
- GAT
- NLP
- Scikit-learn
- TF-IDF
- Hyperparameter Optimization

## Key Features

✓ Document graph construction
✓ Multi-head Graph Attention Network
✓ Quantum-inspired optimization
✓ Multiple dataset evaluation
✓ Reproducible experiments

## Results
| Dataset              | Accuracy | F1-score (Macro) | ROC-AUC |
| -------------------- | -------: | ---------------: | ------: |
| Synthetic PolitiFact |   91.01% |           86.07% |  92.49% |
| GossipCop++          |   80.28% |           80.24% |  85.58% |
| PolitiFact (New)     |   79.05% |           78.94% |  86.89% |
| PolitiFact++         |   78.67% |           78.36% |  93.40% |
| GossipCop            |   68.61% |           68.60% |  76.28% |
| All Text Sources     |   64.17% |           64.15% |  70.93% |
| FactCheck Binary     |   61.39% |           61.36% |  66.49% |


## Methodology

The project uses:

Graph Attention Network (GAT) for graph-based representation learning
Quantum-inspired Grey Wolf Optimization (QGWO) for optimization
Text-based graph construction for misinformation classification
Multiple evaluation metrics:
Accuracy
Precision
Recall
F1-score
ROC-AUC

