# GNN Persuasion Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-1.9+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

This repository contains the implementation for "Modeling Persuasion in Reddit Conversations" - a project that uses Graph Neural Networks (GNNs) to predict persuasive comments in the Reddit Change My View (CMV) subreddit.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Setup and Usage](#setup-and-usage)
- [Future Work](#future-work)
- [Team Members](#team-members)
- [References](#references)

## 🔍 Overview

This study examines how conversation dynamics and user characteristics influence persuasion within the Change My View (CMV) subreddit. We employ Graph Neural Networks (GNNs) and fine-tuned BERT embeddings to predict the likelihood of a comment receiving a delta, an indicator of successful persuasion.

Our experiments reveal that GNN-based models, particularly the Distance-Weighted GNN, outperform baseline models by effectively leveraging conversation structure. Contrary to prior research emphasizing OP malleability, we find that enriched OP embeddings provide minimal additional benefits. These results suggest that capturing conversational depth and structure is more critical for persuasion modeling than OP-specific linguistic features alone.

## 📊 Dataset

The dataset is derived from the Change My View (CMV) subreddit, a structured online discussion platform where users engage in debates and award deltas to comments they find persuasive. The dataset consists of:
- Hundreds of thousands of comments
- Discussions from January 2013 to August 2015
- Explicit delta annotations identifying successful persuasion attempts

Each conversation is structured as a tree, with the Original Poster (OP) as the root node and subsequent replies forming branches. A comment is considered persuasive if the OP explicitly acknowledges it with a delta.

<p align="center">
  
  <img src="https://github.com/user-attachments/assets/0d9f5a94-f783-4a39-bfeb-b9864b5faae9" alt="Example Conversation Graph" width="60%">
  <br>
  <em>Figure 1: Example Conversation Graph with OP Nodes (as root) in black, Delta node in green and All Edges.</em>
</p>

## 🧠 Methodology

Our approach involves three primary modeling strategies to predict the likelihood of a comment receiving a delta:

### 1. Baseline Model
- Uses BERT embeddings for both comments and OPs
- Treats each node independently without incorporating graph-based propagation

### 2. Distance-Weighted GNN Model
- Assigns predefined edge weights based on a comment's depth within the conversation tree
- Leverages two Graph Convolutional Network (GCN) layers to aggregate features
- Enhances message passing by incorporating conversation structure

### 3. Edge-Weighted GNN Model
- Applies learnable Multi-Layer Perceptrons (MLPs) to transform both node features and edge attributes
- Allows for more dynamic message propagation across the conversation structure
- Incorporates edge attributes to capture relationship information

We explore different feature groups:
- Comment-only embeddings (BERT or GNN-based)
- Comment + OP embeddings from standard BERT
- Comment + OP embeddings from fine-tuned BERT

## 📈 Results

### Model Performance
Our experiments show that GNN-based models, especially the Distance-Weighted GNN, outperform baseline models by effectively leveraging conversation structure. The Distance-Weighted GNN shows greater resilience when predicting persuasion at deeper conversation levels compared to traditional BERT-based models.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e723ae20-dbb3-4695-8e04-4e0c3191b170" alt="Results by Model" width="60%">
  <br>
  <em>Figure 3: Results by Model and Input for Comment + OP Node enhanced</em>
</p>

### Key Findings
1. **Distance impact**: GNN-based models exhibit greater resilience when predicting persuasion at deeper conversation levels compared to traditional BERT-based models.
2. **Structure matters**: Distance-aware modeling strategies successfully preserve performance across increasing comment depths.
3. **OP embeddings**: Contrary to expectations, fine-tuned OP embeddings did not significantly enhance classification performance, suggesting that conversational context and structure are the dominant factors in determining persuasion success.

Performance metrics show:
- Models incorporating OP embeddings achieve slightly higher accuracy at distance 1
- However, their overall accuracy slopes decline more steeply than the comment-only model
- The F1-score positive slope decreases significantly with the addition of OP embeddings

| Model | Distance 1 Acc. | Acc. Slope | Dist. 1 F1 | F1 Slope |
|-------|----------------|------------|------------|----------|
| Only Comment Node | 0.77 | -0.003 | 0.251 | 0.005 |
| Comment Node + OP Node | 0.79 | -0.006 | 0.265 | 0.002 |
| Comment Node + OP Fine-Tuned Node | 0.79 | -0.004 | 0.262 | 0.001 |

## 🛠️ Setup and Usage

### Requirements
```
torch
torch_geometric
transformers
pandas
numpy
matplotlib
scikit-learn
```

### Installation
```bash
git clone https://github.com/denisrize/GNN-Persuasion-Prediction.git
cd GNN-Persuasion-Prediction
pip install -r requirements.txt
```

### Running the Experiments
To run the experiments with different models and feature configurations:

```bash
# Run baseline BERT model
python run_experiments.py --model bert --features comment_only

# Run Distance-Weighted GNN with comment and OP embeddings
python run_experiments.py --model distance_weighted_gnn --features comment_op

# Run Edge-Weighted GNN with fine-tuned OP embeddings
python run_experiments.py --model edge_weighted_gnn --features comment_op_finetuned
```

### Evaluating Results
The experiments script will output accuracy and F1-scores for different conversation depths, and will generate visualizations similar to those presented in the paper.

```bash
# Analyze results across different depths
python analyze_results.py --results_file results.json --output_folder visualizations
```

## 🔮 Future Work

- **Feature Engineering**: Explore additional features such as sentiment progression and user engagement metrics.
- **Advanced Graph Architectures**: Explore attention-based GNNs (like GATs) which dynamically weight influential comments.
- **Real-World Applications**: Potential applications include automated moderation, marketing analysis, and research in social psychology.

## 👥 Team Members
- **Denis Rize** - Data acquisition, cleaning, EDA, GNN architectures, and report writing
- **Gad Miller** - Design of GNN architectures, planning and execution of statistical experiments
- **Yuval Schwartz** - Data preprocessing, BERT fine-tuning, and report writing
- **Noam Azulay** - Model evaluation, visualization tools, and project presentations

## 📚 References
1. Tan, C., Niculae, V., Danescu-Niculescu-Mizil, C., & Lee, L. (2016). Winning arguments: Interaction dynamics and persuasion strategies in good-faith online discussions. *Proceedings of the 25th International Conference on World Wide Web*.
2. Wei, Z., Liu, Y., & Li, Y. (2016). Is this post persuasive? ranking argumentative comments in online forum. *Annual Meeting of the Association for Computational Linguistics*.
