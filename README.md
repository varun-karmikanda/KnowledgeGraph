# 🧠 Sentence Boundary Detection in Knowledge Graphs via Entropy

> **Hackathon Project | Team Datanauts**

This project explores the problem of detecting **sentence boundaries** without relying on raw text or punctuation. Instead, we build a **Knowledge Graph (KG)** from SVO (Subject-Verb-Object) triplets and train a model to identify sentence limits using **entropy-based traversal**.

---

## 📌 Problem Statement

Given a paragraph, we:
1. Extract SVO triplets from each sentence.
2. Construct a directed Knowledge Graph:
   - **Nodes** = Subjects & Objects
   - **Edges** = Verbs (directed from Subject → Object)
3. Detect sentence boundaries using **semantic entropy** during traversal — without access to punctuation or original sentence splits.

Inspired by the [Byte Latent Tokenizer (BLT)](https://arxiv.org/abs/2310.08560), we adapt and train its entropy model to operate over node sequences instead of visual patch tokens.

---

## 🛠️ Technologies Used

- `spaCy` + `BeautifulSoup`: SVO triplet extraction
- `networkx` + `matplotlib`: Graph construction & visualization
- `Node2Vec`: Node embedding generation
- `BLT`: Entropy model for traversal stopping
- `scikit-learn`: Evaluation metrics

---

## 🚀 Progress & Results

| Metric                    | Value      |
|--------------------------|------------|
| F1 Score (current)       | 0.0215     |
| Sentence Boundaries Found| 3,304 / 23,115 |
| Traversal Efficiency     | 1.00       |
| Training Epochs          | 5          |

We visualize entropy levels and predicted boundaries across the KG to compare with ground truth.

---

## 📁 Project Structure

```bash
KnowledgeGraph/
├── blt/                   # BLT base code (forked)
├── blt_integration/       # Custom boundary detection & evaluation
├── data/                  # Raw text, triplets, embeddings, ground truth
├── outputs/               # Predicted boundaries & visualizations
├── svo_extractor.py       # Extracts SVO triplets from HTML
├── build_graph.py         # Constructs KG from triplets
├── environment.yml        # Conda environment definition
└── requirements.txt       # Python dependencies
