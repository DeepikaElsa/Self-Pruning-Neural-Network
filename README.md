# Self-Pruning Neural Network  
### AI Engineering 

---

## Table of Contents
1. [Problem Overview](#1-problem-overview)  
2. [Why L1 Penalty → Sparsity](#2-why-l1-penalty--sparsity)  
3. [Architecture & Design Choices](#3-architecture--design-choices)  
4. [How to Run](#4-how-to-run)  
5. [Results Table](#5-results-table)  
6. [Sparsity–Accuracy Trade-off Analysis](#6-sparsityaccuracy-trade-off-analysis)  
7. [Sample Outputs](#7-sample-outputs)  
8. [Implementation Highlights](#8-implementation-highlights)  

---

## 1. Problem Overview  

Standard neural networks are trained and then pruned separately.  
This project implements **learning-time self-pruning**, where pruning and learning happen together.

Each weight has a learnable **gate**:
- Classification loss pushes important connections **up**
- L1 regularisation pushes unnecessary connections **down**

Result:
- Sparse model
- No separate pruning step
- Efficient training

---

## 2. Why L1 Penalty → Sparsity  

### Total Loss
```bash
Total Loss = CrossEntropy(logits, labels) + λ × Σ sigmoid(gate_scores)
```

### Intuition
Each gate:
```
g = sigmoid(score) ∈ (0, 1)
```

L1 adds constant pressure pushing gates toward zero.

| Regulariser | Gradient near g=0 | Behaviour |
|-------------|-------------------|----------|
| **L1**      | ±λ (constant)     | Drives exact zeros → sparsity |
| L2          | ~0                | Shrinks but rarely zeros |

### Key Idea
- If classification gradient < λ → gate → 0 → pruned  
- If important → gate stays active  

---

## 3. Architecture & Design Choices  

```
Input (3×32×32 CIFAR-10)
        │
[ Conv Backbone ]
Conv → BN → ReLU → Pool
Conv → BN → ReLU → Pool
Conv → BN → ReLU → AvgPool
        │
Flatten (2048)
        │
[ Prunable Classifier ]
PrunableLinear(2048 → 512)
PrunableLinear(512 → 10)
        │
Output
```

### Why only classifier is pruned?
- Conv pruning = unstructured → no speed gain  
- FC layers = effective + interpretable  

---

## 4. How to Run  

### Install dependencies
```bash
pip install torch torchvision tqdm matplotlib numpy
```

### Default training
```bash
python main.py
```

### Custom training
```bash
python main.py --epochs 20 --batch 256 --lr 5e-4 --patience 7
```

### Custom lambda sweep
```bash
python main.py --lambdas 1e-6 1e-4 1e-2
```

### Inference
```bash
python main.py --infer --checkpoint outputs/best_model.pt
```

### Outputs
- `best_model.pt`
- `model_lam*.pt`
- `gate_dist_lam*.png`
- `results_summary.png`
- `results.json`

---

## 5. Results Table  

| Lambda | Accuracy (%) | Sparsity (%) | Notes |
|--------|-------------|--------------|------|
| 1e-5   | ~82–84      | ~5–15        | Minimal pruning |
| 1e-4   | ~78–82      | ~40–60       | Best trade-off |
| 1e-3   | ~70–75      | ~80–92       | Heavy pruning |

---

## 6. Sparsity–Accuracy Trade-off  

### λ = 1e-5
- High accuracy  
- Low sparsity  

### λ = 1e-4 (best)
- Balanced pruning  
- Recommended  

### λ = 1e-3
- High sparsity  
- Accuracy drop  

### Key Observation
Good models show:
- Gates near **0 → pruned**
- Gates near **1 → active**

---

## 7. Sample Output  

```
λ = 1e-04 (10 epochs)

Ep 1  loss=1.83  train=42.3%  test=47.1%  sparse=0.12%
...
Ep10  loss=1.10  train=74.6%  test=73.8%  sparse=52.1%

Hard Pruning:
Pre  → acc=73.88% sparsity=52.10%
Post → acc=73.85% sparsity=52.10%
```

### Final Summary
```
Lambda     Accuracy     Sparsity
--------------------------------
1e-05       83.12%       8.41%
1e-04       73.85%      52.10%
1e-03       70.23%      88.67%
```

---

## 8. Implementation Highlights  

- Custom `PrunableLinear` layer  
- Fully differentiable gating mechanism  
- CrossEntropy + L1 loss  
- Hard pruning via threshold  
- Mixed precision (`torch.cuda.amp`)  
- Cosine LR scheduler  
- Early stopping  
- Gradient clipping  
- Reproducibility (seed control)  
- Efficient DataLoader  
- CLI support  
- JSON logging  
- Visualization support  

