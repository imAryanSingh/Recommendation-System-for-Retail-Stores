# Customer Segmentation & Recommendation System for Retail Stores

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-KMeans-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3D_Viz-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-EDA-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-1D9E75?style=for-the-badge)
![IIT Ropar](https://img.shields.io/badge/Built_at-IIT_Ropar-orange?style=for-the-badge)

**K-Means customer segmentation on 200 mall customers — three clustering experiments across age, income, and spending dimensions, with 3D interactive visualisation. Segments directly map to targeted product recommendations.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imAryanSingh/Recommendation-System-for-Retail-Stores/blob/main/RecomndatonSysRetailStore.ipynb)

[Overview](#overview) · [Dataset](#dataset) · [Segmentation Results](#segmentation-results) · [How Segments Map to Recommendations](#how-segments-map-to-recommendations) · [Visualisations](#visualisations) · [Setup](#setup) · [Code Fixes](#code-fixes)

</div>

---

## Overview

Retail stores lose revenue by treating every customer the same. A customer who earns $120k/year and spends freely needs completely different recommendations than one who earns $20k and shops carefully. This project uses **K-Means clustering** to segment 200 mall customers into distinct behavioural groups — each segment then maps directly to a targeted recommendation strategy.

The project runs **three independent clustering experiments**, progressively adding dimensions to discover richer customer patterns:

| Experiment | Features used | Optimal K | Business insight |
|-----------|--------------|-----------|-----------------|
| 1 | Age × Spending Score | **4 clusters** | Young high-spenders vs cautious older customers |
| 2 | Annual Income × Spending Score | **5 clusters** | The classic 5-segment retail model |
| 3 | Age × Annual Income × Spending Score | **6 clusters** | Full 3D behavioural profiling (interactive Plotly) |

---

## Dataset

**Mall Customers Dataset** — 200 retail mall customers

| Feature | Type | Range | Mean |
|---------|------|-------|------|
| CustomerID | Integer | 1 – 200 | — |
| Gender | Categorical | Male / Female | 112F / 88M |
| Age | Integer | 18 – 70 | 38.8 years |
| Annual Income (k$) | Integer | $15k – $137k | $60.6k |
| Spending Score (1-100) | Integer | 1 – 99 | 50.2 |

```
200 customers · 5 features · 0 missing values · 112 Female / 88 Male
Age range: 18–70 · Income range: $15k–$137k · Spending score: 1–99
```

**Key correlations found during EDA:**
- Annual Income and Spending Score have near-zero correlation overall — but strong within clusters
- Young customers (18–30) show bimodal spending: either very high OR very low
- Female customers slightly outspend male customers at equivalent income levels

---

## Segmentation Results

### Experiment 1 — Age × Spending Score (K=4)

Elbow method identifies **4 optimal clusters**:

```
Cluster  │  Age range  │  Spending Score  │  Segment name
─────────┼─────────────┼──────────────────┼──────────────────────
  0      │  20 – 40    │     60 – 99      │  Young High Spenders
  1      │  40 – 70    │     40 – 60      │  Mature Moderate Spenders
  2      │  18 – 35    │      1 – 40      │  Young Cautious Buyers
  3      │  30 – 65    │     60 – 99      │  Mid-Age High Spenders
```

### Experiment 2 — Annual Income × Spending Score (K=5)

This is the **most business-relevant segmentation** — elbow identifies **5 classic retail segments**:

```
Cluster  │  Income     │  Spending Score  │  Segment name              │  Size
─────────┼─────────────┼──────────────────┼────────────────────────────┼──────
  0      │  Low        │  Low (1–40)      │  Careful Customers          │  ~35
  1      │  Low        │  High (60–99)    │  Impulsive Customers        │  ~22
  2      │  Medium     │  Medium (40–60)  │  Standard Customers         │  ~78
  3      │  High       │  Low (1–40)      │  Conservative High-Earners  │  ~22
  4      │  High       │  High (60–99)    │  Target Customers ⭐        │  ~23
```

> **Cluster 4 (Target Customers)** = the highest-value retail segment. High income + high willingness to spend. These customers should receive premium product recommendations and loyalty benefits.

### Experiment 3 — Age × Income × Spending Score (K=6, 3D)

Three-dimensional segmentation reveals **6 distinct clusters** visualised in an interactive 3D Plotly scatter plot. The additional age dimension splits the "Standard Customers" group into younger and older sub-segments, enabling age-appropriate recommendations on top of income-based ones.

---

## How Segments Map to Recommendations

The clusters aren't just interesting patterns — they directly drive a recommendation strategy:

```
Customer arrives at store
         │
         ▼
┌─────────────────────┐
│  Predict segment    │  Input: Age, Income, Spending Score
│  (trained KMeans)   │  Output: Cluster ID (0–4)
└────────┬────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│                  RECOMMENDATION STRATEGY BY SEGMENT            │
├──────────────────────┬─────────────────────────────────────────┤
│  Cluster 0 (Careful) │  Budget products, value packs, discounts│
│  Cluster 1 (Impulse) │  Flash deals, limited stock alerts      │
│  Cluster 2 (Standard)│  Popular items, bestsellers             │
│  Cluster 3 (Conserv.)│  Premium brands, quality over quantity  │
│  Cluster 4 (Target ⭐)│  Luxury items, premium bundles         │
└──────────────────────┴─────────────────────────────────────────┘
```

**Predicting a new customer's segment:**
```python
import numpy as np
from sklearn.cluster import KMeans
import joblib

# Load trained model
model = joblib.load('kmeans_income_spending.pkl')

# New customer: Annual Income = $85k, Spending Score = 72
new_customer = np.array([[85, 72]])
cluster = model.predict(new_customer)[0]

segment_names = {
    0: "Careful Customer",
    1: "Impulsive Customer",
    2: "Standard Customer",
    3: "Conservative High-Earner",
    4: "Target Customer"
}
print(f"Segment: {segment_names[cluster]}")
# Output: Segment: Target Customer
```

---

## Visualisations

The notebook generates **10 visualisation types**:
<img width="1383" height="639" alt="image" src="https://github.com/user-attachments/assets/285fd752-35d7-4f06-8077-1cc03f264950" />
 5-cluster Income × Spending plot
| Plot | What it shows |
|------|-------------|
| Distribution plots (3 features) | Age, Income, Spending Score distributions |
| Gender count plot | 112 Female / 88 Male breakdown |
| 3×3 regression grid | Pairwise relationships between all numeric features |
| Gender scatter: Age vs Income | Income patterns by age, split by gender |
| Gender scatter: Income vs Spending | Spending behaviour by income, split by gender |
| Violin + swarm plots | Feature distributions by gender with individual points |
| Elbow curve (Experiment 1) | K=1–10 inertia to select K=4 |
| Cluster boundary map (Experiment 1) | Decision regions: Age × Spending |
| Elbow curve + cluster map (Experiment 2) | K=5 selection + Income × Spending clusters |
| **3D Plotly scatter (Experiment 3)** | Interactive 3D: rotate/zoom Age × Income × Spending |

---

## Project Structure

```
Recommendation-System-for-Retail-Stores/
│
├── RecomndatonSysRetailStore.ipynb    ← Main Jupyter notebook (41 cells)
├── Mall_Customers.csv                  ← Dataset: 200 customers, 5 features
├── requirements.txt                    ← Python dependencies
├── .gitignore
└── README.md
```

---

## Setup

### Run in Google Colab (recommended — zero setup)

Click the badge at the top of this README:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imAryanSingh/Recommendation-System-for-Retail-Stores/blob/main/RecomndatonSysRetailStore.ipynb)

Then upload `Mall_Customers.csv` when prompted, or mount Google Drive.

### Run locally

```bash
# 1. Clone the repo
git clone https://github.com/imAryanSingh/Recommendation-System-for-Retail-Stores.git
cd Recommendation-System-for-Retail-Stores

# 2. Install dependencies
pip install -r requirements.txt

# 3. Fix the dataset path in Cell 4 (see Code Fixes below)

# 4. Launch
jupyter notebook RecomndatonSysRetailStore.ipynb
```

### requirements.txt

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.12.0
scikit-learn>=1.0.0
plotly>=5.0.0
jupyter>=1.0.0
```

---

## Code Fixes

Three lines in the notebook need updating for modern library versions and local use:

### Fix 1 — Dataset path (Cell 4)

```python
# CURRENT (Colab-only — breaks locally):
dataset = pd.read_csv('/content/Mall_Customers.csv')

# FIX (works everywhere):
import os
dataset = pd.read_csv('Mall_Customers.csv')   # if running from repo root
```

### Fix 2 — Deprecated `sns.distplot` (Cell 17)

`sns.distplot` was removed in Seaborn 0.12+. Replace with `sns.histplot`:

```python
# CURRENT (raises FutureWarning / error in seaborn >= 0.12):
sns.distplot(dataset[x], bins=20)

# FIX:
sns.histplot(dataset[x], bins=20, kde=True)
```

### Fix 3 — Deprecated KMeans `algorithm='elkan'` (Cells 27, 33, 39)

```python
# CURRENT (deprecated in scikit-learn 1.1+):
algorithm = KMeans(n_clusters=4, ..., algorithm='elkan')

# FIX:
algorithm = KMeans(n_clusters=4, ..., algorithm='lloyd')
```

---

## Key Concepts Explained

**Why K-Means for customer segmentation?**
K-Means partitions customers into K groups where each customer belongs to the cluster with the nearest centroid. It's fast (O(n·k·i)), interpretable, and produces actionable segments — unlike deep learning models, you can directly describe what each cluster means in business terms.

**Why the Elbow Method?**
Inertia (sum of squared distances from each point to its cluster centroid) always decreases as K increases — but the rate of decrease sharply slows after the optimal K. The "elbow" in the inertia curve marks this point. Experiment 1 → K=4, Experiment 2 → K=5, Experiment 3 → K=6.

**Why k-means++ initialisation?**
Random centroid initialisation often leads to poor convergence (local minima). `init='k-means++'` spreads initial centroids far apart, giving much more consistent cluster quality across random seeds.

---

## Technologies

| Library | Version | Purpose |
|---------|---------|---------|
| Pandas | ≥1.3 | Data loading, EDA, describe/info/corr |
| NumPy | ≥1.21 | Array operations, meshgrid for decision boundaries |
| Scikit-learn | ≥1.0 | KMeans clustering, elbow computation |
| Matplotlib | ≥3.4 | All 2D plots, cluster boundary visualisation |
| Seaborn | ≥0.12 | Distplots, violin plots, swarm plots, regplots |
| Plotly | ≥5.0 | Interactive 3D scatter for 3-feature clustering |

---

## About the Author

**Aryan Singh** — AI/ML Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-im--aryan--singh-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/im-aryan-singh)
[![GitHub](https://img.shields.io/badge/GitHub-imAryanSingh-181717?style=flat&logo=github)](https://github.com/imAryanSingh)
[![Portfolio](https://img.shields.io/badge/Portfolio-imAryanSingh.github.io-534AB7?style=flat)](https://imAryanSingh.github.io)

*Developed during AI Vicharana Shala residential programme at IIT Ropar (May–Jul 2024)*
*B.Tech CSE · Mohanlal Sukhadia University · GATE 2026 (88.31 percentile)*

---

## Also see

- [Wake-Word Detection — ISRO TRISHNA Satellite](https://github.com/imAryanSingh/Wakeup-Word-Detection-Model-for-voice-commanding-system)
- [Wildfire Prediction from Satellite Imagery](https://github.com/imAryanSingh/Wildfire-Prediction-Using-Satellite-Image-GSoC)
- [Smart Vision Quality Control — Top 0.3% Flipkart GRID](https://github.com/imAryanSingh/Smart-Vision-Technology-Quality-Control)
- [Image · Audio · Emoji Steganography](https://github.com/imAryanSingh/Steganography)
- [E-Commerce Sales Dashboard](https://github.com/imAryanSingh/E-COMMERCE-SALES-DASHBOARD)
