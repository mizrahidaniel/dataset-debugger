# Dataset Debugger 🔍🤖

Visual explorer and debugger for ML training datasets. Find mislabeled data, class imbalance, and outliers before they ruin your model.

## The Problem

ML models fail because of **bad training data**, but debugging datasets is painful:

- **Mislabeled examples** → Model learns garbage
- **Class imbalance** → Biased predictions
- **Outliers** → Hurt generalization
- **Data drift** → Model degrades over time

**Current tools:** Manual Jupyter notebooks, custom scripts, prayer.

## What It Does

**Automatic dataset health checks:**

✅ **Label Distribution** - Visualize class balance, detect severe imbalance  
✅ **Outlier Detection** - Flag suspicious examples (Isolation Forest)  
✅ **Duplicate Finder** - Catch identical or near-duplicate samples  
✅ **Label Quality** - Predict mislabeled examples (confident wrong predictions)  
✅ **Data Drift** - Compare train/val/test distributions  

## Quick Start

```bash
pip install dataset-debugger

# Analyze any dataset
dataset-debug images/ --labels labels.csv --output report/

# Interactive dashboard
dataset-debug images/ --dashboard --port 8080
```

## Example Output

```
🔍 Dataset Health Report
========================

📊 Label Distribution:
   cat: 1,247 (62.4%)  ⚠️ IMBALANCED
   dog: 753 (37.6%)

🚨 Outliers Detected: 23 suspicious examples
   - images/cat_001.jpg (anomaly score: 0.95)
   - images/dog_089.jpg (anomaly score: 0.92)

🏷️ Potential Mislabels: 15 examples
   - images/cat_042.jpg → Predicted: dog (confidence: 0.98)

📈 Data Drift: Validation set differs from train (KL divergence: 0.34)

💾 Full report saved to report/
```

## Tech Stack

- **Backend:** Python + scikit-learn (outlier detection, clustering)
- **Visualization:** Matplotlib, Plotly (interactive charts)
- **Dashboard:** Streamlit (quick prototyping) or FastAPI + React
- **ML:** Pre-trained models (CLIP for images, BERT for text) for label quality checks

## Roadmap

- [ ] Image dataset support (CV tasks)
- [ ] Text dataset support (NLP tasks)
- [ ] Tabular dataset support (regression, classification)
- [ ] Interactive dashboard with filtering
- [ ] Auto-fix suggestions (remove outliers, rebalance classes)
- [ ] Integration with Weights & Biases, MLflow

## Use Cases

- **Before training:** Catch data issues early
- **Model debugging:** "Why is my model underperforming?"
- **Data collection:** Validate new data before adding to dataset
- **Continuous monitoring:** Detect drift in production

## Contributing

Looking for:
- ML engineers (improve outlier detection algorithms)
- Data viz experts (better dashboard UX)
- CV/NLP specialists (pre-trained model integration)

## License

MIT
