# Company Bankruptcy Prediction

Binary classification of company bankruptcy using 95 financial ratio features from the Taiwan Economic Journal. Built for the MF815 Machine Learning course (Boston University, Spring 2026).

## Results

**Best Model: XGBoost + ECOD outlier features — Val F1 = 0.5060 (bankrupt class)**

| Model | Val F1 (Bankrupt) | AUC-ROC |
|-------|-------------------|---------|
| XGBoost + ECOD | **0.5060** | 0.9618 |
| XGBoost (tuned) | 0.4706 | 0.9618 |
| LightGBM (tuned) | 0.4483 | 0.9605 |
| Logistic Regression | 0.4337 | 0.9155 |
| Random Forest (tuned) | 0.3944 | 0.9489 |

## Key Techniques

- **Class imbalance handling**: `scale_pos_weight` outperformed SMOTE and its variants
- **ECOD outlier detection**: Added as engineered feature, improved F1 by +0.035
- **Optuna Bayesian optimization**: 200-trial TPE search for XGBoost hyperparameters
- **Threshold optimization**: Via precision-recall curve, model-specific optimal thresholds
- **Multi-seed averaging**: Final submission averaged predictions from 10 random seeds

## Dataset

- **Source**: Taiwan Economic Journal (via Kaggle)
- **Training**: 5,455 samples (96.88% non-bankrupt, 3.12% bankrupt)
- **Test**: 1,364 samples
- **Features**: 95 financial ratios

## Project Structure

```
├── bankruptcy_ZhankaiZhang_v1.ipynb   # Main notebook (final version)
├── bankruptcy_ZhankaiZhang.ipynb      # Earlier iteration
├── bankruptcy_prediction.ipynb         # Initial exploration
├── company-bankruptcy-prediction.ipynb # Supplementary analysis
├── report.md                           # Full project report
├── Report_ZhankaiZhang.pdf            # Report (PDF)
├── Notebook_ZhankaiZhang.pdf          # Notebook export (PDF)
├── model_comparison.png               # Model performance visualization
├── submission.csv                      # Final Kaggle submission
├── train.csv                           # Training data
├── test.csv                            # Test data
└── Additional reading_imbalanced data.pdf  # Reference material
```
