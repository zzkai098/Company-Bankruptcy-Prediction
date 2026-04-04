---
title: ""
---

\newpage

\begin{center}
\vspace*{3cm}
{\Huge\textbf{Company Bankruptcy Prediction}}\\[0.5cm]
{\Large A Kaggle Competition Project}\\[2cm]
{\large\textbf{Course:} MF815 Machine Learning}\\[0.3cm]
{\large\textbf{Author:} Zhankai Zhang}\\[0.3cm]
{\large\textbf{Date:} March 2026}\\[3cm]
\end{center}

\newpage

# Table of Contents

1. Introduction and Data Overview
2. Methods
3. Hyperparameter Tuning
4. Results
5. Best Model Selection
6. Additional Contributions
7. Conclusion

\newpage

## 1. Introduction and Data Overview

This project tackles binary classification of company bankruptcy using 95 financial ratio features from the Taiwan Economic Journal. The training set contains 5,455 samples with severe class imbalance: 96.88% non-bankrupt (class 0) vs. 3.12% bankrupt (class 1, 170 samples). The test set contains 1,364 samples without labels.

**Data preprocessing pipeline:**

- **Sentinel values:** 21 features contain placeholder values (>= 1e9) for undefined ratios. These were replaced with NaN. Two columns with >50% missing were dropped (*Total Asset Growth Rate*, *Cash Turnover Rate*).
- **Imputation:** Median imputation fitted on training data only. Binary missingness indicators were added for features with >5% missing rate where the missingness was informative (bankruptcy rate ratio > 1.2x).
- **Scaling:** StandardScaler applied only for Logistic Regression; tree-based models used unscaled features.
- **Evaluation setup:** 80/20 stratified holdout split + 5-fold Stratified K-Fold CV. All reported Val F1 scores use probability threshold optimization via `precision_recall_curve`.

## 2. Methods

Three classification methods were chosen for their complementary strengths on imbalanced tabular data:

### 2.1 Random Forest

Random Forest handles imbalance via `class_weight='balanced'`, which inversely weights samples by class frequency. As a bagging ensemble of decorrelated trees, RF provides a non-boosting baseline and naturally estimates feature importance. Tested on both full features (90) and a curated 13-feature subset selected by Pearson correlation and Predictive Power Score (PPS) analysis.

### 2.2 XGBoost

XGBoost uses gradient boosting with `scale_pos_weight` to upweight the minority class directly in the loss function. Its built-in L1/L2 regularization (`reg_alpha`, `reg_lambda`) and tree-complexity penalties (`gamma`, `min_child_weight`) help prevent overfitting on the small positive class (170 samples). XGBoost was the primary model due to its strong performance on structured/tabular data.

### 2.3 Logistic Regression

Logistic Regression with `class_weight='balanced'` provides an interpretable linear baseline. It was trained on StandardScaler-transformed features. Despite being the simplest model, its probabilistic outputs are well-calibrated, making threshold optimization particularly effective.

**Why these three:** They span three distinct model families — bagging (RF), boosting (XGBoost), and linear (LR) — providing diversity in modeling assumptions and enabling meaningful comparison.

## 3. Hyperparameter Tuning

### 3.1 XGBoost — Optuna Bayesian Optimization (200 trials)

Optuna with TPE (Tree-structured Parzen Estimator) sampler was used for Bayesian hyperparameter optimization. The scorer was `f1_score` evaluated within each CV fold.

| Parameter | Search Range | Best Value |
|-----------|-------------|------------|
| n_estimators | 100–800 | 578 |
| max_depth | 3–7 | 6 |
| learning_rate | 0.01–0.3 (log) | 0.030 |
| scale_pos_weight | 10–60 | 57.4 |
| gamma | 0.0–5.0 | 2.80 |
| min_child_weight | 1–20 | 17 |
| subsample | 0.5–1.0 | 0.636 |
| colsample_bytree | 0.5–1.0 | 0.624 |
| reg_alpha | 1e-4–10 (log) | 0.413 |
| reg_lambda | 1e-4–10 (log) | 0.918 |

**Key observations:** The high `gamma=2.80` and `min_child_weight=17` indicate strong regularization is needed to prevent overfitting on only 136 positive training samples (80% split). The high `scale_pos_weight=57.4` (above the natural ratio of ~31) suggests the model benefits from aggressively upweighting the minority class during training.

### 3.2 Random Forest — Optuna (50 trials)

| Parameter | Best Value |
|-----------|------------|
| n_estimators | 374 |
| max_depth | 23 |
| min_samples_split | 16 |
| min_samples_leaf | 8 |
| max_features | 0.74 |
| class_weight | balanced |

### 3.3 Logistic Regression

Logistic Regression used `class_weight='balanced'` with default regularization (C=1.0, L2 penalty). No extensive tuning was performed as the model has few hyperparameters and the primary tuning lever is the probability threshold.

## 4. Results

### 4.1 Baseline Model Comparison

All models were evaluated using the same pipeline: 5-fold stratified CV for F1 and AUC-ROC, then threshold-optimized F1 on the validation set.

| Model | Accuracy | F1 (Class 0) | F1 (Class 1) | AUC-ROC | Threshold |
|-------|----------|--------------|--------------|---------|-----------|
| XGBoost | 0.95 | 0.97 | 0.4255 | 0.9625 | 0.0675 |
| Logistic Regression | 0.96 | 0.98 | 0.4337 | 0.9155 | 0.9171 |
| LightGBM | 0.96 | 0.98 | 0.4324 | 0.9578 | 0.2329 |
| Random Forest | 0.96 | 0.98 | 0.4158 | 0.9210 | 0.1267 |

**Observations:**

- All models achieve >95% accuracy — but this is misleading due to the 97/3 class split. A model predicting all non-bankrupt achieves 96.88% accuracy.
- F1 of class 0 (non-bankrupt) is uniformly high (~0.97-0.98) across all models.
- **F1 of class 1 (bankrupt/default) ranges from 0.42 to 0.43** — the discriminating metric.
- AUC-ROC is high (>0.91) for all models, indicating strong ranking ability. The bottleneck is converting good rankings into good class predictions.
- Optimal thresholds vary wildly (0.07 to 0.92), confirming that threshold optimization is essential and model-specific.

### 4.2 After Hyperparameter Tuning

| Model | CV F1 | Val F1 (Class 1) | AUC-ROC |
|-------|-------|-------------------|---------|
| XGBoost (tuned) | 0.5435 | **0.4706** | 0.9618 |
| LightGBM (tuned) | 0.5882 | 0.4483 | 0.9605 |
| RF (tuned) | 0.5329 | 0.3944 | 0.9489 |

Tuning improved XGBoost's Val F1 from 0.4255 to **0.4706** (+10.6%). LightGBM achieved the highest CV F1 (0.5882) but lower Val F1, suggesting some overfitting in cross-validation.

### 4.3 XGBoost Extensions

ECOD (Empirical CDF Outlier Detection) was applied as a feature engineering technique: an outlier score was computed for each sample and added as a binary feature (`Is_outlier`).

| Extension | CV F1 | Val F1 (Class 1) |
|-----------|-------|-------------------|
| XGBoost (tuned) | 0.5435 | 0.4706 |
| **+ ECOD** | **0.5263** | **0.5060** |
| + Yeo-Johnson | 0.5113 | 0.4872 |
| + RFECV | 0.5165 | 0.4578 |
| + Focal Loss | 0.2866 | 0.4225 |

ECOD improved Val F1 to **0.5060** — the best result achieved.

## 5. Best Model Selection

**Selected model: XGBoost (tuned + ECOD), Val F1 = 0.5060**

The selection was based on the following reasoning:

1. **XGBoost outperformed RF and LR** in both CV and validation F1 after tuning, with the highest AUC-ROC (0.96).

2. **ECOD provided the largest single improvement** (+0.035 Val F1) by capturing outlier patterns that complement the original financial ratios. Bankrupt companies are 5.7x more likely to be ECOD outliers.

3. **SMOTE and other sampling methods were tested and rejected.** All three variants (SMOTE, SMOTE-ENN, SMOTE+Tomek) degraded Val F1 from 0.5060 to 0.41-0.43. Since `scale_pos_weight` already handles imbalance in the loss function, adding oversampling creates double compensation.

4. **Ensemble methods (stacking, probability averaging) did not outperform the single model.** Stacking with LR meta-learner achieved Val F1 = 0.4828, and simple probability averaging reached 0.4828. The three gradient boosting models (XGBoost, LightGBM, CatBoost) produce highly correlated predictions, limiting diversity gain.

**Final submission pipeline:**

- Removed 11 collinear features (|r| > 0.95) → 79 features
- Retrained on full training data (5,455 samples) with ECOD outlier feature
- Averaged predictions from 10 random seeds to reduce variance
- Applied OOF-derived threshold (0.6004)
- OOF F1 on full training set: **0.4988**
- Test predictions: 55 / 1,364 predicted as bankrupt (4.03%)

## 6. Additional Contributions (Beyond Course Methods)

| Technique | Purpose | Outcome |
|-----------|---------|---------|
| ECOD outlier detection | Feature engineering | Improved Val F1 by +0.035 |
| Optuna Bayesian tuning | Hyperparameter optimization | More efficient than grid/random search |
| Focal Loss | Custom loss for imbalance | Did not improve over scale_pos_weight |
| CatBoost (ordered boosting) | Model diversity | Val F1 = 0.4286 (below XGBoost) |
| OOF Stacking with rank normalization | Ensemble learning | Val F1 = 0.4828 (below single model) |
| Multi-seed averaging | Variance reduction | Used in final submission |
| Threshold optimization via precision-recall curve | Post-processing | Critical for all models |

## 7. Conclusion

The best performing model is **XGBoost with ECOD outlier features**, achieving an estimated F1-score of ~0.50 on the default (bankrupt) class. Key findings:

1. **Threshold optimization is the single most impactful technique** for imbalanced classification — improving F1 by 10-20 percentage points over the default 0.5 cutoff.

2. **Built-in class weighting outperforms external sampling.** `scale_pos_weight` in XGBoost was more effective than SMOTE, SMOTE-ENN, and SMOTE+Tomek Links.

3. **Simple models can match complex ensembles** when the positive class is extremely small (170 samples). Stacking and model averaging failed to outperform the best single model.

4. With only 170 bankrupt companies in training (3.12%), the fundamental limitation is data scarcity, not model complexity.

## References

- Kuhn, M. & Johnson, K. (2013). *Applied Predictive Modeling*, Ch. 16: Remedies for Severe Class Imbalance.
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
- Li, Z. et al. (2022). ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions. *TKDE*.
- Chawla, N. V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321-357.

**AI Disclosure:** Claude Code (Anthropic) was used for code assistance, debugging, and exploratory analysis. All model selection decisions, result interpretation, and report writing were conducted by the author.
