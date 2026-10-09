# Company Bankruptcy Prediction

**Zhankai Zhang** · BU MF815 Machine Learning · Kaggle competition project · March 2026

![Python](https://img.shields.io/badge/Python-XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-3776AB)
![Optuna](https://img.shields.io/badge/Optuna-200--trial%20TPE-854CE5)
![Imbalance](https://img.shields.io/badge/positive%20class-3.12%25-red)
![Models](https://img.shields.io/badge/configurations%20tested-29-blue)

Binary classification of company bankruptcy from **95 financial ratios** (Taiwan Economic Journal),
under severe class imbalance — **170 bankrupt firms out of 5,455**. Twenty-nine configurations across
five model families were evaluated on a single protocol; the deliverable is **XGBoost with an ECOD
outlier feature**, at **Val F1 = 0.5060** on the bankrupt class (AUC-ROC 0.9627).

The headline finding is not the model but the measurement: **threshold optimisation moves F1 by
10–20 points**, while four fashionable techniques — SMOTE and its variants, focal loss, stacking, and
model averaging — were each tested and **rejected for failing to beat the simpler alternative**.

---

## 1. Introduction and Data Overview

The training set contains 5,455 samples with severe class imbalance: **96.88% non-bankrupt** (class 0)
vs **3.12% bankrupt** (class 1, 170 samples). The test set contains 1,364 unlabelled samples.

![Class imbalance](figures/class_imbalance.png)

An accuracy of 96.88% is available for free by predicting "never bankrupt", so **accuracy is not a
usable metric here** — every result below is reported on **F1 of the bankrupt class**.

### Data preprocessing pipeline

- **Sentinel values.** 21 features contain placeholder values (`>= 1e9`) for undefined ratios. Financial
  ratios should not exceed ~100×, so these were replaced with `NaN`. Two columns with >50% missing were
  dropped (*Total Asset Growth Rate*, *Cash Turnover Rate*).

  ![Sentinel values by column](figures/sentinel_values.png)

- **Imputation.** Median imputation **fitted on training data only**. Binary missingness indicators were
  added for features with >5% missing rate where the missingness was itself informative (bankruptcy rate
  ratio > 1.2×).

  ![Missing rates and the 50% drop threshold](figures/missing_rates.png)

- **Scaling.** `StandardScaler` applied only for Logistic Regression; tree-based models used unscaled features.
- **Evaluation setup.** 80/20 stratified holdout + 5-fold Stratified K-Fold CV. All reported Val F1 scores
  use probability threshold optimisation via `precision_recall_curve`.

### Exploratory checks

**Leakage check on the row index.** Before modelling, the `Unnamed: 0` index column was tested for class
information — if row order correlates with the label, any model will "learn" it and the score is fiction.
The two class distributions overlap, so the index carries no signal:

![ID distribution by class](figures/id_leakage_check.png)

**Feature–target relationships.** Top-20 features by |Pearson correlation|, and the correlation structure
among them — the latter motivating the collinearity pruning in §5:

![Top 20 features by correlation with bankruptcy](figures/corr_target_top20.png)
![Correlation heatmap among top 20](figures/corr_heatmap_top20.png)

**Distributions by class** for the 12 most correlated features (1st/99th-percentile clipped, medians dashed):

![Feature distributions, bankrupt vs non-bankrupt](figures/feature_distributions.png)

**Predictive Power Score vs Pearson.** PPS captures non-linear predictors that linear correlation misses;
comparing the two sets identifies features worth keeping for tree models even when their Pearson
correlation is near zero:

![PPS vs Pearson](figures/pps_vs_pearson.png)

---

## 2. Methods

Three classification methods were chosen for their complementary strengths on imbalanced tabular data.

### 2.1 Random Forest

Random Forest handles imbalance via `class_weight='balanced'`, which inversely weights samples by class
frequency. As a bagging ensemble of decorrelated trees, RF provides a non-boosting baseline and naturally
estimates feature importance. Tested on both full features (90) and a curated 13-feature subset selected
by Pearson correlation and Predictive Power Score (PPS) analysis.

### 2.2 XGBoost

XGBoost uses gradient boosting with `scale_pos_weight` to upweight the minority class directly in the loss
function. Its built-in L1/L2 regularisation (`reg_alpha`, `reg_lambda`) and tree-complexity penalties
(`gamma`, `min_child_weight`) help prevent overfitting on the small positive class (170 samples). XGBoost
was the primary model due to its strong performance on structured/tabular data.

### 2.3 Logistic Regression

Logistic Regression with `class_weight='balanced'` provides an interpretable linear baseline. It was trained
on `StandardScaler`-transformed features. Despite being the simplest model, its probabilistic outputs are
well calibrated, making threshold optimisation particularly effective.

**Why these three:** they span three distinct model families — bagging (RF), boosting (XGBoost), and linear
(LR) — providing diversity in modelling assumptions and enabling meaningful comparison. LightGBM and
CatBoost were added later as additional boosting implementations.

---

## 3. Hyperparameter Tuning

### 3.1 XGBoost — Optuna Bayesian optimisation (200 trials)

Optuna with the TPE (Tree-structured Parzen Estimator) sampler was used for Bayesian hyperparameter
optimisation. The scorer was `f1_score` evaluated within each CV fold.

| Parameter | Search range | Best value |
|---|---|---|
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

**Key observations.** The high `gamma = 2.80` and `min_child_weight = 17` indicate strong regularisation is
needed to prevent overfitting on only 136 positive training samples (80% split). The high
`scale_pos_weight = 57.4` — above the natural ratio of ~31 — suggests the model benefits from aggressively
upweighting the minority class during training.

### 3.2 Random Forest — Optuna (50 trials)

| Parameter | Best value |
|---|---|
| n_estimators | 374 |
| max_depth | 23 |
| min_samples_split | 16 |
| min_samples_leaf | 8 |
| max_features | 0.74 |
| class_weight | balanced |

### 3.3 Logistic Regression

Logistic Regression used `class_weight='balanced'` with default regularisation (C = 1.0, L2 penalty). No
extensive tuning was performed, as the model has few hyperparameters and the primary tuning lever is the
probability threshold.

---

## 4. Results

### 4.1 Baseline model comparison

All models were evaluated using the same pipeline: 5-fold stratified CV for F1 and AUC-ROC, then
threshold-optimised F1 on the validation set.

| Model | Accuracy | F1 (Class 0) | F1 (Class 1) | AUC-ROC | Threshold |
|---|---|---|---|---|---|
| XGBoost | 0.95 | 0.97 | 0.4255 | 0.9625 | 0.0675 |
| Logistic Regression | 0.96 | 0.98 | **0.4337** | 0.9155 | 0.9171 |
| LightGBM | 0.96 | 0.98 | 0.4324 | 0.9578 | 0.2329 |
| Random Forest | 0.96 | 0.98 | 0.4158 | 0.9210 | 0.1267 |

![Baseline comparison](figures/model_comparison_baseline.png)

**Observations.**

- All models achieve >95% accuracy — but this is misleading due to the 97/3 class split. A model predicting
  all non-bankrupt achieves 96.88% accuracy.
- F1 of class 0 (non-bankrupt) is uniformly high (~0.97–0.98) across all models.
- **F1 of class 1 (bankrupt/default) ranges from 0.42 to 0.43** — the discriminating metric.
- AUC-ROC is high (>0.91) for all models, indicating strong ranking ability. The bottleneck is converting
  good rankings into good class predictions.
- **Optimal thresholds vary wildly (0.07 to 0.92)**, confirming that threshold optimisation is essential and
  model-specific.

### 4.2 After hyperparameter tuning

| Model | CV F1 | Val F1 (Class 1) | AUC-ROC |
|---|---|---|---|
| XGBoost (tuned) | 0.5435 | **0.4706** | 0.9618 |
| LightGBM (tuned) | **0.5882** | 0.4483 | 0.9605 |
| RF (tuned) | 0.5329 | 0.3944 | 0.9489 |

Tuning improved XGBoost's Val F1 from 0.4255 to **0.4706** (+10.6%). LightGBM achieved the highest CV F1
(0.5882) but lower Val F1 — **suggesting some overfitting in cross-validation**.

Threshold–F1 curves for the tuned model, and for the tuned model with the ECOD feature added:

![Threshold curve — XGBoost tuned](figures/threshold_curve_xgb_tuned.png)
![Threshold curve — XGBoost tuned + ECOD](figures/threshold_curve_xgb_ecod.png)

### 4.3 XGBoost extensions

ECOD (Empirical CDF Outlier Detection) was applied as a feature engineering technique: an outlier score was
computed for each sample and added as a binary feature (`Is_outlier`).

| Extension | CV F1 | Val F1 (Class 1) |
|---|---|---|
| XGBoost (tuned) | 0.5435 | 0.4706 |
| **+ ECOD** | 0.5263 | **0.5060** |
| + Yeo-Johnson | 0.5113 | 0.4872 |
| + RFECV | 0.5165 | 0.4578 |
| + Focal Loss | 0.2866 | 0.4225 |

ECOD improved Val F1 to **0.5060** — the best result achieved.

### 4.4 All 29 configurations — best per model family

Twenty-nine configurations were evaluated in total (Baseline / SMOTE variants / Optuna-tuned, across five
families). The best of each family:

| Model family | Best configuration | Accuracy | F1 (Class 0) | F1 (Class 1) | AUC-ROC |
|---|---|---|---|---|---|
| **XGBoost** | XGB (tuned + ECOD) | 0.9624 | 0.9805 | **0.5060** | 0.9627 |
| LightGBM | LGBM (tuned) | **0.9707** | **0.9849** | 0.4483 | 0.9605 |
| Logistic Reg. | LR (all features) | 0.9569 | 0.9776 | 0.4337 | 0.9155 |
| CatBoost | CatBoost (tuned) | 0.9633 | 0.9811 | 0.4286 | **0.9638** |
| Random Forest | RF (all features) | 0.9615 | 0.9801 | 0.4167 | 0.9487 |

![All configurations — F1, AUC-ROC, Accuracy](figures/model_comparison_final.png)

**The choice of metric decides the winner.** CatBoost has the best AUC-ROC (0.9638) and LightGBM the best
accuracy (0.9707) and class-0 F1 (0.9849) — yet XGBoost + ECOD is the deliverable because it wins on the
metric that matters when 3.12% of the data is positive. Reporting any of the other three numbers as the
headline would be defensible-looking and wrong.

---

## 5. Best Model Selection

**Selected model: XGBoost (tuned + ECOD), Val F1 = 0.5060.** The selection was based on the following
reasoning:

1. **XGBoost outperformed RF and LR** in both CV and validation F1 after tuning, with a top-tier AUC-ROC (0.96).
2. **ECOD provided the largest single improvement** (+0.035 Val F1) by capturing outlier patterns that
   complement the original financial ratios. Bankrupt companies are **5.7× more likely** to be ECOD outliers.
3. **SMOTE and other sampling methods were tested and rejected.** All three variants (SMOTE, SMOTE-ENN,
   SMOTE+Tomek) degraded Val F1 from 0.5060 to 0.41–0.43. Since `scale_pos_weight` already handles imbalance
   in the loss function, adding oversampling creates **double compensation**.
4. **Ensemble methods did not outperform the single model.** Stacking with an LR meta-learner reached Val F1
   = 0.4828, and probability averaging the same. The best pair (XGB + CatBoost) reached 0.4828. The three
   gradient boosting models produce highly correlated predictions, limiting diversity gain.

### Final submission pipeline

- Removed **11 collinear features** (|r| > 0.95): 89 → 78 features
- Added the ECOD `Is_outlier` feature (`contamination = 0.05`) → **79 features**; 273/5,455 train and
  89/1,364 test samples flagged
- Retrained on the full training set (5,455 samples)
- **Averaged predictions from 10 random seeds** to reduce variance
- Applied the OOF-derived threshold (**0.6004**); OOF F1 on the full training set: **0.4988**
- Test predictions: **55 / 1,364 flagged bankrupt (4.03%)** — against a 3.12% training base rate

---

## 6. Additional Contributions (beyond course methods)

| Technique | Purpose | Outcome |
|---|---|---|
| ECOD outlier detection | Feature engineering | Improved Val F1 by **+0.035** |
| Optuna Bayesian tuning | Hyperparameter optimisation | More efficient than grid/random search |
| Focal Loss | Custom loss for imbalance | **Did not improve** over `scale_pos_weight` |
| CatBoost (ordered boosting) | Model diversity | Val F1 = 0.4286 — below XGBoost |
| OOF stacking with rank normalisation | Ensemble learning | Val F1 = 0.4828 — below the single model |
| Multi-seed averaging | Variance reduction | Used in the final submission |
| Threshold optimisation via PR curve | Post-processing | **Critical for all models** |

---

## 7. Conclusion

The best performing model is **XGBoost with ECOD outlier features**, achieving an estimated F1 of ~0.50 on
the bankrupt class. Key findings:

1. **Threshold optimisation is the single most impactful technique** for imbalanced classification —
   improving F1 by 10–20 percentage points over the default 0.5 cutoff.
2. **Built-in class weighting outperforms external sampling.** `scale_pos_weight` in XGBoost was more
   effective than SMOTE, SMOTE-ENN, and SMOTE+Tomek Links.
3. **Simple models can match complex ensembles** when the positive class is extremely small (170 samples).
   Stacking and model averaging both failed to outperform the best single model.
4. With only 170 bankrupt companies in training (3.12%), **the fundamental limitation is data scarcity, not
   model complexity.**

### Limitations

- **A single 80/20 holdout decides the ranking.** With ~34 positive samples in the validation fold, Val F1
  differences below ~0.03 are not separable from sampling noise.
- **Twenty-nine configurations were compared on one validation fold**, so the best-of-29 figure (0.5060) is
  biased upward by selection and no multiple-comparison correction was applied. The OOF estimate (0.4988) is
  the more conservative number.
- **The thresholds are fitted.** Each model's threshold is optimised on the same data used to report its F1
  in §4.1–4.3; only the final submission uses an out-of-fold threshold.
- **Taiwan Economic Journal data, one country and period.** Nothing here should be assumed to transfer to
  another market or vintage without revalidation.

---

## Reproducing this

The final notebook reads `train.csv` / `test.csv` from the repository root and writes `submission.csv`:

```bash
jupyter lab bankruptcy_ZhankaiZhang_v1.ipynb   # run top to bottom
```

Requires `xgboost`, `lightgbm`, `catboost`, `scikit-learn`, `imbalanced-learn`, `optuna`, `pyod` (ECOD),
`ppscore`, `pandas`, `numpy`, `matplotlib`, `seaborn`.

## Repository layout

```
.
├── README.md                             this file
├── report.md                             the written project report (canonical write-up)
├── bankruptcy_ZhankaiZhang_v1.ipynb      FINAL notebook — all reported numbers come from here
├── train.csv · test.csv                  Taiwan Economic Journal data (5,455 / 1,364)
├── submission.csv                        final Kaggle submission (threshold 0.6004)
├── figures/                              figures referenced above, exported from the final notebook
├── docs/                                 report PDF, notebook PDF, assignment brief, reference reading
└── archive/                              earlier notebook iterations and duplicate report exports
```

## References

- Kuhn, M. & Johnson, K. (2013). *Applied Predictive Modeling*, Ch. 16: Remedies for Severe Class Imbalance.
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
- Li, Z. et al. (2022). ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution
  Functions. *TKDE*.
- Chawla, N. V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321–357.

**AI disclosure.** Claude Code (Anthropic) was used for code assistance, debugging, and exploratory
analysis. All model selection decisions, result interpretation, and report writing were conducted by the
author.
