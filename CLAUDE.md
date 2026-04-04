# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Binary classification task: predict company bankruptcy from 95 financial/accounting ratio features. Part of MF815 Machine Learning coursework (also structured as a Kaggle competition).

- **Training data:** `train.csv` — 5,455 samples, 96 columns (95 features + target `Bankrupt?`)
- **Test data:** `test.csv` — 1,364 samples, same structure (no labels)
- **Submission format:** `sample_submission.csv` — two columns: row ID + predicted label
- **Feature metadata:** `metaData.csv` — human-readable names for each feature column
- **Class imbalance:** ~97% non-bankrupt (0), ~3% bankrupt (1) — handle with SMOTE, class weights, or stratified sampling
- **Reference:** `Company Bankruptcy Prediction.pdf` — problem description and requirements

## Competition Background

- **Competition:** 3rd International Competition in Data Science & AI, organized by the International Society of Data Scientists (ISODS).
- **Eligibility:** University students + graduates within 5 years; teams may optionally include a faculty/industry advisor.
- **Citation:** Adoy101, Cory Wang, ISODS Competitions, and Nguyen Gia Nhu. *Company Bankruptcy Prediction*. Kaggle, 2022.
- **Committee chairs:** Dr. Sebastian Tomczak (Wroclaw University of Science and Technology), Dr. Gia Nhu Nguyen (Duy Tan University), Dr. Christopher Do (ISODS President), Dr. Nam Nguyen & Mr. Cory Wang (ISODS Vice Presidents).

## Dataset Background

- **Source:** Emerging Markets Information Service (EMIS) — a database covering emerging market companies.
- **Geography:** Polish companies.
- **Time period:** Bankrupt companies observed 2000–2012; still-operating companies observed 2007–2013.
- **Objective:** Assess the financial condition of a company and its long-term operation prospects.
- **Evaluation metric: F1-score** (higher is better; used as the Kaggle ranking metric). Do NOT optimize for accuracy.

## Evaluation & Submission

- **Metric:** F1-score = `2 * (precision * recall) / (precision + recall)`
  - Precision = true positives / all predicted positives
  - Recall = true positives / all actual positives
  - F1 weights precision and recall **equally** — avoid sacrificing one for the other.
  - Moderately good performance on both is favored over perfect recall with poor precision (or vice versa).
- **Submission file format** (CSV, two columns):
  ```
  id,class
  1,1
  2,0
  3,1
  ```
  - `id`: row index from test set
  - `class`: predicted label (1 = bankrupt, 0 = not bankrupt)
  - File must include a header row.

## Feature Glossary (63 named KPIs)

| # | Feature Description |
|---|---|
| 1 | net profit / total assets |
| 2 | total liabilities / total assets |
| 3 | working capital / total assets |
| 4 | current assets / short-term liabilities |
| 5 | [(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365 |
| 6 | retained earnings / total assets |
| 7 | EBIT / total assets |
| 8 | book value of equity / total liabilities |
| 9 | sales / total assets |
| 10 | equity / total assets |
| 11 | (gross profit + extraordinary items + financial expenses) / total assets |
| 12 | gross profit / short-term liabilities |
| 13 | (gross profit + depreciation) / sales |
| 14 | (gross profit + interest) / total assets |
| 15 | (total liabilities * 365) / (gross profit + depreciation) |
| 16 | (gross profit + depreciation) / total liabilities |
| 17 | total assets / total liabilities |
| 18 | gross profit / total assets |
| 19 | gross profit / sales |
| 20 | (inventory * 365) / sales |
| 21 | sales (n) / sales (n-1) |
| 22 | profit on operating activities / total assets |
| 23 | net profit / sales |
| 24 | gross profit (in 3 years) / total assets |
| 25 | (equity - share capital) / total assets |
| 26 | (net profit + depreciation) / total liabilities |
| 27 | profit on operating activities / financial expenses |
| 28 | working capital / fixed assets |
| 29 | logarithm of total assets |
| 30 | (total liabilities - cash) / sales |
| 31 | (gross profit + interest) / sales |
| 32 | (current liabilities * 365) / cost of products sold |
| 33 | operating expenses / short-term liabilities |
| 34 | operating expenses / total liabilities |
| 35 | profit on sales / total assets |
| 36 | total sales / total assets |
| 37 | (current assets - inventories) / long-term liabilities |
| 38 | constant capital / total assets |
| 39 | profit on sales / sales |
| 40 | (current assets - inventory - receivables) / short-term liabilities |
| 41 | total liabilities / ((profit on operating activities + depreciation) * (12/365)) |
| 42 | profit on operating activities / sales |
| 43 | rotation receivables + inventory turnover in days |
| 44 | (receivables * 365) / sales |
| 45 | net profit / inventory |
| 46 | (current assets - inventory) / short-term liabilities |
| 47 | (inventory * 365) / cost of products sold |
| 48 | EBITDA (profit on operating activities - depreciation) / total assets |
| 49 | EBITDA (profit on operating activities - depreciation) / sales |
| 50 | current assets / total liabilities |
| 51 | short-term liabilities / total assets |
| 52 | (short-term liabilities * 365) / cost of products sold |
| 53 | equity / fixed assets |
| 54 | constant capital / fixed assets |
| 55 | working capital |
| 56 | (sales - cost of products sold) / sales |
| 57 | (current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation) |
| 58 | total costs / total sales |
| 59 | long-term liabilities / equity |
| 60 | sales / inventory |
| 61 | sales / receivables |
| 62 | (short-term liabilities * 365) / sales |
| 63 | sales / short-term liabilities |
| 64 | sales / fixed assets |

> Features 65–95 are not named in the public metadata. Columns 85 and 94 are binary flag variables (0/1): `Liability-Assets Flag` and `Net Income Flag`.

## Key Data Characteristics

- All features are continuous numerical financial ratios (profitability, liquidity, solvency, cash flow, turnover, growth)
- Two binary flag columns (col 85, col 94) should be treated separately from continuous features
- Missing values are present — impute with column median (computed on train only); drop columns with >50% missing
- Evaluation must use **F1-score** (the competition metric), plus AUC-ROC and Precision-Recall for diagnostics

## Files

| File | Description |
|------|-------------|
| `train.csv` | Training set with `Bankrupt?` label |
| `test.csv` | Test set (no label) — generate predictions for submission |
| `sample_submission.csv` | Correct submission format |
| `metaData.csv` | Feature name descriptions |
| `bankruptcy_prediction.ipynb` | Main analysis notebook |

## Environment

Jupyter notebooks with standard Python ML stack: `pandas`, `numpy`, `scikit-learn`, `matplotlib`/`seaborn`. No build or test runner — work is done interactively in notebooks.

To start a notebook server:
```bash
jupyter notebook
```

---

## Handling Imbalanced Data (Notes from Kuhn & Johnson, Ch. 16)

Source: *Applied Predictive Modeling*, Chapter 16 — "Remedies for Severe Class Imbalance"

### Context and Why It Matters

- Class imbalance occurs when one or more classes have very low proportions in the training data relative to others.
- This project has ~97% non-bankrupt (0) vs ~3% bankrupt (1) — a roughly 32:1 majority-to-minority ratio, which is severe.
- Without correction, models will predict almost every sample as non-bankrupt, achieving high accuracy (~97%) but near-zero sensitivity on the class of interest.
- Key symptom: predicted probability distributions become highly left-skewed — most samples get very low predicted probability of the minority class regardless of their true label.
- Accuracy and Kappa can be misleading under severe imbalance. Do NOT use accuracy as the primary metric; use F1, AUC-ROC, or Precision-Recall.

### Core Diagnostic: What Imbalance Does to Models

From Table 16.1 (caravan insurance, 6% event rate, similar structure to bankruptcy):

| Model | Accuracy | Sensitivity | Specificity | AUC-ROC |
|---|---|---|---|---|
| Random forest | 93.5% | 6.8% | 99.0% | 0.757 |
| FDA (MARS) | 93.8% | 1.7% | 99.7% | 0.754 |
| Logistic regression | 93.9% | 1.7% | 99.8% | 0.727 |

- Accuracy looks great but sensitivity (recall of the minority class) is nearly 0.
- The imbalance overwhelms patterns useful for predicting the minority class.
- AUC-ROC is a better comparison metric because it is independent of the probability cutoff.

---

### Technique 1: Model Tuning for Sensitivity

**Mechanics:** Tune the model to maximize sensitivity (recall of the minority class) rather than accuracy or AUC-ROC during cross-validation.

**Pros:**
- Simple; no changes to training data or model architecture.
- Can yield modest improvements in sensitivity without hurting specificity.

**Cons:**
- Often insufficient on its own for severe imbalance.
- Some models (e.g., random forest) show no meaningful trend in sensitivity across tuning parameters when the imbalance is severe.
- FDA (MARS) improved sensitivity from ~0% to only 5.4% when increasing model complexity — still far too low to be useful.

**Recommendation for bankruptcy task:** Use as a supplement, not a primary strategy. Tune with F1 or sensitivity as the cross-validation metric.

---

### Technique 2: Alternate Probability Cutoffs

**Mechanics:** Instead of using the default 0.5 probability threshold to predict the positive class, find a lower threshold that improves sensitivity at an acceptable cost to specificity.

Two methods for selecting the new cutoff:
1. **Closest-to-perfect-model:** Find the point on the ROC curve geometrically closest to (0% FPR, 100% TPR) — the upper-left corner.
2. **Youden's J Index:** `J = Sensitivity + Specificity - 1`. Find the cutoff maximizing J. Often yields a similar cutoff to method 1.

**Example result** (random forest, caravan data):
- Default cutoff 0.50 → Sensitivity 9.5%, Specificity 99%
- Cutoff 0.064 → Sensitivity 61.2%, Specificity 76.1%

**Critical rule:** Derive the new cutoff from a held-out **evaluation set**, not from the training set or test set.
- Training set probabilities have optimistic bias → inaccurate cutoff.
- Using the test set for cutoff selection contaminates your final performance estimate (Ewald 2006 shows this exaggerates test set performance).

**What alternate cutoffs can and cannot do:**
- They can only trade off sensitivity vs. specificity (move samples between rows of the confusion matrix).
- They do NOT increase the underlying separation between classes — the model parameters are unchanged.
- They do NOT improve AUC-ROC, which stays the same.

**Recommendation for bankruptcy task:** After training any model, use a separate validation fold or a dedicated eval split to determine the cutoff that maximizes F1 (not Youden's J, since the competition metric is F1, not balanced accuracy). This is the most important post-processing step.

---

### Technique 3: Adjusting Prior Probabilities (for Bayesian models)

**Mechanics:** For models that use prior probabilities (naive Bayes, discriminant analysis), the default priors are derived from training data class frequencies, which will strongly bias predictions toward the majority class. Replace them with more balanced priors.

**Example:** FDA model on caravan data with balanced priors (60% event, 40% non-event) achieved Sensitivity 71.2%, Specificity 66.9% on test set vs. near-zero sensitivity with natural priors.

**Note:** Like cutoff adjustment, this does not change the underlying ranking of samples — AUC-ROC is identical. It only changes the decision boundary.

**Decision boundary formula for two classes:**

```
Predict class 1 if:   p1/p2  >  (C(1|2) * pi_2) / (C(2|1) * pi_1)
```

Where:
- `p1`, `p2` = estimated class probabilities
- `C(j|i)` = cost of predicting class j when true class is i
- `pi_1`, `pi_2` = prior probabilities of each class

For equal costs and equal priors, this reduces to predicting class 1 when `p1 > 0.5`.

**Recommendation for bankruptcy task:** Applicable only if using LDA/QDA or naive Bayes. For tree-based or ensemble models, use class weights or sampling instead.

---

### Technique 4: Unequal Case Weights

**Mechanics:** Assign higher weights to minority class samples during training. Many models support `class_weight` or `sample_weight` parameters. This is mathematically equivalent to duplicating minority class samples.

**Relationship to sampling:** Case weighting and up-sampling are functionally very similar. Weighting is usually cleaner to implement since it avoids actually duplicating rows.

**Recommendation for bankruptcy task:** Use `class_weight='balanced'` or `class_weight={0:1, 1:32}` (inverse of class frequency) in sklearn classifiers (LogisticRegression, SVC, RandomForestClassifier, GradientBoostingClassifier all support this). Always tune the weight ratio via cross-validation.

---

### Technique 5: Sampling Methods

#### 5a. Up-sampling (Over-sampling)

**Mechanics:** Sample minority class cases with replacement until the minority class is roughly equal in size to the majority class. Naively duplicates existing minority samples (or creates slight variations).

**Pros:** Simple; increases minority class representation.

**Cons:**
- Because resampling folds will likely contain the same duplicated sample in both train and validation, resampling performance estimates become optimistically biased.
- Does not create new information — just repeats existing minority points.

**Sklearn:** `from imblearn.over_sampling import RandomOverSampler`

#### 5b. Down-sampling (Under-sampling)

**Mechanics:** Randomly remove majority class samples until classes are balanced.

**Pros:** Reduces training time; avoids optimistic bias in resampling better than up-sampling.

**Cons:** Discards potentially useful majority class information.

**Internal down-sampling in Random Forest:** Random forest can down-sample internally during bootstrap by using class as a stratification variable. Each tree in the ensemble then sees a balanced bootstrap sample. This approach showed the **most robust** results in the chapter's experiments (maintained AUC-ROC on both evaluation and test sets while improving sensitivity/specificity trade-off).

#### 5c. SMOTE (Synthetic Minority Over-Sampling Technique)

**Mechanics (Chawla et al. 2002):**
1. Randomly select a minority class sample.
2. Find its K nearest neighbors (default K=5) in the minority class.
3. Randomly interpolate between the selected point and one of its neighbors to create a new synthetic sample.
4. Optionally also down-sample the majority class.

**Key property:** SMOTE fills in the space encompassing the minority class rather than expanding it — new synthetic samples are very similar to existing minority class samples. This makes boundaries less rigid than simple duplication.

**Parameters to tune:**
- Amount of over-sampling of minority class
- Amount of down-sampling of majority class
- Number of neighbors K used for synthesis

**Pros:**
- Creates new (not just duplicate) minority class samples.
- Can yield better calibrated decision boundaries.
- Allows better sensitivity/specificity trade-offs compared to cutoff adjustment alone.

**Cons:**
- Resampling performance estimates can still be biased (synthetic samples may appear in both train and holdout folds).
- No universally superior to simpler methods — results vary by model and dataset.

**Python:** `from imblearn.over_sampling import SMOTE`

```python
from imblearn.over_sampling import SMOTE
sm = SMOTE(k_neighbors=5, random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
```

**Empirical results summary from chapter** (Table 16.3, random forest, caravan data):

| Method | Eval AUC | Test AUC | Sensitivity | Specificity |
|---|---|---|---|---|
| Original | 0.757 | 0.738 | 64.4% | 75.9% |
| Down-sampling | 0.794 | 0.730 | 81.4% | 70.3% |
| Down-sampling (internal) | 0.792 | **0.764** | 78.0% | 68.3% |
| Up-sampling | 0.755 | 0.739 | 71.2% | 68.1% |
| SMOTE | 0.767 | 0.747 | 78.0% | 67.7% |

Key takeaway: Internal down-sampling (via random forest's bootstrap stratification) was the **most robust** — it held its test AUC closest to its evaluation AUC. SMOTE showed overfitting between eval and test. Up-sampling gave negligible AUC improvement.

**General finding from literature (Weiss & Provost 2001, Batista et al. 2004, Van Hulse et al. 2007):** Sampling can mitigate imbalance but there is no single best method — different models react differently, and results must be validated.

---

### Technique 6: Cost-Sensitive Training

**Mechanics:** Instead of optimizing accuracy or Gini impurity, incorporate asymmetric misclassification costs directly into the model's training objective. Making a false negative (missing a bankrupt company) X times more costly than a false positive.

**Decision rule with costs** (two-class case):

```
Predict class 1 if:   p1/p2  >  (C(1|2) * pi_2) / (C(2|1) * pi_1)
```

For equal priors: predict class 1 when `p1/p2 > C(1|2)/C(2|1)`, i.e., the probability ratio exceeds the cost ratio.

**Cost-sensitive Gini criterion** (Breiman et al. 1984, CART):

```
Gini* = [C(1|2) + C(2|1)] * p1 * p2
```

Note: the Gini criterion symmetrizes costs — C(i|j) and C(j|i) are averaged. This means you cannot independently tune false positive vs. false negative costs through the split criterion alone.

**Model-specific notes:**

- **SVM (class-weighted):** Assign unequal costs per class (not per error type). High class weights for minority class increase penalty for misclassifying those points. Cannot produce calibrated probabilities when class weights are used — must use discrete class predictions only. Tune the weight ratio alongside the C parameter.
- **CART (rpart):** Uses a cost matrix `loss` passed via the `parms` argument. Cost structure is: `matrix[true_class, predicted_class]`. Cost-sensitive trees tend to be smaller — they greedily focus on minimizing costly errors and stop splitting when remaining errors are cheap.
- **C5.0:** Also accepts a cost matrix (using the transpose convention vs. rpart). Models optimized for lower cost values maximize Kappa; higher costs increase sensitivity but plateau around cost=25-30x.

**Important caveat:** With unequal costs, predicted class probabilities and discrete class predictions may be inconsistent. The final class prediction is a function of both the probability and the cost structure. Do NOT use raw class probabilities as confidence scores when cost-sensitive training is used.

**Recommendation for bankruptcy task:** Use `class_weight` in sklearn (which effectively implements cost-sensitive training for most classifiers). For gradient boosting (XGBoost, LightGBM), use `scale_pos_weight` (XGBoost) or `is_unbalance=True` / `class_weight` (LightGBM). Tune the cost ratio via cross-validated F1.

---

### Metrics Guide

| Metric | Definition | Notes for imbalanced data |
|---|---|---|
| **Accuracy** | (TP + TN) / total | Misleading — a model predicting all non-bankrupt gets 97% accuracy |
| **Sensitivity (Recall)** | TP / (TP + FN) | Rate at which true bankruptcies are detected; critical for this task |
| **Specificity** | TN / (TN + FP) | Rate at which true non-bankruptcies are correctly classified |
| **Precision** | TP / (TP + FP) | Of all predicted bankrupt, how many actually are |
| **F1-score** | 2 * (Precision * Recall) / (Precision + Recall) | Competition metric; balances precision and recall equally |
| **Kappa** | Measures agreement beyond chance | Better than accuracy under imbalance; 0 = random, 1 = perfect |
| **AUC-ROC** | Area under ROC curve | Threshold-independent; good for model comparison; use for tuning |
| **Youden's J** | Sensitivity + Specificity - 1 | Maximized at the "best" ROC cutoff; not the same as F1 |
| **Lift** | At x% of population contacted, what % of events found | Useful for operational decisions; similar to ROC under severe imbalance |

**Key insight:** AUC-ROC is the best metric for comparing models against each other (it is cutoff-independent). But the final operational metric is F1, so after selecting a model via AUC-ROC, tune the probability cutoff to maximize F1 on a held-out validation set.

**When AUC-ROC and lift curves look very similar:** This is a symptom of severe class imbalance (as shown in Fig. 16.1 of the chapter). Under normal class balance, these curves diverge in shape.

---

### Practical Recommendations for the Bankruptcy Task (97/3 split)

1. **Never optimize for accuracy.** The competition metric is F1. Always tune using F1 or AUC-ROC in cross-validation.

2. **Use stratified k-fold cross-validation** so that each fold maintains the ~3% event rate. Use `StratifiedKFold` in sklearn.

3. **Apply at least one of:** class weighting, SMOTE, or internal down-sampling (random forest). Simple down-sampling or up-sampling alone had limited effect in experiments; class weights and SMOTE performed best.

4. **Derive the probability cutoff from a validation set, not the training or test set.** Use a dedicated eval split or out-of-fold predictions to find the threshold maximizing F1.

5. **For model selection, use AUC-ROC** (threshold-independent, fair model comparison). For final threshold selection, switch to F1.

6. **Random forest with internal stratified bootstrap** was the most robust sampling approach in the chapter's experiments — it maintained consistent AUC-ROC between evaluation and test sets.

7. **For gradient boosting and SVMs,** use class weights (`scale_pos_weight` in XGBoost, `class_weight` in sklearn) rather than resampling. These methods integrate cost-sensitivity directly into training.

8. **SMOTE warning:** When using SMOTE before cross-validation, synthetic samples may appear in both train and validation folds of the same CV split, causing optimistic performance estimates. Always apply SMOTE **inside** the cross-validation loop (on training folds only), never on the full dataset before splitting.

9. **Do not trust class probabilities from cost-sensitive models** (SVM with class weights, CART with cost matrix) for ranking or cutoff selection — use Kappa, sensitivity, and specificity instead.

10. **Reasonable starting point for cost/weight ratio:** With a 97/3 split, start with a minority class weight of ~32 (the inverse of the class frequency ratio) and tune from there.

---

### Model-Specific Recommendations

| Model | Imbalance Strategy | Parameter |
|---|---|---|
| Logistic Regression | Class weights | `class_weight='balanced'` or `class_weight={0:1, 1:32}` |
| Random Forest | Internal stratified bootstrap OR class weights | `class_weight='balanced_subsample'`; `max_samples` with stratification |
| SVM | Class weights (no probabilities when used) | `class_weight='balanced'` |
| CART / Decision Tree | Cost matrix or class weights | `class_weight='balanced'` |
| XGBoost | Scale positive weight | `scale_pos_weight=97/3 ≈ 32` |
| LightGBM | Class weights or imbalance flag | `is_unbalance=True` or `class_weight='balanced'` |
| Naive Bayes / LDA | Adjust prior probabilities | Set priors to balanced values (e.g., 0.5/0.5) |
| Any model | SMOTE on training data | Apply inside CV loop only |
| Any model | Alternate cutoff | Use eval set + `precision_recall_curve` to find F1-maximizing threshold |

### Key Formulas

```
# F1-score (competition metric)
F1 = 2 * (Precision * Recall) / (Precision + Recall)

# Youden's J (for ROC-based cutoff selection)
J = Sensitivity + Specificity - 1

# Decision rule with costs and priors
# Predict minority class (1) if:
p1/p2 > (C(1|2) * prior_2) / (C(2|1) * prior_1)

# For equal costs, starting class weight ratio
weight_minority = n_majority / n_minority  # e.g., 5277/178 ≈ 30 for bankruptcy data

# SMOTE default parameterization
k_neighbors = 5  # number of nearest neighbors for synthesis

# Cost-sensitive Gini (CART)
Gini* = [C(1|2) + C(2|1)] * p1 * p2
```
