# Per-Nurse Day-Grouped Cross-Validation Results Summary

## Overview
This analysis trains separate Decision Tree and Random Forest models for each nurse using leave-one-day-out cross-validation (LODO-CV). Each nurse is modeled independently on their own data, with no information leakage across days.

## Key Constraints Applied
- **Data Source**: `data/Aditya/` (nurses CE and EG excluded due to missing class 0 labels)
- **Window Strategy**: 30-second sliding windows with 5-second stride, computed separately per day
- **Day Grouping**: All windows from a test day stay in test; no cross-day leakage
- **Class Balance Filtering**: 
  - Train: ≥200 windows, ≥30 class 0, ≥30 class 1, each class ≥5%
  - Test: ≥100 windows, ≥20 class 0, ≥20 class 1, each class ≥5%
- **Normalization**: Per-nurse z-score normalization computed on all training windows

## Nurses & Folds Summary

| Nurse | Total Days | Days with Windows | Folds Attempted | Folds Valid |
|-------|-----------|------------------|-----------------|-------------|
| 6B    | 6         | 6                | 6               | 2           |
| 7E    | 5         | 5                | 5               | 1           |
| 83    | 8         | 8                | 8               | 3           |
| 8B    | 5         | 5                | 5               | 2           |
| 94    | 9         | 9                | 9               | 3           |
| E4    | 11        | 11               | 11              | 4           |

**Total**: 6 nurses modeled, 15 valid folds (out of many attempted)

**Excluded Nurses** (reason):
- 15, 5C, 7A, BG, DF, F5: No valid folds after class-mix filtering (all test days were single-class or imbalanced)
- 6D: Insufficient days for cross-validation

## Average Performance Across All Valid Folds

| Model          | Accuracy | Bal. Accuracy | F1 (Binary) | F1 (Macro) | Precision | Recall |
|----------------|----------|---------------|-------------|-----------|-----------|--------|
| Decision Tree  | 0.4813   | 0.4477        | 0.4988      | 0.3168    | 0.4236    | 0.7085 |
| Random Forest  | 0.5228   | 0.4889        | 0.5545      | 0.3482    | 0.5873    | 0.7717 |

**Key Observations**:
- RF outperforms DT on all metrics
- High recall (~71-77%) but lower precision (~42-59%), indicating the models are biased toward positive predictions
- Balanced accuracy ~44-49%, suggesting difficulty discriminating between stress (1) and non-stress (0)

## Per-Nurse Performance

### Nurse 6B (2 valid folds)
- **Fold 1** (test day 2020-06-23): Both models predict all positive
  - Accuracy: 48%, Precision: 48%, Recall: 100%
  - Train heavily imbalanced (92% positive)
  
- **Fold 5** (test day 2020-06-29): More balanced
  - DT Accuracy: 85.7%, F1: 0.923
  - RF Accuracy: 85.6%, F1: 0.923
  - Test is 86% positive; models learn well

### Nurse 7E (1 valid fold)
- **Fold 3** (test day 2020-10-31): Very challenging
  - Train: 34% positive, Test: 82% positive (class mismatch)
  - DT: 17.6% accuracy (mostly predicts negative)
  - RF: 18.5% accuracy (predicts mostly negative, few positive)
  - High mismatch between train/test label distributions

### Nurse 83 (3 valid folds)
- **Fold 3** (test day 2020-11-02): Balanced, good performance
  - Train: 91.8% positive, Test: 74% positive
  - RF: 66.9% accuracy, F1: 0.801
  
- **Folds 4 & 5** (test days 2020-11-05, 2020-11-06): Extreme imbalance
  - Test: 27% and 20% positive (test skewed toward negative)
  - Both models overpredict positive → low accuracy (~20-27%)

### Nurse 8B (2 valid folds)
- **Fold 1** (test day 2020-07-15): Imbalanced
  - Accuracy: 47%, Recall: 100% (predicts all positive)
  - Train: 72% positive
  
- **Fold 3** (test day 2020-07-17): More balanced
  - Accuracy: 34-35%, both models still overpredict

### Nurse 94 (3 valid folds)
- Mixed performance; folds with high class imbalance show poor metrics
  - **Fold 5**: RF achieves 56.6% accuracy with 0.643 F1
  - **Fold 7**: RF achieves 69.1% accuracy (test: 43% positive)
  - **Fold 8**: Very low accuracy (~9-23%, test only 9% positive)

### Nurse E4 (4 valid folds, strongest performer)
- Consistently produces valid balanced folds
  - **Fold 2**: RF achieves 87.8% accuracy, 0.935 F1 (test: 88% positive)
  - **Fold 6**: RF achieves 76.6% accuracy, 0.868 F1 (test: 79% positive)
  - **Fold 8**: RF achieves 86% accuracy, 0.925 F1 (test: 93% positive)
  - **Fold 10**: RF achieves 35.7% accuracy (test: 40% positive, but train is 87% positive)

## Key Insights

### 1. **Class Imbalance is Critical**
- When test day has very different label distribution than train, models perform poorly
- Example: 7E Fold 3 has train at 34% positive, test at 82% positive → ~18% accuracy
- Example: 83 Folds 4-5 have test at ~27-20% positive, train at ~90% → ~20-27% accuracy

### 2. **Model Overpredicts Positive**
- High recall (70%+) but lower precision (40-60%) across the board
- Models learned from training data where positive is often dominant, so they're biased
- This is actually expected given the raw data: many nurses have heavily positive-skewed labels

### 3. **Random Forest Slightly Better**
- RF beats DT on nearly every fold
- RF average accuracy: 52.3% vs DT 48.1%
- RF average F1: 0.555 vs DT 0.499

### 4. **E4 Most Reliable**
- Only nurse with all 4 folds passing balance criteria
- Produces more diverse day-wise splits → more realistic train/test mismatch scenarios
- Other nurses have fewer valid folds due to imbalanced day distributions

### 5. **Data Quality Concerns**
- 6 out of 12 nurses (50%) were completely excluded due to lack of balanced days
- The valid folds often show extreme imbalance between train and test days
- This suggests each nurse's data has inherent class imbalance across different days/times

## Column Definitions

| Column | Meaning |
|--------|---------|
| `accuracy` | Fraction of correct predictions overall |
| `balanced_accuracy` | Average recall per class; good for imbalanced data |
| `f1_binary` | Harmonic mean of precision and recall for positive class |
| `f1_macro` | Average F1 across both classes |
| `precision` | TP / (TP + FP); what fraction of positive predictions are correct |
| `recall` | TP / (TP + FN); what fraction of actual positives were detected |
| `tn, fp, fn, tp` | Confusion matrix values |
| `train_pos_rate` | Proportion of positive class in training set |
| `test_pos_rate` | Proportion of positive class in test set |

## Recommendations

1. **Consider different CV strategies**:
   - Stratified day split instead of random days (ensure each day has similar class mix)
   - Temporal validation (train on earlier days, test on later) to simulate deployment

2. **Address class imbalance**:
   - Use class weights (already applied) or try SMOTE on training
   - Adjust decision threshold instead of default 0.5

3. **Investigate nurse-day patterns**:
   - Some nurses have naturally skewed label distribution; investigate if this reflects real stress patterns
   - Check if certain nurses/days have sensor/annotation quality issues

4. **Use ensemble across nurses**:
   - Current: per-nurse models only
   - Alternative: ensemble folds from valid nurses to produce a more stable model

5. **Lower class-mix thresholds**:
   - Current thresholds filter out most nurses
   - Consider relaxing to 10-20% minimum class rate to get more folds
   - Trade-off: more folds but potentially noisier evaluation
