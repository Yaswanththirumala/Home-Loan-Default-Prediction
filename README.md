# Home Loan Default Prediction (PRCP-1006)

Machine learning project that predicts which home loan customers are likely to default, using application data and customer credit history across 7 related datasets, so a bank can prioritize risky applications for review.

## Business Problem
A bank wants to understand which customers and customer segments are more likely to default on a loan, and to identify the factors behind it. The project covers a full data analysis report, a predictive model, a model comparison, and a report on the challenges faced.

## Project Tasks
1. Prepare a complete data analysis report on the given data.
2. Build a predictive model to identify the factors and customer segments linked to loan default.
3. Compare multiple models and recommend a candidate model for production.
4. Report the challenges faced and the techniques used.

## Dataset
Seven related tables, all linked by customer ID (`SK_ID_CURR`):

| Dataset | Rows | Columns |
|---|---|---|
| application_train | 307,511 | 122 |
| bureau | 1,716,428 | 17 |
| bureau_balance | 27,299,925 | 3 |
| POS_CASH_balance | 10,001,358 | 8 |
| credit_card_balance | 3,840,312 | 23 |
| previous_application | 1,670,214 | 37 |
| installments_payments | 13,605,401 | 8 |

- Target: `TARGET` (1 = defaulted, 0 = did not default)
- Only 8.07% of customers defaulted, so the target is heavily imbalanced.
- No duplicate rows were found in any dataset.
- The data is not included in this repository because of its size. It is available from the course project source.

## Approach
1. **Data quality checks:** shapes, data types, missing values and duplicates in all 7 tables
2. **EDA:** target distribution, default rate by customer group, numeric features vs target, correlation analysis
3. **Feature engineering:** combined the history tables into one row per customer using aggregated values (mean, sum, min, max) plus record counts, giving 390 columns for 307,511 customers
4. **Preprocessing pipeline:** median imputation and scaling for numeric features; most-frequent imputation and one-hot encoding for categorical features
5. **Class imbalance:** stratified 80/20 train/test split and class weights
6. **Modeling:** compared 3 classification models and selected a candidate using F1-score
7. **Risk segmentation:** grouped customers into lower, medium and higher risk by predicted default probability

## Model Comparison
Results on the hold-out test set (61,503 customers):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Random Forest | 0.8091 | 0.2073 | 0.4834 | 0.2902 | 0.7461 | 0.2193 |
| Logistic Regression | 0.7067 | 0.1730 | 0.6969 | 0.2772 | 0.7711 | 0.2566 |
| Decision Tree | 0.6606 | 0.1484 | 0.6761 | 0.2433 | 0.7155 | 0.1952 |

**Candidate model: Random Forest**, selected for the highest F1-score. Accuracy alone is misleading here because 92% of customers did not default. Logistic Regression has the better ROC-AUC and recall, so it is a strong choice if catching more defaulters matters more than reducing false alarms.

## Key Insights
- **Risk segments separate customers well.** In the test set, the actual default rate was 1.3% for the lower-risk group, 5.3% for medium risk and 20.7% for higher risk. The higher-risk group defaults about 15 times as often as the lower-risk group.
- **Most important features (Random Forest):** the three external source scores (`EXT_SOURCE_2`, `EXT_SOURCE_3`, `EXT_SOURCE_1`), customer age (`DAYS_BIRTH`), and credit bureau history such as the age of previous credits and employment length.
- **Credit history matters.** Many of the top features come from the aggregated credit bureau and installment payment tables, which supports combining all 7 datasets.

## Recommendations
- Use the predicted default probability to prioritize applications for extra review.
- Combine historical credit behavior with current application details.
- Use the risk segments to apply different verification steps.
- Do not use the model as the only basis for approving or rejecting a loan.
- Monitor performance, drift and fairness after deployment.

## Challenges and Techniques
| Challenge | Technique | Reason |
|---|---|---|
| Class imbalance (8.07% defaults) | Stratified split and class weights | Keep both classes represented and give more weight to defaults |
| Many missing values | Median and most-frequent imputation | Train without deleting large numbers of customers |
| Categorical variables | One-hot encoding | Convert categories to numbers and handle unseen values safely |
| Different numeric scales | StandardScaler | Comparable scales for linear models |
| Multiple history tables | Customer-level aggregation on `SK_ID_CURR` | One modeling row per customer |
| Customer ID in features | Removed before modeling | An identifier is not a meaningful predictor |
| Large data (27M+ rows in one table) | Aggregation before modeling | Keep the modeling table manageable |
| Model selection | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC | A fuller view than accuracy alone |

## Limitations
- Results come from a single hold-out split; time-based validation would better reflect real use.
- Precision is low (about 21%), so many flagged customers would not actually default.
- Historical data may contain bias or change over time.
- Predicted probabilities should be calibrated before being used as business risk scores.
- Production use would need fairness testing, explainability, monitoring and compliance review.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Jupyter Notebook

## How to Run
1. Clone the repository.
2. Install the requirements: `pip install -r requirements.txt`
3. Download the dataset from the course project source, extract it, and place the 7 CSV files in a folder named `Data` beside the notebook.
4. Run `PRCP-1006-HomeLoanDef_Complete.ipynb` from top to bottom. This needs a lot of memory, since some tables have millions of rows.
