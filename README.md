# Mid_term_project

# Bank Marketing Dataset Machine Learning Project

## The Task
The dataset comes from the UCI “Bank Marketing Data Set” and Kaggle mirror. It contains information about direct marketing campaigns (phone calls) of a Portuguese banking institution.  

**Goal:** Predict whether a client will subscribe to a term deposit (`y`).

## Approach
- **Exploratory Data Analysis (EDA):** Investigated demographics, previous campaign outcomes, and economic indicators.  
- **Preprocessing:**  
  - Missing values imputed.  
  - One-Hot Encoding for categorical features.  
  - Standardization of numerical features.  
  - Excluded `duration` (not usable for production).  
- **Feature Engineering:**  
  - Log transforms for skewed counts (`campaign`, `previous`, `pdays`).  
  - Flag for `pdays=999`.  
  - Interaction term `month × day_of_week`.  
  - Binned call intensity (`campaign_bin`).  
- **Class Balancing:**  
  - Tested both `class_weight="balanced"` and **SMOTE** oversampling.  
- **Models trained:**  
  - Logistic Regression  
  - kNN  
  - Decision Tree  
  - Gradient Boosting (sklearn)  
  - XGBoost  
  - LightGBM  

## Pipeline
Pipelines combined preprocessing, optional SMOTE, and the model.  
Evaluation was performed with **stratified train/validation split**. RandomizedSearchCV and Hyperopt (Bayesian optimization) were used for boosting hyperparameter tuning.  

## Results
Performance was measured with ROC AUC, F1, Precision, Recall.

| name              |   train_auc |   valid_auc |     f1 | precision |   recall |
|-------------------|-------------|-------------|--------|-----------|----------|
| XGBoost (Hyperopt)|      0.8330 |      0.8152 | 0.3726 |    0.6732 |   0.2575 |
| XGBoost           |      0.8384 |      0.8124 | 0.3745 |    0.6713 |   0.2600 |
| LightGBM          |      0.9215 |      0.8094 | 0.3979 |    0.6449 |   0.2877 |
| LogisticRegression|      0.7934 |      0.8015 | 0.3377 |    0.6946 |   0.2231 |
| DecisionTree      |      0.7930 |      0.7949 | 0.3755 |    0.6704 |   0.2608 |
| kNN               |      0.8700 |      0.7781 | 0.3634 |    0.6500 |   0.2522 |



- **Best model:** XGBoost (and LightGBM close behind).  
- Boosting models captured nonlinear interactions and gave the best AUC.  
- Logistic Regression is interpretable but weaker.  
- kNN underperformed in high dimensions.  
- Decision Tree was too simple and overfit.

## Feature Importance & SHAP
- **Top drivers:** previous campaign outcome, number of previous contacts, `euribor3m`, `emp.var.rate`, and call-related variables (`month`, `day_of_week`, `campaign`).  
- SHAP analysis confirmed that favorable past outcomes and good economic indicators increase the probability of subscription.  
- Age and job groups (e.g., retired, students, management) also showed expected effects.

## Threshold Tuning & Segmentation
- Threshold tuning using PR-curve improved recall at controlled precision.  
- Subgroup thresholds (e.g., retired vs. self-employed) allow tailored business strategies.

## Conclusions
- **Best model:** XGBoost, with AUC ≈ 0.81.  
- **Feature importance** aligns with business intuition.  
- **Next steps:**  
  - Further balancing (SMOTE variations, cost-sensitive training).  
  - Enrich with external economic features.  
  - Explore ensembles/stacking.  
  - Optimize thresholds for specific business objectives (recall vs. precision trade-off).  
  - Calculate business ROI (cost per call vs. profit per deposit).
