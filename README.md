# 📉 Customer Churn Prediction

This project predicts customer churn using machine learning to support proactive customer retention strategies.


# 📊 Dataset

Telco Customer Churn Dataset

7,043 customers

21 features

Churn rate: 26.5% (class imbalance present)


# 🧠 Approach

Data cleaning and preprocessing

Removed customerID to prevent data leakage

One-hot encoding using pd.get_dummies()

Feature scaling using StandardScaler

Logistic Regression as baseline model

Handled class imbalance using class_weight='balanced'

Model evaluation using Confusion Matrix, Classification Report, ROC-AUC

5-Fold Cross Validation to verify model stability


# 📊 Model Comparison

| Model                   | Recall | Precision | Accuracy |
| ----------------------- | ------ | --------- | -------- |
| Baseline Logistic       | 55%    | 64%       | 80%      |
| Logistic + Class Weight | 79%    | 49%       | 72%      |


# ✅ Final Model Selection

The Logistic Regression model with class_weight='balanced' was selected as the final model.

It improved churn recall from 55% to 79%, significantly reducing missed churn customers (false negatives).

Although overall accuracy decreased slightly, the higher recall better aligns with the business objective of identifying at-risk customers.

ROC-AUC remained strong at approximately 0.83, indicating stable discriminatory performance.

# 🎯 Business Impact

The final model can identify 79% of churn customers in advance, enabling businesses to:

Design targeted retention campaigns

Reduce revenue loss

Improve customer lifetime value


# 🔗 Author

Tabassum Shaikh

