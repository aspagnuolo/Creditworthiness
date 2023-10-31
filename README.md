# Creditworthiness Assessment for Credit Card Issuance

Assessing a customer's creditworthiness is crucial for financial institutions before granting credit. This project aims to classify customers based on their payment behaviors.

## Overview

The primary goal of this project is to:
- Understand the distribution and characteristics of the customers.
- Classify customers into "good", "bad", or "intermediate" based on their payment behaviors.
- Handle missing data and provide meaningful imputations.

## Key Observations

**DATA LABELING**
- We have more data on recent clients than older ones, suggesting a clearer view of their recent credit history.
- The majority of customers have paid their debts on time, but a significant portion has shown varying degrees of late payment.
  
**Classification Criteria**
- **Good Customers**: Those who have had no delays or only minor delays (status "0" or "1") in the past six months.
- **Bad Customers**: Those who have had severe delays (status "2", "3", "4", or "5") in the past six months or had late payments between 30 and 59 days (status "1") 2 or more times in the past six months.
- **Intermediate Customers**: Those who do not fit in the above two categories. They are further refined based on their recent payment behaviors.

**Handling Missing Data**
- The missing values in the `YEARS_EMPLOYED` column correspond to the `NAME_INCOME_TYPE` as "Pensioner". These missing values are imputed with zero, indicating that they are not currently employed.

## Model Evaluation & Feature Selection

**Classification Techniques**
- Various classification techniques were explored to assess their performance in predicting the creditworthiness of customers. This foundational work laid the groundwork for further refining the approach using ensemble methods.

**ENSEMBLE LEARNING**
- Techniques like RandomForest-XGBoost and RandomForest-kNN have been explored. The goal was to combine the strengths of different models to achieve balanced and robust performance metrics. Based on a balance between various metrics, the Ensemble of RandomForest and XGB seems to be the most promising choice.

**Recursive Feature Elimination - RFE**
- RFE was employed to maintain model interpretability and understand the relative importance of the features. This approach is especially beneficial in this context, where an interpretable relationship between features and output is desired.

## Conclusion

Understanding and predicting a customer's credit behavior is of utmost importance to lending institutions. This project serves as a starting point for financial institutions to make informed decisions based on recent payment behaviors and other relevant data points.

---

**Note**: For a detailed understanding and step-by-step analysis, please refer to the Jupyter notebook "Creditworthiness.ipynb" in this repository.
