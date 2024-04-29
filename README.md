# Telecommunication Company Customer Churn Prediction: Data Science Project

![satelite image](docs/satelite.png)

## Overview
Customer churn analysis is a critical task for telecommunications companies, allowing them to predict and prevent customer attrition.

This project focuses on building machine learning models in Python to analyze and predict customer churn. The project focuses on exploratory data analysis (EDA), model development, and evaluation.

### Dataset

The dataset used for this project is available [here](https://www.kaggle.com/datasets/shilongzhuang/telecom-customer-churn-by-maven-analytics).

### Objectives

The main objectives of this project are:

- Clean and preprocess the dataset, handling missing values and incorrect data types.
- Perform exploratory data analysis (EDA) to understand the dataset's characteristics and relationships.
- Develop machine learning models to predict customer churn.
- Evaluate model performance and select the best-performing model.

### Key Insights

- The dataset consists of 7043 entries with 33 columns, including demographic details, service usage, and churn-related attributes.
- Initial data preprocessing involved handling incorrect data types and missing values.
- Exploratory analysis revealed imbalances in target classes and relationships between features.
- Feature engineering included creating dummy variables for categorical features and handling multicollinearity.
- Data normalization was applied to ensure consistent scaling across variables.

### EDA Highlights
- Distribution analysis of target variable (`Churn Label`) revealed class imbalance.
- Significant imbalances were observed in features such as `Senior Citizen`, `Dependents`, and `Phone Service`.
- Most customers opted for `Month-to-month` contracts and `Electronic check` payment methods.
- `Tenure Months` and `Monthly Charges` showed distinct distribution patterns among customers.

### Model Selection and Performance

Four models were compared based on accuracy scores:

| Model           | Accuracy 	|
|-----------------|-------------|
| KNeighbors      | 0.827     	|
| Logistic Regression | 0.927 	|
| Random Forest   | 0.927     	|
| gradient boosting   | 0.929   |
| dummy classifier   | 0.739    |
| SVC   | 0.918     		|

The gradient boosting outperformed other models with an accuracy of 0.929. 

### Model Evaluation

### Classification Report

|           | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| No        | 0.95      | 0.95   | 0.95     | 1302    |
| Yes       | 0.86      | 0.87   | 0.86     | 459     |
|           |           |        |          |         |
| **Accuracy** |           |        | 0.93     | 1761    |
| **Macro Avg** | 0.91      | 0.91   | 0.91     | 1761    |
| **Weighted Avg** | 0.93   | 0.93   | 0.93     | 1761    |

This classification report summarizes the performance metrics (precision, recall, F1-score, and support) of a machine learning model for binary classification. The model achieved an accuracy of 93% on a dataset with 1761 instances, consisting of two classes: "No" and "Yes". The report provides insights into the model's ability to correctly classify instances of each class based on precision, recall, and F1-score.

### Hyperparameter tuning
Using GridSearchCV to search for the best hyperparameter setting. Resulting The tuned parameters for this model include:

```json
{
	"learning rate": 0.2,
	"max_depth": 3,
	"n_estimator": 100,

}
```

### How to Use
To use this project:
1. Clone the repository from GitHub.
2. Ensure Python and required libraries are installed.
3. Run the Jupyter Notebook or Python script.
4. Explore data preprocessing, EDA, model development, and evaluation sections.
5. Deploy the best-performing model for predicting customer churn.


### Contact

For questions or collaborations, feel free to reach out to [Naufal Fauzan](https://github.com/andnaufal).
