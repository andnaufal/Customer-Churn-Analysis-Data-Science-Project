import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = pd.read_pickle("../../../Customer-Churn-Analysis-Data-Science-Project/data/interim/telco_churn.csv")
#handling wrong data types
data['Total Charges'] = pd.to_numeric(data['Total Charges'], errors='coerce')
data['Total Charges'] = data['Total Charges'].astype("float")
# handling null values
data['Churn Reason'] = data['Churn Reason'].fillna("Not Churning")
data['Total Charges'] = data['Total Charges'].fillna(data['Total Charges'].mean())

#dropping feature that wont add any value
data=data.drop(['CustomerID', 'Count', 'Lat Long', 'Latitude', 'Longitude', 'Zip Code','Churn Reason','City','Churn Value','State','Country'], axis=1)

#Check Multicolinearity using VIF----------------------------------------------
numerical_data = data.select_dtypes(include=[np.number])
from statsmodels.stats.outliers_influence import variance_inflation_factor
def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

calc_vif(numerical_data)
#drop total charges to avoid multicolinearity
data = data.drop('Total Charges', axis=1)

data.to_pickle("../../data/processed/telco_churn_clean.csv")