import pandas as pd
data = pd.read_csv('../../../Customer-Churn-Analysis-Data-Science-Project/data/raw/Telco_customer_churn.csv')
data.to_pickle("../../data/interim/telco_churn.csv")