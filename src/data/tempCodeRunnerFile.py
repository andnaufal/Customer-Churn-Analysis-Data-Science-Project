import pandas as pd
data = pd.read_csv("../../data/Telco_customer_churn.csv")
data.to_pickle("../../data/interim/telco_churn.csv")