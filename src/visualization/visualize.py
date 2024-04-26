import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
import math

data = pd.read_pickle("../../../Customer-Churn-Analysis-Data-Science-Project/data/processed/telco_churn_clean.csv")

#distribution of target analysis
df = data
sns.countplot(x ='Churn Label', data = df)
plt.show()

#distribution of independece variable 
fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True) 
# This line of code is creating a subplot grid for multiple plots using the subplots function from Matplotlib.
sns.countplot(x="Gender", data=df, ax=axes[0, 0])
sns.countplot(x="Senior Citizen", data=df, ax=axes[0, 1])
sns.countplot(x="Partner", data=df, ax=axes[0, 2])
sns.countplot(x="Paperless Billing", data=df, ax=axes[1, 0])
sns.countplot(x="Dependents", data=df, ax=axes[1, 1])
sns.countplot(x="Phone Service", data=df, ax=axes[1, 2])
plt.show()

sns.countplot(x="Internet Service", data = df)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True) 
sns.countplot(x="Contract", data=df, ax=axes[0])
sns.countplot(x="Payment Method", data=df, ax=axes[1])
plt.show()

fig, axes = plt.subplots(1,2, figsize=(12, 7))
sns.histplot(df["Tenure Months"], ax=axes[0])
sns.histplot(df["Monthly Charges"], ax=axes[1])
plt.show()

## Set and compute the Correlation Matrix
sns.set(style="white")
corr = data.corr()


# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure and a diverging colormap
f, ax = plt.subplots(figsize=(18, 15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.5, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()





