import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#read the data
data = pd.read_pickle("../../data/processed/telco_churn_clean.csv")

#split the dependent variable from the independent variable
y = data['Churn Label']
x = data.drop('Churn Label', axis=1)

#split the data into training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size = .75) #puts 75 percent of the data into a training set and the remaining 25 percent into a testing set.
print('Shape of x_train and y_train: ',x_train.shape, y_train.shape)
print('Shape of x_test and y_test: ',x_test.shape, y_test.shape)

#convert the categorical into dummy variable
x_train_dum = pd.get_dummies(x_train[['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service',
       'Multiple Lines', 'Internet Service', 'Online Security',
       'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV',
       'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method']])

x_test_dum = pd.get_dummies(x_test[['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service',
       'Multiple Lines', 'Internet Service', 'Online Security',
       'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV',
       'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method']])

x_train = pd.concat([x_train, x_train_dum], axis = 1)
x_test = pd.concat([x_test, x_test_dum], axis = 1)

x_train.drop(['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service',
       'Multiple Lines', 'Internet Service', 'Online Security',
       'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV',
       'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method',], axis = 1, inplace = True)

x_test.drop(['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service',
       'Multiple Lines', 'Internet Service', 'Online Security',
       'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV',
       'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method'], axis = 1, inplace = True)

#handling imbalanced data using SMOTE
# Import the necessary libraries
from imblearn.over_sampling import SMOTE

# Creating an instance of SMOTE
smote = SMOTE()

# Balancing the data
x_train, y_train = smote.fit_resample(x_train, y_train)

#applying normalization and scalinng
sc_x = StandardScaler()
x_train2 = pd.DataFrame(sc_x.fit_transform(x_train))
x_train2.columns = x_train.columns.values
x_train2.index = x_train.index.values
x_train = x_train2

x_test2 = pd.DataFrame(sc_x.transform(x_test))
x_test2.columns = x_test.columns.values
x_test2.index = x_test.index.values
x_test = x_test2

#assesing multiple model
def create_models():
    models = []
    models.append(('dummy_classifier', DummyClassifier()))
    models.append(('k_nearest_neighbors', KNeighborsClassifier()))
    models.append(('logistic_regression', LogisticRegression()))
    models.append(('support_vector_machines', SVC()))
    models.append(('random_forest', RandomForestClassifier()))
    models.append(('gradient_boosting', GradientBoostingClassifier()))
    
    return models

# create a list with all the algorithms we are going to assess
models = create_models()

# test the accuracy of each model using default hyperparameters
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    # fit the model with the training data
    model.fit(x_train, y_train).predict(x_test)
    # make predictions with the testing data
    predictions = model.predict(x_test)
    # calculate accuracy 
    accuracy = accuracy_score(y_test, predictions)
    # append the model name and the accuracy to the lists
    results.append(accuracy)
    names.append(name)
    # print classifier accuracy
    print('Classifier: {}, Accuracy: {})'.format(name, accuracy))
    

#evaluating the model
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import os
clf = GradientBoostingClassifier() 
#establishing random_state for reproducibility
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, clf.predict(x_test), labels=clf.classes_)

# Define title and color
title = "kndjdfs"
color = 'black'

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()

plt.show()

# Fit Logistic Regression on the Training dataset:
    
clf = GradientBoostingClassifier() 
#establishing random_state for reproducibility
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)


from sklearn.metrics import classification_report

print(classification_report(y_test, clf.predict(x_test)))

#hyperparameter tuning using GridSearchCV
# Import necessary libraries
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for GridSearchCV
param_grid = {
	'n_estimators': [50, 100, 200],
	'learning_rate': [0.01, 0.1, 0.2],
	'max_depth': [3, 5, 7],
}

# Initialize the Gradient Boosting model
gb_model = GradientBoostingClassifier()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the model to the training data using GridSearchCV
grid_search.fit(x_train, y_train)

# Get the best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Make predictions on the test set using the best model
y_pred_best = best_model.predict(x_test)

# Evaluate the best model
accuracy_best = accuracy_score(y_test, y_pred_best)

# Print the results
print("Best Parameters:", best_params)
print(f"Best Model Accuracy: {accuracy_best}")

# applying the hyperparameter setting to the model
# Instantiate a new model with the best hyperparameters
best_model = GradientBoostingClassifier(learning_rate=0.2, max_depth=3, n_estimators=100)

# Retrain the model using the best hyperparameters on the entire training data
best_model.fit(x_train, y_train)

# Predict on the test set
y_pred = best_model.predict(x_test)

# Evaluate the model performance on the test set
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on the test set:", accuracy)

#save the model
import os
import pickle

# Absolute path to the models directory
save_dir = "../../Customer-Churn-Analysis-Data-Science-Project/src/models"

# File path
filename = os.path.join(save_dir, "GradientBoostingClassifier.pkl")

# Save the trained model
with open(filename, 'wb') as file:
    pickle.dump(clf, file)