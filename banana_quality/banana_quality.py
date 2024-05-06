# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Importing models 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# Report metric
from sklearn.metrics import classification_report, confusion_matrix

# Read and investigate data
data = pd.read_csv("banana_quality.csv")
# print(data.head())

# Create labelencoder object
labelencder = LabelEncoder()

# Prepare train and test sets
X = data.drop("Quality", axis=1)
Y = labelencder.fit_transform(data.loc[:, "Quality"])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# Create the models
log_reg = LogisticRegression()
ridge_class = RidgeClassifier()
gauss_nb = GaussianNB()
neural_network = MLPClassifier()
random_for = RandomForestClassifier()
dec_tree = DecisionTreeClassifier()

# Train the models
log_reg.fit(x_train, y_train)
ridge_class.fit(x_train, y_train)
gauss_nb.fit(x_train, y_train)
neural_network.fit(x_train, y_train)
random_for.fit(x_train, y_train)
dec_tree.fit(x_train, y_train)

# Create predictions
log_pred = log_reg.predict(x_test)
ridge_pred = ridge_class.predict(x_test)
gauss_pred = gauss_nb.predict(x_test)
nn_pred = neural_network.predict(x_test)
rand_pred = random_for.predict(x_test)
dec_pred = dec_tree.predict(x_test)

# Reports
log_rep = classification_report(y_test, log_pred)
print("*****Logistic Regression*****")
print(log_rep)
ridge_rep = classification_report(y_test, ridge_pred)
print("*****Ridge Classifier*****")
print(ridge_rep)
gauss_rep = classification_report(y_test, gauss_pred)
print("*****Gauss Naive Bayes*****")
print(gauss_rep)
nn_rep = classification_report(y_test, nn_pred)
print("*****Neural Network*****")
print(nn_rep)
dec_rep = classification_report(y_test, dec_pred)
print("*****Decision Tree*****")
print(dec_rep)
print("*****Confusion Matrixes*****")
log_rep = confusion_matrix(y_test, log_pred)
print("*****Logistic Regression*****")
print(log_rep)
ridge_rep = confusion_matrix(y_test, ridge_pred)
print("*****Ridge Classifier*****")
print(ridge_rep)
gauss_rep = confusion_matrix(y_test, gauss_pred)
print("*****Gauss Naive Bayes*****")
print(gauss_rep)
nn_rep = confusion_matrix(y_test, nn_pred)
print("*****Neural Network*****")
print(nn_rep)
dec_rep = confusion_matrix(y_test, dec_pred)
print("*****Decision Tree*****")
print(dec_rep)


# Neural Network is the best decision for this dataset