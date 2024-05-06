# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Importing models
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Report metrics
from sklearn.metrics import classification_report, confusion_matrix

# Read and investigate the data
data = pd.read_csv("human_evo.csv")
# print(data.head())
# print(data.columns)
# print(data.info)
# print(data["Genus_&_Specie"].value_counts())
# print(data["Hip"].value_counts())
# print(data["Zone"].value_counts())
# print(data["Location"].value_counts())
# print(data["biped"].value_counts())
# print(data["Habitat"].value_counts())
# print(data["Jaw_Shape"].value_counts())

# LabelEncoder object
labelEncoder = LabelEncoder()

# Creating the datas
# X = data[["Hip", "Zone", "Location", "biped", "Habitat", "Jaw_Shape", "Height", "Incisor_Size", "Time", "Cranial_Capacity"]]
X = data.drop("Genus_&_Specie", axis=1)
for col in X.columns:
    X[col] = labelEncoder.fit_transform(X[col])

Y = labelEncoder.fit_transform(data.loc[:, "Genus_&_Specie"])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
print("Y_train : ", y_train)
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

# More than 1 choice, we have 1 accuracy