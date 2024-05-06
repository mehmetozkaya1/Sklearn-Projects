import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import RidgeClassifier, LogisticRegression

train = pd.read_csv("diabetes_train.csv")
test = pd.read_csv("diabetes_test.csv")

corr = train.corr()
print(corr)

x_train = train.drop("Outcome", axis=1)
x_train = x_train[["Pregnancies", "Glucose", "BMI", "Age"]]
x_test = test.drop("Outcome", axis=1)
x_test = x_test[["Pregnancies", "Glucose", "BMI", "Age"]]
y_train = train.loc[:, "Outcome"]
y_test = test.loc[:, "Outcome"]

dec_tree = DecisionTreeClassifier()
rand_tree = RandomForestClassifier()
neural_network = MLPClassifier()
bernoulli = BernoulliNB()
gaussian = GaussianNB()
ridge = RidgeClassifier()
log_reg = LogisticRegression()

dec_tree.fit(x_train, y_train)
rand_tree.fit(x_train, y_train)
neural_network.fit(x_train, y_train)
bernoulli.fit(x_train, y_train)
gaussian.fit(x_train, y_train)
ridge.fit(x_train, y_train)
log_reg.fit(x_train, y_train)

dec_pred = dec_tree.predict(x_test)
rand_pred = rand_tree.predict(x_test)
nn_pred = neural_network.predict(x_test)
bernoulli_pred = bernoulli.predict(x_test)
gauss_pred = gaussian.predict(x_test)
ridge_pred = ridge.predict(x_test)
log_pred = log_reg.predict(x_test)

print("*****Decision Tree*****")
print(classification_report(y_test, dec_pred))
print("*****Random Forest*****")
print(classification_report(y_test, rand_pred))
print("*****Neural Network*****")
print(classification_report(y_test, nn_pred))
print("*****Bernoulli NB*****")
print(classification_report(y_test, bernoulli_pred))
print("*****Gaussian NB*****")
print(classification_report(y_test, gauss_pred))
print("*****Ridge Classifier*****")
print(classification_report(y_test, ridge_pred))
print("*****Logistic Regression*****")
print(classification_report(y_test, log_pred))
print("Confusion Matrixes")
print("*****Decision Tree*****")
print(confusion_matrix(y_test, dec_pred))
print("*****Random Forest*****")
print(confusion_matrix(y_test, rand_pred))
print("*****Neural Network*****")
print(confusion_matrix(y_test, nn_pred))
print("*****Bernoulli NB*****")
print(confusion_matrix(y_test, bernoulli_pred))
print("*****Gaussian NB*****")
print(confusion_matrix(y_test, gauss_pred))
print("*****Ridge Classifier*****")
print(confusion_matrix(y_test, ridge_pred))
print("*****Logistic Regression*****")
print(confusion_matrix(y_test, log_pred))

# Most important factors for diabetes is glucose level, BMI, age and pregnancies. So we should use these features. (Corralations investigated) # Logistic Regression is the best