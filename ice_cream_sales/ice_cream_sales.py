import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv("ice_cream_sales.csv")
# print(data.head())

X = data.drop("Ice Cream Profits", axis=1)
Y = data.loc[:, "Ice Cream Profits"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=2)

linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

linear_pred = linear_model.predict(x_test)

comparison = pd.DataFrame({"Actual Values" : y_test, "Predictions" : linear_pred})
print(comparison)

err = mean_squared_error(y_test, linear_pred)
print(err)