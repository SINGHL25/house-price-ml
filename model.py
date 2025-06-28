import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('../data/housing.csv')
X = data[['rm']]
y = data['medv']

model = LinearRegression()
model.fit(X, y)

print("Model trained.")
