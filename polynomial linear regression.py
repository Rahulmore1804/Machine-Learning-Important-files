import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values


from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(x,y)

from sklearn.preprocessing import PolynomialFeatures
poly_regbeg = PolynomialFeatures(degree=4)
x_poly = poly_regbeg.fit_transform(x)
poly_regressor = LinearRegression()
poly_regressor.fit(x_poly,y)

plt.scatter(x,y,color='red')
plt.plot(x,linear_regressor.predict(x),color='blue')
plt.title("truth or bloff with linear model")
plt.xlabel('postion')
plt.ylabel('salary')
plt.show()

plt.scatter(x,y,color='red')
plt.plot(x, poly_regressor.predict(poly_regbeg.fit_transform(x)),color='blue')
plt.title("truth or bloff with poly model")
plt.xlabel('postion')
plt.ylabel('salary')
plt.show()

X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, poly_regressor.predict(poly_regbeg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



print(poly_regressor.predict(poly_regbeg.fit_transform([[6.5]])))
print(linear_regressor.predict([[6.5]]))
