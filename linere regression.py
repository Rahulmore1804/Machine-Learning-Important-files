#librarby called

from statistics import linear_regression
from tkinter import Variable
from unittest import result
import numpy as np
import matplotlib.pyplot as plt
import pandas  as pd

# dataset called

dataset = pd.read_csv('Salary_Data.csv')
# print(dataset)
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
# print(x)
# print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.3,random_state=42)
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)


# applying MLmodel
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# predicting test result
y_pred = regressor.predict(x_test)

# visualising result of training
plt.scatter(x_train,y_train,color = 'yellow')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('salary vs exp of trainset')
plt.xlabel("years")
plt.ylabel('salary')
plt.show()

# visualising result of testing
plt.scatter(x_test,y_test,color = 'yellow')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('salary vs exp of testset')
plt.xlabel("years")
plt.ylabel('salary')
plt.show()



# for knowing value for 1 Variable

print(regressor.predict([[12]]))


# Therefore, our model predicts that the salary of an employee with 12 years of experience is $ 138967,5.
# **Important note:** Notice that the value of the feature (12 years) was input in a double pair of square brackets. 
# That's because the "predict" method always expects a 2D array as the format of its inputs. 
# And putting 12 into a double pair of square brackets makes the input exactly a 2D array. Simply put:
# 12  {scalar}
# [12]  {1D array}
# [[12]] {2D array}


print(regressor.coef_)
print(regressor.intercept_)



# [9345.94244312]
# 26816.192244031183
# Therefore, the equation of our simple linear regression model is:
# Salary=9345.94×YearsExperience+26816.19 
# Important Note: To get these coefficients we called the "coef_" and "intercept_" attributes from our regressor object.
#  Attributes in Python are different than methods and usually return a simple value or an array of values.










