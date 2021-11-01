#Today's Machiene Learning Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
#extraction
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,2].values
#polynomial regression
poly=PolynomialFeatures(degree=4)
x_poly=poly.fit_transform(x)
#into linear
le=LinearRegression()
le.fit(x_poly,y)
y_pred=le.predict(x_poly)
#ploting graph through matplotlib
plt.scatter(x,y,c='red')
plt.plot(x,y_pred,c='blue')
plt.title('Our Predictions (Polynomial Regression)')
plt.xlabel('Positions in company')
plt.ylabel('Salaries Offered')
plt.show()
print('yeah the salary results are in near predicted values =',*le.predict(poly.fit_transform([[6.5]])))
