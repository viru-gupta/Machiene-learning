import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#extracting data
dataset=pd.read_csv('Salary_data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
#splitting
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#regression model
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_predi=regressor.predict(x_test)
#ploting train graph
plt.scatter(x_train,y_train,c='orange')
plt.plot(x_train,regressor.predict(x_train),c='green')
plt.title('train set')
plt.xlabel('Years of experience')
plt.ylabel('salary')
plt.show()
#ploting test graph
plt.scatter(x_test,y_test,c='pink')
plt.plot(x_test,regressor.predict(x_test),c='blue')
plt.title('test set')
plt.xlabel('Years of experience')
plt.ylabel('salary')
plt.show()
