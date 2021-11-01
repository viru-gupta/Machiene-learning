import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
data=pd.read_csv('iris_csv-210120-132620.csv')
iris=data.values
x=iris[:,:-1]
y=iris[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
#visulaization
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
sns.boxenplot(x='class',y='sepalwidth',data=data)
plt.figure(figsize=(10,10))
plt.subplot(2,2,2)
sns.boxenplot(x='class',y='sepallength',data=data)
plt.figure(figsize=(10,10))
plt.subplot(2,2,3)
sns.boxenplot(x='class',y='petallength',data=data)
plt.figure(figsize=(10,10))
plt.subplot(2,2,4)
sns.boxenplot(x='class',y='petalwidth',data=data)
#applying ml

#support vector classification
sv=SVC(gamma='auto',max_iter=10000)
sv.fit(x_train,y_train)
y_pred=sv.predict(x_test)
acc=round(accuracy_score(y_pred,y_test),2)*100
print('Support vector classification accuracy is : ',acc)
  
#decison tree classifier
tree=DecisionTreeClassifier(random_state=0)
tree.fit(x_train,y_train)
y_pred=tree.predict(x_test)
acc=round(accuracy_score(y_pred,y_test),2)*100
print('Dicesion tree classification accuracy is : ',acc)

#logitic regression
reg=LogisticRegression(max_iter=100000)
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
acc=round(accuracy_score(y_test,y_pred),2)*100
print('Support vector classification accuracy is : ',acc)
