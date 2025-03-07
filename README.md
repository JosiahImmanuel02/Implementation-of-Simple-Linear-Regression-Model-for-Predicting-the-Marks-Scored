# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## NAME:JOSIAH IMMANUEL A
## REGNO:212223043003

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse) 
*/
```

## Output:

## HEAD VALUES

![Screenshot 2025-03-07 090534](https://github.com/user-attachments/assets/ca7d6c7e-b918-4d6d-97b3-7b3ecccc0ce9)


## TAIL VALUES

![Screenshot 2025-03-07 090706](https://github.com/user-attachments/assets/defc45a4-f9c0-44d2-b8ac-a367bdd19bcf)


## COMPARE DATASET

![Screenshot 2025-03-07 090742](https://github.com/user-attachments/assets/1e1c584d-c1c6-449a-b8e5-1c86a7a1e007)


## PREDICATION OF X AND Y VALUES 

![Screenshot 2025-03-07 091938](https://github.com/user-attachments/assets/3034cb55-3909-4dcb-a49e-934b3355b731)


## TRAINING SET

![Screenshot 2025-03-07 090837](https://github.com/user-attachments/assets/0951291e-d4f2-4ed7-9395-3a07f85d07a3)


## TESETING TEST

![Screenshot 2025-03-07 090852](https://github.com/user-attachments/assets/13496d97-7d4d-4741-90d5-4e64a98abc5d)


## MSE,MAE and RMSE

![Screenshot 2025-03-07 090916](https://github.com/user-attachments/assets/b67c30d3-45ee-42ea-830b-dcf5aad33f20)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
