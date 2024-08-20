# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sri Vignesh G
RegisterNumber:  212223040204
*/
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("student_scores.csv")

print(df.tail())
print(df.head())
df.info()

x = df.iloc[:, :-1].values  # Hours
y = df.iloc[:,:-1].values   # Scores

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

print("X_Training:", x_train)
print("X_Test:", x_test)
print("Y_Training:", y_train)
print("Y_Test:", y_test)

reg = LinearRegression()
reg.fit(x_train, y_train)

Y_pred = reg.predict(x_test)

print("Predicted Scores:", Y_pred)
print("Actual Scores:", y_test)

a = Y_pred - y_test
print("Difference (Predicted - Actual):", a)

plt.scatter(x_train, y_train, color="green")
plt.plot(x_train, reg.predict(x_train), color="red")
plt.title('Training set (Hours vs Scores)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test, y_test, color="blue")
plt.plot(x_test, reg.predict(x_test), color="green")
plt.title('Testing set (Hours vs Scores)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mae = mean_absolute_error(y_test, Y_pred)
mse = mean_squared_error(y_test, Y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
```

## Output:

### TRAINIG SET INPUT:
#### X_VALUES:
![Screenshot 2024-08-20 070445](https://github.com/user-attachments/assets/7bbfabec-e602-44bd-9eee-daa36af5b4c2)

#### Y_VALUES:
![Screenshot 2024-08-20 070500](https://github.com/user-attachments/assets/b7c27420-1747-45f5-8e82-37f657c4d29d)

### TEST SET:
#### X_VALUES:
![Screenshot 2024-08-20 070452](https://github.com/user-attachments/assets/a3e78b66-90c5-4e47-b832-4803967adbe8)

#### Y_VALUES:
![Screenshot 2024-08-20 070506](https://github.com/user-attachments/assets/f3cd0649-991d-47a0-b70b-db1f81348faf)


### TRAINING SET:
![download](https://github.com/user-attachments/assets/1821741a-01b3-49f3-b535-2face178de80)


### TEST SET:
![download](https://github.com/user-attachments/assets/940d8033-da2a-47d6-b79c-22719e493627)


### MEAN SQUARE ERROR, MEAN ABSOLUTE ERROR AND RMSE
![image](https://github.com/user-attachments/assets/feaf1e38-5d29-4b41-ac86-c579959446a8)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
