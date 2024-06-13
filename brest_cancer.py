import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score

df=pd.read_csv(r"D:\coding journey\aiml\python\udemy\Machine Learning A-Z (Codes and Datasets)\Part 3 - Classification\Section 14 - Logistic Regression\Python\Final Folder\Dataset\breast_cancer.csv")
x=df.iloc[:, 1:-1].values
y=df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_predict=lr.predict(x_test)
print(accuracy_score(y_test,y_predict))

from sklearn.model_selection import cross_val_score
cr = cross_val_score(estimator=lr, X=x_train, y=y_train, cv=10)
print(cr.mean())