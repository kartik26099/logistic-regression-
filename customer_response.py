import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score

df=pd.read_csv(r"D:\coding journey\aiml\python\udemy\Machine Learning A-Z (Codes and Datasets)\Part 3 - Classification\Section 14 - Logistic Regression\Python\Social_Network_Ads.csv")
x=df.iloc[:, :-1].values
y=df.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)
y_predct=classifier.predict(x_test)
print(classifier.predict(x_test))
print("\n",y_test)

print(confusion_matrix(y_test,y_predct))
print(accuracy_score(y_test,y_predct))