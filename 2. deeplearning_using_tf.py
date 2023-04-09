import pandas as pd
import numpy as np
df = pd.read_csv("data.csv")
print(df.head())
feats = ['department','salary']
df_final = pd.get_dummies(df,columns=feats,drop_first=True)
from sklearn.model_selection import train_test_split
X = df_final.drop(['left'],axis=1).values
y = df_final['left'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(9, kernel_initializer = "uniform",activation = "relu", input_dim=18))
classifier.add(Dense(1, kernel_initializer = "uniform",activation = "sigmoid"))
classifier.compile(optimizer= "adam",loss = "binary_crossentropy",metrics = ["accuracy"])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 1)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
