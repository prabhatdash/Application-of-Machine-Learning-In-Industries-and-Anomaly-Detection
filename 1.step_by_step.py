import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset=pd.read_csv("insurance.csv")
print(dataset)

label=LabelEncoder()
label.fit(dataset.sex.drop_duplicates())
dataset.sex=label.transform(dataset.sex)
label.fit(dataset.smoker.drop_duplicates())
dataset.smoker=label.transform(dataset.smoker)
label.fit(dataset.region.drop_duplicates())
dataset.region=label.transform(dataset.region)
print(dataset)

X_lin=dataset.drop(['charges'],axis=1)
y_lin=dataset[['charges']]

X_lin_train,X_lin_test,y_lin_train,y_lin_test=train_test_split(X_lin,y_lin,test_size=0.3,random_state=42)
Linear_model=LinearRegression()
Linear_model.fit(X_lin_train,y_lin_train)
pred=Linear_model.predict(X_lin_test)
print(pred)
accuracy=Linear_model.score(X_lin_test,pred)
print(accuracy)

for idx,col_name in enumerate(X_lin_train.columns):
    print("The coefficient for {} is {}".format(col_name,Linear_model.coef_[0][idx]))
intercept=Linear_model.intercept_[0]
print(intercept)
