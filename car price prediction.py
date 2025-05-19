import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
import numpy as np

#Data collection and processing
car_dataset=pd.read_csv(r'C:\Users\thenn\OneDrive\Desktop\car_prediction_data.csv')

#inspecting the 5 rows in the dataset

print(car_dataset.head())
#getting the number of rows and columns count in the form of tuple
print(car_dataset.shape)
#getting information about the car dataset
print(car_dataset.info())

#checking the null values in the dataset
print(car_dataset.isnull().sum())

#checking the distrtibution of categorical data
print(car_dataset['Fuel_Type'].value_counts())
print(car_dataset['Seller_Type'].value_counts())
print(car_dataset['Transmission'].value_counts())



#encoding the categorical data into the numerical data
car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2},"Seller_Type":{'Dealer':0,'Individual':1},'Transmission':{'Manual':0,'Automatic':1}},inplace=True)
print(car_dataset.head())


#split the data and target features
x,y=car_dataset.drop(columns=['Car_Name',"Selling_Price"],axis=1),car_dataset['Selling_Price']

#training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=2)
print(x.shape,x_train.shape,x_test.shape)

#model prediction1
lin_regmodel=LinearRegression()
lin_regmodel.fit(x_train,y_train)

x_train_pred1=lin_regmodel.predict(x_train)
error1=metrics.r2_score(x_train_pred1,y_train)
print(f'accuarcy of training (linear):{error1}')


x_test_pred1=lin_regmodel.predict(x_test)
error2=metrics.r2_score(x_train_pred1,y_train)
print(f'accuarcy of training (linear):{error2}')


plt.scatter(y_train,x_train_pred1)
plt.xlabel('actual price')
plt.ylabel('prediction price')
plt.title('actual price vs prediction price')
plt.show()



plt.scatter(y_test,x_test_pred1)
plt.xlabel('actual price')
plt.ylabel('prediction price')
plt.title('actual price vs prediction price')
plt.show()



#model prediction2
lasso_regmodel=Lasso()
lasso_regmodel.fit(x_train,y_train)

x_train_pred2=lasso_regmodel.predict(x_train)
error3=metrics.r2_score(x_train_pred2,y_train)
print(f'accuarcy of training (linear):{error3}')


x_test_pred2=lasso_regmodel.predict(x_test)
error4=metrics.r2_score(x_train_pred2,y_train)
print(f'accuarcy of training (linear):{error4}')


plt.scatter(y_train,x_train_pred2)
plt.xlabel('actual price')
plt.ylabel('prediction price')
plt.title('actual price vs prediction price')
plt.show()



plt.scatter(y_test,x_test_pred2)
plt.xlabel('actual price')
plt.ylabel('prediction price')
plt.title('actual price vs prediction price')
plt.show()



#lasso predicting accurately compared to linearreggression

input_data = np.array(x_train.iloc[44]).reshape(1, -1)
actual_price = y_train.iloc[44]

# Predict using Lasso model
predicted_price = lasso_regmodel.predict(input_data)[0]

print("\nSample Prediction:")
print(f"Actual Price   : {actual_price}")
print(f"Predicted Price: {predicted_price:.2f}")