import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
import wandb

wandb.init(project="Airbnb Tuning",name='Tuned RF')

data = pd.read_csv('C:/Users/delim/Desktop/AI in A&F Indiv Assignment/Practical Assessment/Prediction/dataset_8.csv')
# Train test split
X = data.iloc[:,data.columns != 'price']
Y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=7)
# Standardize dataset
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=475, max_depth=32)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

wandb.sklearn.plot_regressor(model, X_train, X_test, y_train, y_test)
wandb.sklearn.plot_outlier_candidates(model, X_train, y_train)
wandb.sklearn.plot_residuals(model, X_train, y_train)

RMSE = sqrt(mean_squared_error(y_test, y_pred))
MSE = mean_squared_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
wandb.log({"RMSE":RMSE, "MSE":MSE, "MAE":MAE})