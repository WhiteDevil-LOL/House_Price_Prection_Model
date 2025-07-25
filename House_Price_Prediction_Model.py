import pandas
import sklearn
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error , mean_squared_error ,r2_score

data = pandas.read_csv("Data.csv")
X = data[['Area','NumBedrooms','NumBathrooms']]
y = data[['Price']]

print("--- Features ---")
print(X.head())
print("--- Target ---")
print(y.head())

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 1)
print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

model = LinearRegression()
print("--- Training Model ---")
model.fit(X_train,y_train)
print("Model Training Complete....")

y_pred = model.predict(X_test)
print("\nActual Prices(Test set): ")
print(y_test)
print("\n Predicted Prices on the given Test Set: ")
print(y_pred)

mse = mean_squared_error(y_test,y_pred)
print(f"\nMean Squared Error (MSE) on Test Set: {mse:.2f}")
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE) on Test Set: {rmse:.2f}")
r2 = r2_score(y_test,y_pred)
print(f"R-squared (R2) on Test Set: {r2:.2f}")

print("--- Model Coefficients ---")
for feature, coef in zip(X.columns, model.coef_.flatten()):
    print(f"{feature}: {coef:.2f}")
print(f"Intercept: {model.intercept_.item():.2f}")

new_data = pandas.read_csv("NewHouseData.csv")
X_new = new_data[['Area', 'NumBedrooms', 'NumBathrooms']]
predicted_prices = model.predict(X_new)
print("Predicted Price(s) For the New House(s):")
print(np.round(predicted_prices, 3))