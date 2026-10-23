"""Week 4 - Linear Regression on the House Prices dataset."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("../data/house_prices.csv")

features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
data = df[features + ["SalePrice"]].dropna()
X, y = data[features], data["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)

print("MAE :", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("R2  :", r2_score(y_test, pred))
print("5-fold CV R2:", cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2").mean())
print(pd.Series(model.coef_, index=features).sort_values(ascending=False))

plt.scatter(y_test, pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("Actual Price"); plt.ylabel("Predicted Price"); plt.title("Actual vs Predicted")
plt.savefig("outputs/actual_vs_predicted.png", dpi=150)
