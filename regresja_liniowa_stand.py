import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

df_retu = pd.read_csv('dane.csv', sep=';')

df_retu["Subclass"] = df_retu["Subclass"].replace({
    "JER______t-shirts_s_s": "JER_t-shirts_s_s", 
    "W_dressess": "W_dresses"
})
df_retu["Channel"] = df_retu["Channel"].replace({"ECOMMM": "ECOM"})

df_ret = df_retu[df_retu['Channel'] == 'RETAIL']
print(df_ret)

df_retail = df_ret.fillna(0)

X_test = df_retail[df_retail['Season'] == "AW 2023"]
X_walid = df_retail[df_retail['Season'] == "SS 2023"]
X_train = df_retail[(df_retail['Season'] != "SS 2023") & (df_retail['Season'] != "AW 2023")]

zmienna_zależna = "RETU"
zmienne_niezależne = ["ClearanceWeek", "PriceDiffFromLastRegular", 'Last5DaysSLSU', 'StockCalculationDate']

X_train_data = X_train[zmienne_niezależne]
y_train = X_train[zmienna_zależna]

X_walid_data = X_walid[zmienne_niezależne]
y_walid = X_walid[zmienna_zależna]

X_test_data = X_test[zmienne_niezależne]
y_test = X_test[zmienna_zależna]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_data)
X_walid_scaled = scaler.transform(X_walid_data)
X_test_scaled = scaler.transform(X_test_data)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_walid_pred = model.predict(X_walid_scaled)
y_test_pred = model.predict(X_test_scaled)

print("Wyniki na zbiorze walidacyjnym:")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_walid, y_walid_pred):.2f}")
print(f"Współczynnik R^2: {r2_score(y_walid, y_walid_pred):.2f}")

print("\nWyniki na zbiorze testowym:")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_test_pred):.2f}")
print(f"Współczynnik R^2: {r2_score(y_test, y_test_pred):.2f}")

coefficients = pd.DataFrame({
    "Variable": zmienne_niezależne,
    "Coefficient": model.coef_
})
print("\nWspółczynniki regresji liniowej:")
print(coefficients)

print(f"\nWyraz wolny: {model.intercept_:.2f}")

for feature in zmienne_niezależne:
    plt.figure(figsize=(8, 6))
    sns.regplot(
        x=X_test_data[feature], 
        y=y_test, 
        scatter_kws={"color": "blue", "alpha": 0.6}, 
        line_kws={"color": "red", "linewidth": 2}
    )
    plt.xlabel(feature)
    plt.ylabel("RETU (Zmienna zależna)")
    plt.title(f"Regresja dla zmiennej {feature}")
    plt.grid(True)
    plt.show()