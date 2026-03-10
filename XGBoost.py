import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

df_retu = pd.read_csv('dane.csv', sep=';')

df_retu["Subclass"] = df_retu["Subclass"].replace({
    "JER______t-shirts_s_s": "JER_t-shirts_s_s", 
    "W_dressess": "W_dresses"
})
df_retu["Channel"] = df_retu["Channel"].replace({"ECOMMM": "ECOM"})

df_retail = df_retu[df_retu['Channel'] == 'RETAIL']

object_columns = df_retail.select_dtypes(include=['object']).columns

df_retail_enc = df_retail.copy()

label_encoders = {}
for col in object_columns:
    le = LabelEncoder()
    df_retail_enc[col] = le.fit_transform(df_retail[col])
    label_encoders[col] = le

for col, le in label_encoders.items():
    print(f"Kolumna: {col}")
    print(dict(zip(le.classes_, range(len(le.classes_)))))
    print()
    
print(df_retail_enc.dtypes)

X_test = df_retail_enc[df_retail_enc['Season'] == 2]
X_walid = df_retail_enc[df_retail_enc['Season'] == 5]
X_train = df_retail_enc[(df_retail_enc['Season'] != 5) & (df_retail_enc['Season'] != 2)]

print(X_test.shape)
print(X_walid.shape)
print(X_train.shape)

zmienna_zależna = "RETU"
zmienne_niezależne = ["ClearanceWeek", 'Last4DaysSLSU', 'Last5DaysSLSU', "Subclass"]

X_train_data = X_train[zmienne_niezależne]
y_train = X_train[zmienna_zależna]

X_walid_data = X_walid[zmienne_niezależne]
y_walid = X_walid[zmienna_zależna]

X_test_data = X_test[zmienne_niezależne]
y_test = X_test[zmienna_zależna]

xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
xgb_model.fit(X_train_data, y_train)
preds_xgb = xgb_model.predict(X_walid_data)

degree = 2
polyreg_model = Pipeline([
    ('poly_features', PolynomialFeatures(degree=degree)),
    ('lin_reg', LinearRegression())
])

polyreg_model.fit(X_train_data, y_train)

preds_poly = polyreg_model.predict(X_walid_data)

wyniki = pd.DataFrame(
    data={
        'RSS': [
            np.sum(np.square(preds_poly - y_walid)),
            np.sum(np.square(preds_xgb - y_walid))
        ],
        'R2': [
            r2_score(y_walid, preds_poly),
            r2_score(y_walid, preds_xgb)
        ],
        'MAE': [
            mean_absolute_error(y_walid, preds_poly),
            mean_absolute_error(y_walid, preds_xgb)
        ]
    },
    index=['PolyReg', 'XGBoost']
)

print(wyniki)

xgb_model2 = xgb.XGBRegressor(
    objective='reg:squarederror',
    learning_rate=0.021, 
    n_estimators=100, 
    max_depth=5
)
xgb_model2.fit(X_train_data, y_train)
preds_xgb2 = xgb_model2.predict(X_walid_data)

wyniki2 = pd.DataFrame(
    data={
        'RSS': [
            np.sum(np.square(preds_poly - y_walid)),
            np.sum(np.square(preds_xgb2 - y_walid))
        ],
        'R2': [
            r2_score(y_walid, preds_poly),
            r2_score(y_walid, preds_xgb2)
        ],
        'MAE': [
            mean_absolute_error(y_walid, preds_poly),
            mean_absolute_error(y_walid, preds_xgb2)
        ]
    },
    index=['PolyReg', 'XGBoost']
)

print(wyniki2)
