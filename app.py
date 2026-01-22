import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("vehicle_service_dataset.csv")


le = LabelEncoder()
df['Driving_Type'] = le.fit_transform(df['Driving_Type'])
df['Oil_Quality'] = le.fit_transform(df['Oil_Quality'])

X = df.drop(['Vehicle_ID', 'Remaining_Days_To_Service'], axis=1)
y = df['Remaining_Days_To_Service']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
Predicted= y_pred

