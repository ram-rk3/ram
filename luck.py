import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("vehicle_service_dataset.csv")

# Encode categorical columns
le = LabelEncoder()
df['Driving_Type'] = le.fit_transform(df['Driving_Type'])
df['Oil_Quality'] = le.fit_transform(df['Oil_Quality'])

# Features and target
X = df.drop(['Vehicle_ID', 'Remaining_Days_To_Service'], axis=1)
y = df['Remaining_Days_To_Service']  

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict
y_pred = rf_model.predict(X_test)

# Evaluate
print("Model Evaluation Metrics:\n")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# Sample predictions
predictions_df = pd.DataFrame({
    'Actual': y_test.reset_index(drop=True),
    'Predicted': y_pred
})
print("\nSample Predictions vs Actual:")
print(predictions_df.head())
