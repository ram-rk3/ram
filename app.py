import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# TITLE
# ----------------------------
st.title("🚗 Vehicle Service Prediction")

# ----------------------------
# YOUR ORIGINAL CODE (UNCHANGED)
# ----------------------------
df = pd.read_csv("vehicle_service_dataset.csv")

le = LabelEncoder()
df['Driving_Type'] = le.fit_transform(df['Driving_Type'])
df['Oil_Quality'] = le.fit_transform(df['Oil_Quality'])

X = df.drop(['Vehicle_ID', 'Remaining_Days_To_Service'], axis=1)
y = df['Remaining_Days_To_Service']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
)
rf_model.fit(X_train, y_train)

# ----------------------------
# USER INPUT (ADDED)
# ----------------------------
st.header("Enter Vehicle Details")

vehicle_age = st.number_input("Vehicle Age", 0)
mileage = st.number_input("Mileage Since Last Service", 0)
days_since_service = st.number_input("Days Since Last Service", 0)
engine_hours = st.number_input("Engine Hours", 0)

driving_type = st.selectbox(
    "Driving Type",
    ["City", "Highway", "Mixed"]
)

oil_quality = st.selectbox(
    "Oil Quality",
    ["Poor", "Moderate", "Good"]
)

brake_wear = st.number_input("Brake Wear Level", 0)
previous_breakdowns = st.number_input("Previous Breakdowns", 0)

# ----------------------------
# PREDICTION (ADDED)
# ----------------------------
if st.button("Predict"):
    driving_encoded = le.transform([driving_type])[0]
    oil_encoded = le.transform([oil_quality])[0]

    user_input = np.array([[vehicle_age,
                             mileage,
                             days_since_service,
                             engine_hours,
                             driving_encoded,
                             oil_encoded,
                             brake_wear,
                             previous_breakdowns]])

    result = rf_model.predict(user_input)

    st.success(f"Remaining Days To Service: {int(result[0])}")
