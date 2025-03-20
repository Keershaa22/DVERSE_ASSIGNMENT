import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib  # To save the model and scaler

# Load dataset
def load_data():
    file_path = "telemonitoring_parkinsons_updrs.data.csv"  # Replace with your file path
    data = pd.read_csv(file_path)
    return data

# Train and save the model
def train_and_save_model():
    # Load data
    data = load_data()

    # Identify the target column
    target_col = 'total_UPDRS'

    # Convert target to binary classes
    threshold = data[target_col].median()
    data[target_col] = (data[target_col] > threshold).astype(int)

    # Prepare the data for modeling
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Drop non-feature columns
    if 'subject#' in X.columns:
        X = X.drop(columns=['subject#'])
    if 'test_time' in X.columns:
        X = X.drop(columns=['test_time'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        seed=42
    )
    model.fit(X_train_scaled, y_train)

    # Save the model and scaler
    joblib.dump(model, "parkinsons_model.pkl")
    joblib.dump(scaler, "parkinsons_scaler.pkl")
    print("Model and scaler saved successfully.")

if __name__ == "__main__":
    train_and_save_model()