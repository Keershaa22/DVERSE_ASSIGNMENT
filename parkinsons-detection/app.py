import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import librosa
import pyaudio
import wave
import os

# Load the model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("parkinsons_model.pkl")
    scaler = joblib.load("parkinsons_scaler.pkl")
    return model, scaler

# Load dataset for evaluation
@st.cache_data
def load_data():
    file_path = "telemonitoring_parkinsons_updrs.data.csv"  # Replace with your file path
    data = pd.read_csv(file_path)
    return data

# Evaluate the model
def evaluate_model(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    return accuracy, conf_matrix, class_report, fpr, tpr, roc_auc

# Function to record audio
def record_audio(filename, duration=5, sample_rate=16000):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=1024)
    frames = []

    st.write("Recording...")
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    st.write("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

# Function to extract features from audio
def extract_features(filename):
    y, sr = librosa.load(filename, sr=None)
    features = {}

    # Extract jitter
    jitter = librosa.feature.rms(y=y)[0].mean()
    features['Jitter(%)'] = jitter

    # Extract shimmer
    shimmer = librosa.feature.zero_crossing_rate(y)[0].mean()
    features['Shimmer(dB)'] = shimmer

    # Extract HNR (Harmonic-to-Noise Ratio)
    hnr = librosa.effects.harmonic(y).mean()
    features['HNR'] = hnr

    return features

# Streamlit App
def main():
    st.set_page_config(page_title="Parkinson's Disease Prediction", layout="wide")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Prediction", "Model Evaluation", "Data Exploration", "Voice Analysis"])

    # Load the model and scaler
    model, scaler = load_model_and_scaler()

    # Load data for evaluation
    data = load_data()
    target_col = 'total_UPDRS'
    threshold = data[target_col].median()
    data[target_col] = (data[target_col] > threshold).astype(int)
    X = data.drop(columns=[target_col, 'subject#', 'test_time'])
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Home Page
    if page == "Home":
        st.title("Welcome to the Parkinson's Disease Prediction App")
        st.write("""
        ### What is Parkinson's Disease?
        Parkinson's disease is a progressive neurological disorder that affects movement. 
        It develops gradually, sometimes starting with a barely noticeable tremor in just one hand. 
        While tremors are common, the disorder also commonly causes stiffness or slowing of movement.

        ### About This App
        This application uses machine learning to predict whether a patient has Parkinson's disease 
        based on specific features extracted from voice recordings and other clinical measurements. 
        The model is trained using the XGBoost algorithm and provides accurate predictions along with 
        detailed insights into the model's performance.
        """)
        st.image("parkinsons_image.jpg", use_container_width=True)  # Updated to use_container_width
        st.write("### Features:")
        st.write("- **Prediction Page**: Input patient data and get predictions.")
        st.write("- **Model Evaluation**: View model accuracy, confusion matrix, and other metrics.")
        st.write("- **Data Exploration**: Explore the dataset interactively.")
        st.write("- **Voice Analysis**: Record your voice and analyze it for Parkinson's symptoms.")

    # Prediction Page
    elif page == "Prediction":
        st.title("Parkinson's Disease Prediction")
        st.write("Input patient data to predict whether they have Parkinson's disease.")

        # Input form for essential features
        st.sidebar.header("Input Features for Prediction")
        feature_columns = [
            'age', 'motor_UPDRS', 'Jitter(%)', 'Shimmer(dB)', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'
        ]
        input_data = {}
        for col in feature_columns:
            if col == 'age':
                input_data[col] = st.sidebar.slider(f"Enter {col}", min_value=5.0, max_value=80.0, value=10.0)
            elif col == 'motor_UPDRS':
                input_data[col] = st.sidebar.slider(f"Enter {col}", min_value=0.0, max_value=50.0, value=10.0)
            elif col == 'Jitter(%)':
                input_data[col] = st.sidebar.slider(f"Enter {col}", min_value=0.0, max_value=2.0, value=0.5)
            elif col == 'Shimmer(dB)':
                input_data[col] = st.sidebar.slider(f"Enter {col}", min_value=0.0, max_value=1.0, value=0.2)
            elif col == 'NHR':
                input_data[col] = st.sidebar.slider(f"Enter {col}", min_value=0.0, max_value=1.0, value=0.1)
            elif col == 'HNR':
                input_data[col] = st.sidebar.slider(f"Enter {col}", min_value=0.0, max_value=40.0, value=20.0)
            elif col == 'RPDE':
                input_data[col] = st.sidebar.slider(f"Enter {col}", min_value=0.0, max_value=1.0, value=0.5)
            elif col == 'DFA':
                input_data[col] = st.sidebar.slider(f"Enter {col}", min_value=0.0, max_value=1.0, value=0.5)
            elif col == 'PPE':
                input_data[col] = st.sidebar.slider(f"Enter {col}", min_value=0.0, max_value=1.0, value=0.5)

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Add missing columns and set their values to 0
        missing_columns = set(X.columns) - set(input_df.columns)
        for col in missing_columns:
            input_df[col] = 0.0

        # Ensure the columns are in the correct order
        input_df = input_df[X.columns]

        # Filter the input_df to include only the feature_columns
        input_df_filtered = input_df[feature_columns]

        # Display input data
        st.write("### Input Data for Prediction")
        st.write(input_df_filtered)

        # Make predictions
        if st.button("Predict"):
            # Scale the input data
            input_scaled = scaler.transform(input_df)

            # Make predictions
            predictions = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)[:, 1]
            prediction_labels = ["Healthy" if pred == 0 else "Parkinson's" for pred in predictions]

            # Display predictions
            st.write("### Prediction Results")
            st.write(f"Prediction: **{prediction_labels[0]}**")
            st.write(f"Probability of Parkinson's Disease: **{prediction_proba[0]:.2%}**")

            # Severity Analysis
            st.write("### Severity Analysis")
            if prediction_labels[0] == "Parkinson's":
                if prediction_proba[0] < 0.5:
                    severity = "Mild"
                elif 0.5 <= prediction_proba[0] < 0.8:
                    severity = "Moderate"
                else:
                    severity = "Severe"
                st.write(f"Severity: **{severity}**")

                # Progress Bar for Severity
                st.write("#### Severity Progress Bar")
                progress_value = float(prediction_proba[0])
                st.progress(progress_value)
                st.write(f"Severity Level: {severity} (Probability: {prediction_proba[0]:.2%})")

                # Gauge Chart for Severity
                st.write("#### Severity Gauge Chart")
                fig = px.pie(
                    values=[progress_value, 1 - progress_value],
                    names=[severity, "Healthy"],
                    hole=0.6,
                    title="Severity Gauge"
                )
                fig.update_traces(textinfo='none')
                st.plotly_chart(fig, use_container_width=True)

            # Probability Distribution Chart
            st.write("### Probability Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(prediction_proba, kde=True, bins=20, ax=ax)
            ax.set_xlabel("Probability of Parkinson's Disease")
            ax.set_ylabel("Frequency")
            st.pyplot(fig, use_container_width=True)

    # Model Evaluation Page
    elif page == "Model Evaluation":
        st.title("Model Evaluation")
        st.write("### Model Performance Metrics")

        # Evaluate the model
        accuracy, conf_matrix, class_report, fpr, tpr, roc_auc = evaluate_model(model, scaler, X_test, y_test)

        # Display accuracy
        st.write(f"**Accuracy**: {accuracy:.2%}")

        # Confusion Matrix
        st.write("### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig, use_container_width=True)

        # Classification Report
        st.write("### Classification Report")
        st.table(pd.DataFrame(class_report).transpose())

        # ROC Curve
        st.write("### ROC Curve")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig, use_container_width=True)

    # Data Exploration Page
    elif page == "Data Exploration":
        st.title("Data Exploration")
        st.write("Explore the dataset interactively.")

        # Feature Distribution Plots
        st.write("### Feature Distribution Plots")
        feature_to_plot = st.selectbox("Select a feature to plot", X.columns)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(X[feature_to_plot], kde=True, ax=ax)
        ax.set_title(f"Distribution of {feature_to_plot}")
        st.pyplot(fig, use_container_width=True)

        # Correlation Heatmap
        st.write("### Correlation Heatmap")
        corr = X.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig, use_container_width=True)

        # Pair Plots
        st.write("### Pair Plots")
        pair_plot_features = st.multiselect("Select features for pair plot", X.columns, default=X.columns[:3])
        if len(pair_plot_features) > 1:
            fig = sns.pairplot(X[pair_plot_features])
            st.pyplot(fig, use_container_width=True)
        else:
            st.warning("Please select at least two features for pair plot.")

    # Voice Analysis Page
    elif page == "Voice Analysis":
        st.title("Voice Analysis for Parkinson's Disease")
        st.write("Record your voice and analyze it for Parkinson's symptoms.")

        # Record audio
        audio_filename = "recorded_audio.wav"
        if st.button("Start Recording"):
            record_audio(audio_filename, duration=5)
            st.audio(audio_filename, format="audio/wav")

        # Analyze audio
        if st.button("Analyze Voice"):
            if os.path.exists(audio_filename):
                # Extract features
                features = extract_features(audio_filename)
                st.write("### Extracted Features")
                st.write(features)

                # Convert features to DataFrame
                input_df = pd.DataFrame([features])

                # Add missing columns and set their values to 0
                missing_columns = set(X.columns) - set(input_df.columns)
                for col in missing_columns:
                    input_df[col] = 0.0

                # Ensure the columns are in the correct order
                input_df = input_df[X.columns]

                # Make predictions
                input_scaled = scaler.transform(input_df)
                predictions = model.predict(input_scaled)
                prediction_proba = model.predict_proba(input_scaled)[:, 1]
                prediction_labels = ["Healthy" if pred == 0 else "Parkinson's" for pred in predictions]

                # Display predictions
                st.write("### Prediction Results")
                st.write(f"Prediction: **{prediction_labels[0]}**")
                st.write(f"Probability of Parkinson's Disease: **{prediction_proba[0]:.2%}**")
            else:
                st.error("Please record your voice first.")

# Run the app
if __name__ == "__main__":
    main()