
About the Project : CHATBOT
The GD Car Museum Chatbot is an interactive web-based assistant designed to provide visitors with quick and easy access to information about the Gedee Car Museum. The chatbot allows users to inquire about the museum's history, collections, guided tours, ticket prices, accessibility, and more. It enhances user experience by providing structured responses to frequently asked questions.

Tech Stack Used :
Streamlit: For building the web application and chatbot interface.
Python: The core programming language for the backend logic.
spaCy: For natural language processing and query understanding.
JSON: Used to store and retrieve structured museum data.

Project Summary :
The chatbot leverages spaCy NLP to analyze user queries and fetch relevant information from a pre-defined JSON database containing museum details. The home page provides an overview of the museum along with an image, while the chatbot page enables visitors to ask questions in a user-friendly manner. The chatbot is designed to improve visitor engagement by offering instant responses related to museum facilities, timings, ticket prices, and special exhibits.



About the Project: HAND RECOGNITION
The Hand Detection and Gesture Recognition project is an advanced computer vision application that detects hands, tracks their movement, identifies fingers, and recognizes gestures in real time. This project enables interaction with digital interfaces through hand movements, making it useful for various applications like sign language interpretation, virtual controls, and human-computer interaction.

Tech Stack Used:
Python: Core programming language for implementing the project.
OpenCV: For image processing and real-time hand tracking.
MediaPipe: For efficient hand landmark detection.
NumPy: For numerical computations.
Machine Learning: Used for gesture classification and recognition.

Project Summary:
The project consists of multiple components working together:
Hand Detection: Identifies and localizes hands in a given frame.
Hand Tracking: Continuously follows hand movement in real time.
Finger Detection: Determines the number of raised fingers.
Gesture Recognition: Analyzes finger positions to interpret specific gestures.
Main Application: Integrates all modules to provide a functional interface for gesture-based interaction.
This project enhances interactive experiences by enabling seamless communication between users and digital systems using hand gestures.




PROJECT: PARKINSONS DISEASE PREDICTION
This repository contains a machine learning model to predict the severity of Parkinson's disease based on telemonitoring data. The model is trained using an XGBoost classifier, and it categorizes patients based on the total_UPDRS score.

Features:
Data Preprocessing: Cleans and scales the dataset.
Feature Engineering: Drops unnecessary columns.
Model Training: Uses an XGBoost classifier.
Binary Classification: Converts total_UPDRS into binary labels based on the median.
Model Saving: Saves the trained model and scaler for future use.

Dataset:
The dataset used for training comes from telemonitoring data of Parkinson's disease patients. It includes acoustic and movement-related features to predict disease severity.

Tech Stack Used: 
Python: Main programming language for model development.
Pandas: Data manipulation and preprocessing.
NumPy: Numerical computations.
Scikit-Learn: Data preprocessing (StandardScaler), model evaluation, and train-test splitting.
XGBoost: Gradient boosting model for classification.
Joblib: Model and scaler saving for future use.

