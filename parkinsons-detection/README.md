Parkinson's Disease Severity Prediction
This repository contains a machine learning model to predict the severity of Parkinson's disease based on telemonitoring data. The model is trained using an XGBoost classifier, and it categorizes patients based on the total_UPDRS score.

Features

Data Preprocessing: Cleans and scales the dataset.

Feature Engineering: Drops unnecessary columns.

Model Training: Uses an XGBoost classifier.

Binary Classification: Converts total_UPDRS into binary labels based on the median.

Model Saving: Saves the trained model and scaler for future use.

 Dataset

The dataset used for training comes from telemonitoring data of Parkinson's disease patients. It includes acoustic and movement-related features to predict disease severity.

Tech Stack Used: 

Python: Main programming language for model development.
Pandas: Data manipulation and preprocessing.
NumPy: Numerical computations.
Scikit-Learn: Data preprocessing (StandardScaler), model evaluation, and train-test splitting.
XGBoost: Gradient boosting model for classification.
Joblib: Model and scaler saving for future use.

