🫁 Prediction of Lung Cancer using Machine Learning
🔍 Project Overview

This project develops a machine learning system to predict lung cancer based on patient health parameters and lifestyle indicators. The goal is to enable early detection, improve treatment planning, and reduce mortality rates.
The best trained model in this project achieved high classification accuracy, demonstrating the potential of AI-assisted healthcare screening.

📂 Dataset

You can download the dataset from:
👉 Lung Cancer Dataset

📑 Sample Data Structure
{
  "Age": [45, 60, ...],                # Age of the patient
  "Gender": ["Male", "Female", ...],   # Gender of the patient
  "Smoking": [1, 0, ...],              # Smoking status (1 = Yes, 0 = No)
  "Yellow_Fingers": [1, 0, ...],       # Symptom indicator
  "Anxiety": [1, 0, ...],              # Anxiety status
  "Peer_Pressure": [0, 1, ...],        # Social factor
  "Chronic_Disease": [0, 1, ...],      # Presence of chronic illness
  "Fatigue": [1, 0, ...],              # Fatigue level
  "Wheezing": [1, 0, ...],             # Symptom indicator
  "Coughing": [1, 0, ...],             # Symptom indicator
  "Lung_Cancer": ["YES", "NO", ...]    # Target variable
}

⚙️ Requirements

Install required libraries before running the project:

pip install pandas scikit-learn matplotlib seaborn

🚀 Steps to Run

Clone or download this repository.

Download and place the dataset in the project folder.

Run the training script:

python lung_cancer_prediction.py

📊 Workflow

Data Loading & Exploration

Import dataset with pandas

Handle missing values & encode categorical data

Preprocessing

Convert categorical variables (e.g., Gender, Yes/No) into numeric

Normalize features if required

Model Training

Algorithms used: Logistic Regression, Decision Tree, Random Forest, etc.

Data split: 80% training / 20% testing

Evaluation

Accuracy Score

Confusion Matrix

Classification Report

Visualization

Heatmaps for feature correlation

Confusion matrix plot for classification performance

✅ Output

Prediction: Probability of lung cancer (YES/NO)

Performance Metrics:

Accuracy: XX%

Precision, Recall, F1-score

Graphs:

Confusion Matrix Heatmap

Feature importance plots

🌟 Impact

Early and accurate prediction of lung cancer can:

Improve survival rates

Support doctors in diagnosis

Reduce healthcare costs by enabling timely treatment
