🫁 Lung Cancer Survey Analysis & Prediction
📌 Overview

This project analyzes a Lung Cancer Survey Dataset and builds a machine learning model to predict whether a person has lung cancer or not.
It includes:

Data Cleaning & Preprocessing

Exploratory Data Analysis (EDA) using plots

Correlation Heatmap

Machine Learning Models (Random Forest, optional SVM)

Model Evaluation (Accuracy & Confusion Matrix)

📂 Dataset

File: survey lung cancer.csv

Contains medical survey data with attributes such as:

GENDER (M/F)

Symptoms (e.g., COUGHING, FATIGUE, etc.)

LUNG_CANCER (YES/NO → Target variable)

⚙️ Steps in the Code

Import Libraries

pandas, numpy for data handling

seaborn, matplotlib for visualization

sklearn for machine learning

Load Data

df = pd.read_csv("C:/Users/katam/Desktop/survey lung cancer.csv")


Data Cleaning

Remove duplicates

Encode categorical values:

GENDER: M → 1, F → 0

LUNG_CANCER: YES → 1, NO → 0

Exploratory Data Analysis (EDA)

Dataset summary with .info(), .describe()

Visualizations:

Histograms of features

Pairplot for relationships

Countplot of gender distribution with percentages

Correlation heatmap

Feature & Target Selection

X = df.iloc[:, :-1]      # Features  
y = df['LUNG_CANCER']    # Target  


Train-Test Split

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2, random_state=123
)


Model Training (Random Forest)

model = RandomForestClassifier()
model.fit(X_train, y_train)


Prediction & Evaluation

Accuracy Score

Confusion Matrix visualization

📊 Output Examples

Histograms & Pairplots showing feature distributions.

Correlation Heatmap to check relationships.

Confusion Matrix to evaluate classification performance.

Accuracy score printed in console.

🚀 How to Run

Clone this repo / download the project files.

Place survey lung cancer.csv in the specified path.

Install required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn


Run the Python script:

python lung_cancer_analysis.py

🔮 Future Work

Add Support Vector Machine (SVM) classifier for comparison.

Try hyperparameter tuning with GridSearchCV.

Deploy as a web app (Streamlit/Flask).
