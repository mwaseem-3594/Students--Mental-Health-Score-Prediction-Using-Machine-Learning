# 🎓 Students' Mental Health Score Prediction Using Machine Learning

## 📌 Project Overview

This project focuses on predicting students' **Mental Health Scores** using Machine Learning techniques. The prediction is based on social media usage, demographic information, academic factors, and other relevant features available in the dataset.

The project follows a complete end-to-end Machine Learning workflow, including data cleaning, exploratory data analysis, feature preprocessing, model training, model evaluation, model comparison, feature importance analysis, and model saving.

The objective is to identify patterns in the data and develop a machine learning model capable of accurately predicting students' mental health scores.

---

## 🎯 Project Objectives

The main objectives of this project are to:

* Analyze the dataset and understand the factors related to students' mental health.
* Perform data cleaning and exploratory data analysis.
* Prepare numerical and categorical features for machine learning models.
* Train multiple regression models.
* Evaluate and compare model performance.
* Select the best-performing model.
* Analyze feature importance.
* Save the final trained machine learning pipeline for future predictions.

---

# 📊 Machine Learning Workflow

```text
Data Collection
       ↓
Data Exploration
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature Selection
       ↓
Data Preprocessing
       ↓
Train-Test Split
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Model Comparison
       ↓
Final Model Selection
       ↓
Feature Importance Analysis
       ↓
Model Saving
```

---

# 📂 Dataset

The dataset contains information related to students' social media usage and its potential relationship with mental health.

The target variable used in this project is:

```text
Mental_Health_Score
```

The remaining relevant columns are used as input features for predicting students' mental health scores.

---

# 🧹 Data Cleaning

The dataset was examined to identify potential data quality issues.

The following checks were performed:

* Missing value detection
* Duplicate record detection
* Data type inspection
* Dataset structure analysis

Duplicate records were identified and removed to improve data quality before model training.

---

# 📊 Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the dataset and identify meaningful patterns and relationships between different variables.

The analysis includes:

* Gender distribution
* Academic level distribution
* Most used social media platforms
* Purpose of social media usage
* Stress level distribution
* Mental health score distribution across stress levels
* Distribution of numerical features
* Outlier detection using box plots
* Correlation analysis using a heatmap
* Distribution of categorical features

Data visualizations were created using **Matplotlib** and **Seaborn**.

---

# ⚙️ Data Preprocessing

The dataset contains both numerical and categorical features. Different preprocessing techniques were applied to prepare the data for machine learning models.

## Numerical Features

Numerical features were standardized using:

```python
StandardScaler()
```

## Categorical Features

Categorical features were transformed using:

```python
OneHotEncoder(handle_unknown="ignore")
```

## Preprocessing Pipeline

A `ColumnTransformer` was used to combine numerical and categorical preprocessing steps.

This preprocessing pipeline ensures that the same transformations are consistently applied during both model training and prediction.

---

# ✂️ Train-Test Split

The dataset was divided into training and testing sets.

* **Training Data: 80%**
* **Testing Data: 20%**

The following configuration was used:

```python
test_size=0.2
random_state=42
```

The training dataset was used to train the machine learning models, while the testing dataset was used to evaluate their performance on unseen data.

---

# 🤖 Machine Learning Models

The following regression models were trained and evaluated:

## 1️⃣ Linear Regression

Linear Regression was used as the baseline model to establish an initial performance benchmark.

---

## 2️⃣ Random Forest Regressor

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to improve prediction performance.

---

## 3️⃣ Gradient Boosting Regressor

Gradient Boosting is an ensemble learning algorithm that builds models sequentially, with each new model attempting to improve the errors made by previous models.

---

## 4️⃣ XGBoost Regressor

XGBoost is a gradient boosting algorithm widely used for high-performance machine learning regression tasks.

---

# 📈 Model Evaluation

The models were evaluated using the following regression metrics:

### R² Score

Measures how well the model explains the variation in the target variable. A higher R² score indicates better model performance.

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted values.

### Mean Squared Error (MSE)

Measures the average squared difference between actual and predicted values.

### Root Mean Squared Error (RMSE)

Measures the square root of MSE and provides an interpretable measure of prediction error.

---

# 🏆 Model Performance Comparison

The performance of all trained machine learning models is shown below:

| Model                |     R² Score |          MAE |          MSE |         RMSE |
| -------------------- | -----------: | -----------: | -----------: | -----------: |
| Linear Regression    |     0.789170 |     0.475223 |     0.376376 |     0.613495 |
| 🥇 **Random Forest** | **0.918017** | **0.274497** | **0.146357** | **0.382567** |
| Gradient Boosting    |     0.862654 |     0.376830 |     0.245191 |     0.495168 |
| 🥈 XGBoost           |     0.901797 |     0.306637 |     0.175313 |     0.418704 |

## 📌 Results Summary

Among all tested models, the **Random Forest Regressor achieved the best overall performance**.

### 🥇 Best Model: Random Forest Regressor

**Performance:**

* **R² Score: 0.918017**
* **MAE: 0.274497**
* **MSE: 0.146357**
* **RMSE: 0.382567**

The Random Forest model explained approximately **91.8% of the variation** in the mental health score on the test data, based on the R² metric.

XGBoost was the second-best-performing model with an R² score of **0.901797**.

Based on these results, **Random Forest Regressor was selected as the final model**.

---

# 🔍 Feature Importance Analysis

Feature importance analysis was performed using the Random Forest model.

This analysis helps identify which input features have the greatest influence on the model's predictions.

The top features were extracted using:

```python
feature_importances_
```

A visualization was created to display the **Top 10 Most Important Features** identified by the Random Forest model.

---

# 💾 Saving the Final Model

The final trained model was saved using **Joblib**.

The saved model includes the complete machine learning pipeline:

* Data preprocessing
* Categorical feature encoding
* Numerical feature scaling
* Random Forest regression model

This allows the model to be loaded and used for future predictions without retraining.

```python
import joblib

joblib.dump(
    final_model,
    "Random_ForestRegression_model.pkl"
)
```

---

# 🛠️ Technologies Used

The project was developed using the following technologies and libraries:

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Joblib
* Jupyter Notebook

---

# 📦 Installation

## Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
```

## Navigate to the Project Directory

```bash
cd YOUR_REPOSITORY_NAME
```

## Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
```

---

# 🚀 How to Run the Project

1. Clone this repository.
2. Navigate to the project directory.
3. Install the required dependencies.
4. Ensure the dataset is available in the project directory.
5. Start Jupyter Notebook:

```bash
jupyter notebook
```

6. Open the project notebook.
7. Run all cells sequentially.

---

# 📁 Project Structure

```text
Students-Mental-Health-Prediction/
│
├── Student_Social.ipynb
├── Student Social Media And Mental Health Impact.csv
├── Random_ForestRegression_model.pkl
├── README.md
└── requirements.txt
```

---

# 🎓 Key Learning Outcomes

Through this project, I gained practical experience in:

* Data Cleaning
* Exploratory Data Analysis
* Data Visualization
* Handling Numerical and Categorical Data
* Feature Preprocessing
* Machine Learning Pipelines
* Linear Regression
* Random Forest Regression
* Gradient Boosting Regression
* XGBoost Regression
* Model Evaluation
* Model Comparison
* Feature Importance Analysis
* Saving Machine Learning Models

---

# 🔮 Future Improvements

Possible improvements for this project include:

* Hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`
* Additional feature engineering
* Testing more advanced machine learning models
* Using larger and more diverse datasets
* Cross-validation for more robust model evaluation
* Building an interactive prediction application using Streamlit
* Deploying the machine learning model as a web application

---

# ⭐ Conclusion

This project demonstrates a complete end-to-end machine learning workflow for predicting students' mental health scores.

Four regression models were trained and evaluated:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

Among these models, the **Random Forest Regressor achieved the best performance**, with an **R² score of 0.918017** and an **RMSE of 0.382567**.

The final model demonstrates strong predictive performance on the test dataset and was saved as a complete machine learning pipeline for future predictions.

This project represents a practical application of data analysis and machine learning techniques and is part of my Artificial Intelligence and Machine Learning portfolio.

---

# 👨‍💻 Author

**Muhammad Waseem**

🎓 BS Artificial Intelligence Student
🤖 Machine Learning Enthusiast
🐍 Python Developer

---

⭐ **If you found this project useful, consider giving the repository a star!**
