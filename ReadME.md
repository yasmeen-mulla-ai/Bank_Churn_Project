
Student Name: Yasmeen Mulla

Project: Bank Customer Churn Prediction

# Bank Customer Churn Prediction

## Problem Statement

Customer churn is a major challenge for banks because losing customers directly impacts revenue and business growth. The objective of this project is to build a Machine Learning model that predicts whether a customer is likely to leave the bank based on demographic and account-related information.

---

## Dataset Description

Dataset Name: Churn_Modelling.csv

Total Records: 10,000

Features:

* CreditScore
* Geography
* Gender
* Age
* Tenure
* Balance
* NumOfProducts
* HasCrCard
* IsActiveMember
* EstimatedSalary

Target Variable:

* Exited

  * 1 = Customer Left Bank
  * 0 = Customer Stayed

Removed Columns:

* RowNumber
* CustomerId
* Surname

---

## Methodology

### 1. Data Cleaning

* Checked missing values
* Removed duplicate records
* Removed unnecessary columns
* Performed consistency checks

### 2. Feature Engineering

Created new features:

1. BalanceSalaryRatio
2. ProductsPerTenure
3. AgeGroup

Applied:

* One Hot Encoding
* Standard Scaling

### 3. Data Preprocessing

* Train-Test Split (80:20)
* Encoding
* Scaling

### 4. Model Training

Three Machine Learning Models were trained:

1. Logistic Regression
2. Decision Tree
3. Random Forest

### 5. Model Evaluation

Metrics Used:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

### 6. Deployment

* Saved model using Joblib
* Built Streamlit Web Application

---

## Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | ~81%     |
| Decision Tree       | ~79%     |
| Random Forest       | ~86%     |

Best Model:

Random Forest Classifier

Reason:

Highest accuracy and better overall performance.

---

## Screenshots


### Streamlit Home Page

![alt text](<Screenshot 2026-06-23 225903.png>) ![alt text](<Screenshot 2026-06-23 225823.png>)

### Prediction Output

![alt text](<Screenshot 2026-06-23 225236.png>) ![alt text](<Screenshot 2026-06-23 225315.png>) 
![alt text](<Screenshot (58).png>) ![alt text](<Screenshot (59).png>) ![alt text](<Screenshot (60).png>)
---

## How to Run

### Install Dependencies

pip install -r requirements.txt

### Train Model

python train_model.py

### Run Streamlit Application

streamlit run app.py

### Open Browser

http://localhost:8501

---

## Project Structure

Bank_Churn_Project/

├── Churn_Modelling.csv

├── train_model.py

├── app.py

├── model.pkl

├── preprocessor.pkl

├── requirements.txt

└── README.md

---


Technology Stack:

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Joblib
