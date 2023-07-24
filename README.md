# Cardiovascular_Disease_Prediction
## Overview:
This project aims to develop a machine learning model that predicts the likelihood of cardiovascular disease based on various health and clinical factors. Cardiovascular disease (CVD) is a leading cause of morbidity and mortality worldwide, making early detection and prediction crucial for timely interventions and improved patient outcomes.

## Dataset:
All of the dataset values were collected at the moment of medical examination.
You can find the dataset from the Kaggle [here](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

### Features:
- Age | Objective Feature | age | int (days)
- Height | Objective Feature | height | int (cm) 
- Weight | Objective Feature | weight | float (kg) 
- Gender | Objective Feature | gender | categorical code 
- Systolic blood pressure | Examination Feature | ap_hi | int 
- Diastolic blood pressure | Examination Feature | ap_lo | int 
- Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal 
- Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal 
- Smoking | Subjective Feature | smoke | binary 
- Alcohol intake | Subjective Feature | alco | binary 
- Physical activity | Subjective Feature | active | binary 
- Presence or absence of cardiovascular disease | Target Variable | cardio | binary
## Method:
Data Preprocessing: The dataset is cleaned and processed to handle outliers, and data normalization , there is not any missing value into the dataset

Feature Engineering: Relevant features are selected, and new features may be created based on domain knowledge and data analysis for example for Age feature converting days to the years, Creating an attribute for BMI (Body Mass Index) using weight and height and then after removing outliers from ap_hi and ap_lo , Blood pressure feature was added.

Model Selection: Various machine learning algorithms, such as Logistic Regression, Random Forest,xgboost and KNN , are evaluated to identify the best-performing model.

Model Training and Evaluation: The selected model is trained on the data and evaluated using appropriate metrics  such as  precision, recall, and F1 score.
for clinical dataset the recall score is important 
Threshold Adjustment: The model's threshold may be tuned to achieve a desired balance between recall and precision, depending on the specific requirements and implications of false positives and false negatives.
## Result:
The final trained model achieves promising results in predicting cardiovascular disease risk with a high recall of nearly 70 % compared to other works on this dataset in the Kaggle. The model's performance has been validated through cross-validation to ensure its robustness and generalization to new data, and choosing the hyperparameter values through hyperparameter tuning involves finding the optimal values for the hyperparameters of a machine learning model.

