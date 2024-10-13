# Income-Classification-Project
This project aims to predict whether an individual's income exceeds $50K/year based on the UCI Income Census dataset using various machine learning models. The project involves data preprocessing, feature engineering, model selection, hyperparameter tuning, and evaluation with advanced metrics like AUC-ROC and LIME explainability.

## Table of Contents
- Project Overview
- Dataset
- Project Workflow
- Models Implemented
- Results
- Evaluation Metrics
- How to Run
- Technologies Used
- Conclusion

## Project Overview
This project classifies individuals into income categories (above or below $50K/year). The key objectives include:
- Building, tuning, and evaluating machine learning models to maximize classification accuracy.
- Handling data imbalance using SMOTE.
- Providing model interpretability using LIME.

## Dataset
The dataset used is the UCI Income Census dataset, which contains demographic and employment-related information. The target variable is the income category, split into two classes: <=50K and >50K.
Dataset link: https://archive.ics.uci.edu/dataset/20/census+income

## Project Workflow
### 1. Data Preprocessing:
- Cleaned the dataset by handling missing values and duplicates.
- Encoded categorical features using LabelEncoder.
- Scaled numerical features using StandardScaler.

### 2. Feature Engineering:
- Created new features such as education_hours and age_bin for enhanced model performance.

### 3. Modeling:
- Split data into train-test sets.
- Applied SMOTE to address class imbalance.
- Implemented Random Forest, SVM, XGBoost, Gradient Boosting, and Logistic Regression models.
- Hyperparameter tuning with GridSearchCV and cross-validation.

### 4. Evaluation:
- Assessed model performance using accuracy, precision, recall, F1-score, MCC, AUC-ROC, and confusion matrix.
- Visualized key metrics using AUC-ROC and Precision-Recall curves.

### 5. Explainability:
- Used LIME values to interpret the models and visualize feature importance.

## Models Implemented
- Random Forest Classifier
- XGBoost Classifier
- SVM
- Gradient Boosting Classifier
- Logistic Regression

## Results 
<img width="715" alt="Screenshot 2024-10-13 at 10 17 17 AM" src="https://github.com/user-attachments/assets/1189598d-4f0d-4ff2-80a4-0cdcd21021f6">

## Evaluation Metrics
- Accuracy: Percentage of correct predictions.
- Recall: Ability to identify positive cases (income >50K).
- Precision: Proportion of positive predictions that were correct.
- F1-Score: Harmonic mean of precision and recall.
- AUC-ROC: Measures model's ability to distinguish between classes.
- MCC (Matthews Correlation Coefficient): Balanced measure even with imbalanced classes.

## How to Run
1. Clone the repository:
`git clone https://github.com/yourusername/income-classification-project.git`

2. Install dependencies:
`pip install -r requirements.txt`

3. Run the script:
`python income_classification.py`

## Technologies Used
- Programming Language: Python
- Libraries: pandas, numpy, scikit-learn, XGBoost, imbalanced-learn, matplotlib, seaborn, LIME, joblib
- Tools: Jupyter Notebook, Git

## Conclusion
The XGBoost and Gradient Boosting models yielded the highest accuracy, with SHAP providing insights into model explainability. The project demonstrates the full machine learning pipeline, from data preprocessing to model interpretability.

