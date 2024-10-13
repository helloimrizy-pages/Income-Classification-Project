
# **Income Classification Project**
This project predicts whether an individual's income exceeds **$50K/year** using the UCI Income Census dataset. The project involves **data preprocessing**, **feature engineering**, **model selection**, **hyperparameter tuning**, and **evaluation** using advanced metrics like **AUC-ROC** and **LIME** for explainability.

---

## **Table of Contents**
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Evaluation Metrics](#evaluation-metrics)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)
- [Conclusion](#conclusion)

---

## **Project Overview**
This project classifies individuals into income categories (above or below $50K/year). Key objectives include:
- Building, tuning, and evaluating machine learning models for **classification accuracy**.
- Handling **class imbalance** using **SMOTE**.
- Providing model interpretability with **LIME**.

---

## **Dataset**
The dataset is sourced from the [UCI Income Census Dataset](https://archive.ics.uci.edu/dataset/20/census+income). It contains demographic and employment-related information. The target variable is the income category, split into two classes: <=50K and >50K.

---

## **Project Workflow**
### **1. Data Preprocessing:**
- Cleaned the dataset by handling missing values and duplicates.
- Encoded categorical features using **LabelEncoder**.
- Scaled numerical features using **StandardScaler**.

### **2. Feature Engineering:**
- Created new features such as **education_hours** and **age_bin** for enhanced model performance.

### **3. Modeling:**
- Split data into train-test sets.
- Applied **SMOTE** to address class imbalance.
- Implemented models: **Random Forest**, **SVM**, **XGBoost**, **Gradient Boosting**, and **Logistic Regression**.
- Performed **hyperparameter tuning** with **GridSearchCV** and cross-validation.

### **4. Evaluation:**
- Assessed model performance using **accuracy**, **precision**, **recall**, **F1-score**, **MCC**, **AUC-ROC**, and **confusion matrix**.
- Visualized key metrics using **AUC-ROC** and **Precision-Recall** curves.

### **5. Explainability:**
- Used **LIME** values to interpret the models and visualize feature importance.

---

## **Models Implemented**
- **Random Forest Classifier**
- **XGBoost Classifier**
- **SVM**
- **Gradient Boosting Classifier**
- **Logistic Regression**

---

## **Results**
![Model Performance Comparison](https://github.com/user-attachments/assets/1189598d-4f0d-4ff2-80a4-0cdcd21021f6)

---

## **Evaluation Metrics**
- **Accuracy**: Percentage of correct predictions.
- **Recall**: Ability to identify positive cases (income >50K).
- **Precision**: Proportion of positive predictions that were correct.
- **F1-Score**: Harmonic mean of precision and recall.
- **AUC-ROC**: Measures model's ability to distinguish between classes.
- **MCC (Matthews Correlation Coefficient)**: Balanced measure even with imbalanced classes.

---

## **How to Run**
1. **Clone the repository:**
   ```
   git clone https://github.com/helloimrizy-pages/Income-Classification-Project.git
   ```

2. **Install dependencies:**
   Create a virtual environment (optional but recommended):
   ```
   # Create a virtual environment (optional)
   python -m venv env

   # Activate the virtual environment (on Windows)
   env\Scriptsctivate

   # Activate the virtual environment (on macOS/Linux)
   source env/bin/activate
   ```

   Then, install the dependencies:
   ```
   pip install -r requirements.txt
   ```

3. **Open the Jupyter Notebook:**
   ```
   jupyter notebook
   ```

4. Navigate to the `.ipynb` file (e.g., income_predicted_model.ipynb) and run the cells.

---

## **Technologies Used**
- **Programming Language**: Python
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `XGBoost`, `imbalanced-learn`, `matplotlib`, `seaborn`, `LIME`, `joblib`
- **Tools**: Jupyter Notebook, Git

---

## **Conclusion**
The **XGBoost** and **Gradient Boosting** models yielded the highest accuracy, with **LIME** providing insights into model explainability. The project demonstrates the full machine learning pipeline, from data preprocessing to model interpretability.
