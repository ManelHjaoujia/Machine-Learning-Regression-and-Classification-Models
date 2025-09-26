# Machine Learning: Regression & Classification Models

## Repository Title & Description  
**Title**: Machine-Learning-Regression-and-Classification-Models  
**Description**: This repository contains implementations of various supervised machine learning algorithms for both regression and classification tasks. It covers full pipelines: data preprocessing, model training, hyperparameter tuning, evaluation, and performance comparisons using different datasets and models.

---

## Project Structure / Notebooks Overview

This repository includes several Jupyter notebooks, each exploring a particular model or task:

| Notebook | Task Type | Models / Focus |
|---|---|---|
| `SimpleLinearRegressionModel.ipynb` | Regression | Simple Linear Regression |
| `MultipleLinearRegression.ipynb` | Regression | Multiple Linear Regression |
| `KNN.ipynb` | Classification / Regression | K-Nearest Neighbors |
| `DecisionTree.ipynb` | Classification / Regression | Decision Tree |
| `LogisticRegression.ipynb` | Classification | Logistic Regression |
| `NaiveBayes.ipynb` | Classification | Naive Bayes |
| `SVM.ipynb` | Classification | Support Vector Machine |

Each notebook typically follows this workflow:  
1. **Data loading & exploration**  
2. **Data cleaning / handling missing values**  
3. **Feature engineering / preprocessing**  
4. **Model training & evaluation**  
5. **Hyperparameter tuning** (quand appliqué)  
6. **Comparison / insights**

---

## Common Workflow Across Notebooks

### 1. Data Cleaning  
- Identifying and handling missing values  
- Dropping unnecessary or irrelevant columns  
- Correcting inconsistent categories or typos  

### 2. Feature Engineering & Preprocessing  
- Encoding categorical variables (OneHot, LabelEncoding)  
- Scaling numerical features (StandardScaler, etc.)  
- Possibly creating interaction features or derived features  

### 3. Model Training & Comparison  
- Training multiple algorithms (e.g. Linear Regression, Decision Tree, KNN, SVM, Naive Bayes, Random Forest, etc.)  
- Evaluating using relevant metrics (accuracy, precision, recall, F1 for classification; MAE, RMSE, R² for regression)  

### 4. Hyperparameter Tuning  
- Use `RandomizedSearchCV` or `GridSearchCV` to find optimal hyperparameters  
- Retrain best models with tuned parameters  

### 5. Evaluation & Insights  
- Compare models via metrics  
- Plot results (e.g. ROC curves for classification, residual plots for regression)  
- Discuss findings, pitfalls, and possible improvements  

---

## Dependencies

Below are the core libraries required to run the notebooks:

- Python 3.8+  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- plotly  
- scikit-learn  

You can install them with:

```bash
pip install -r requirements.txt
```

## Key Learnings & Skills Demonstrated

- Proficiency in **data preprocessing**: encoding, scaling, missing value handling  
- Ability to **compare multiple models** and understand their trade-offs  
- Use of appropriate **evaluation metrics** (classification & regression)  
- Experience with **hyperparameter tuning** using `RandomizedSearchCV` / `GridSearchCV`  
- Understanding of **ensemble techniques** (Random Forest, boosting, etc.)  
- Exposure to **end-to-end ML workflows** — from raw data to model insights  

## Author
**Manel Hjaoujia**  
Junior Data Scientist 
Passionate about Data Science & Machine Learning  



