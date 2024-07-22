#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
df = pd.read_csv('data/heart.csv')


# In[9]:


import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np

# Laden des Datensatzes
file_path = 'data/heart.csv'
df = pd.read_csv(file_path)

# Kategorische und numerische Merkmale identifizieren
categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']

# Preprocessing Pipeline für numerische Daten
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Preprocessing Pipeline für kategorische Daten
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Kombination der Preprocessing Schritte
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Modelle definieren
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Aufteilen der Daten in Trainings- und Testdatensätze
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning für Logistic Regression
param_dist_lr = {
    'classifier__C': np.logspace(-4, 4, 20),
    'classifier__solver': ['lbfgs', 'liblinear']
}

# Hyperparameter Tuning für Random Forest
param_dist_rf = {
    'classifier__n_estimators': [50, 100, 200, 300, 400],
    'classifier__max_depth': [None, 10, 20, 30, 40],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Hyperparameter Tuning für Support Vector Machine
param_dist_svm = {
    'classifier__C': np.logspace(-4, 4, 20),
    'classifier__gamma': np.logspace(-4, 4, 20),
    'classifier__kernel': ['rbf', 'poly', 'sigmoid']
}

# Hyperparameter Tuning für Gradient Boosting
param_dist_gb = {
    'classifier__n_estimators': [50, 100, 200, 300],
    'classifier__learning_rate': [0.01, 0.1, 0.2, 0.3],
    'classifier__max_depth': [3, 4, 5, 6]
}

# Hyperparameter Random Dist definieren
param_dists = {
    'Logistic Regression': param_dist_lr,
    'Random Forest': param_dist_rf,
    'Support Vector Machine': param_dist_svm,
    'Gradient Boosting': param_dist_gb
}

# Funktion für Random Search und Grid Search
def random_and_grid_search(models, param_dists, X_train, y_train, X_test, y_test):
    best_estimators = {}
    for name, model in models.items():
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        
        # Randomized Search
        random_search = RandomizedSearchCV(clf, param_dists[name], n_iter=50, cv=5, n_jobs=-1, random_state=42)
        random_search.fit(X_train, y_train)
        
        # Grid Search mit den besten Parametern aus Randomized Search
        best_params = random_search.best_params_
        refined_params = {k: [v] for k, v in best_params.items()}
        grid_search = GridSearchCV(clf, refined_params, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_estimators[name] = grid_search.best_estimator_
        y_pred = best_estimators[name].predict(X_test)
        
        print(f"Best Model: {name}")
        print("Best Parameters from Randomized Search:", best_params)
        print("Best Parameters from Grid Search:", grid_search.best_params_)
        print(classification_report(y_test, y_pred))
        print("\n")
        
    return best_estimators

# Ausführung des Hyperparameter-Tunings und der Evaluation
best_models = random_and_grid_search(models, param_dists, X_train, y_train, X_test, y_test)

best_models


# In[ ]:




