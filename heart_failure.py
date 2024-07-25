#!/usr/bin/env python
# coding: utf-8

# # DSKI ML

# ## Imports

# In[583]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import xgboost as xgb
import shap


# ## Load Data

# In[584]:


# Laden des Datensatzes
file_path = 'data/heart.csv'
df = pd.read_csv(file_path)


# ## Explorative Datenanalyse

# ### Basis Analyse - Datenqualität

# In[585]:


# Überblick über die Daten verschaffen
df.head()
df.info()
df.describe()


# In[586]:


# Überprüfung auf fehlende Werte
print("\nAnzahl fehlender Werte pro Spalte:")
print(df.isnull().sum())

# Überprüfung auf Duplikate
duplikate = df.duplicated().sum()
print(f"\nAnzahl der Duplikate: {duplikate}")

# Verteilung der Zielvariable
print("\nVerteilung der Herzerkrankungen:")
print(df['HeartDisease'].value_counts(normalize=True))


# ### Visualisierung von numerischen Feature

# In[587]:


# Wir filtern die numerischen Spalten heraus
numerische_spalten = df.select_dtypes(include=['float64', 'int64']).columns


# #### Verteilungen der numerischen Deature

# In[588]:


# Histogramme für numerische Variablen mit HeartDisease als Farbe
for column in numerische_spalten:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=column, hue='HeartDisease', bins=30, kde=True, palette='tab10', element="step", alpha=0.5)
    plt.title(f'Histogram of {column} by HeartDisease')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


# #### Boxplots- Ausreißer Analyse
# 
# Da der Nüchternblutzucker(FastingBS) binäre Werte hat macht eine Boxplotdarstellung keinen Sinn (siehe .ReadMe).

# In[589]:


exclude_columns = ['FastingBS', 'HeartDisease']
boxplot_columns = [col for col in numerische_spalten if col not in exclude_columns]

for column in boxplot_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df[column])
    plt.title(f'Boxplot of {column}')
    plt.ylabel(column)
    plt.show()


# In[590]:


# Wir erstellen die Korrelationsmatrix nur für die numerischen Spalten
correlation_matrix = df[numerische_spalten].corr()

# Korrelationsmatrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Korrelationsmatrix der numerischen Variablen')
plt.show()


# ### Visualisierung von kategorischen Feature

# #### Häufigkeitsverteilungen

# In[591]:


# Kategoriale Variablen identifizieren
kategoriale_spalten = df.select_dtypes(include=['object']).columns

# Häufigkeitsverteilungen anzeigen
for column in kategoriale_spalten:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=column, palette='tab10')
    plt.title(f'Häufigkeitsverteilung von {column}')
    plt.xlabel(column)
    plt.ylabel('Anzahl')
    plt.show()


# #### Häufigkeitsverteilung nach der Zielvariable

# In[592]:


# Beziehung zwischen kategorialen Variablen und der Zielvariable
for column in kategoriale_spalten:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=column, hue='HeartDisease', palette='tab10')
    plt.title(f'Relation von {column} mit HeartDisease')
    plt.xlabel(column)
    plt.ylabel('Anzahl')
    plt.show()


# #### Tiefere Analyse der Feature

# In[593]:


# Paarplot der numerischen Variablen
sns.pairplot(df, hue='HeartDisease', palette='tab10')
plt.show()


# In[594]:


# Korrelationsmatrix für kategoriale Variablen
df_encoded = pd.get_dummies(df, drop_first=True)
plt.figure(figsize=(12, 8))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm')
plt.title('Korrelationsmatrix der kodierten Variablen')
plt.show()


# In[595]:


# Altersgruppenanalyse
ziel_variable = df['HeartDisease']

df['Altersgruppe'] = pd.cut(df['Age'], bins=[0, 30, 50, 70, 100], labels=['Jung', 'Mittel', 'Älter', 'Senior'])
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Altersgruppe', hue=ziel_variable)
plt.title('Verteilung der Herzerkrankungen nach Altersgruppe')
plt.show()


# In[596]:


# Analyse des Zusammenhangs zwischen Cholesterin und Blutdruck
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Cholesterol', y='RestingBP', hue=ziel_variable)
plt.title('Zusammenhang zwischen Cholesterin und Ruheblutdruck')
plt.show()


# In[597]:


# Analyse des Zusammenhangs zwischen Alter und maximaler Herzfrequenz
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='MaxHR', hue=ziel_variable)
plt.title('Zusammenhang zwischen Alter und maximaler Herzfrequenz')
plt.show()


# ## Data Preparation 

# ### Datensatz überprüfen

# In[598]:


# Fehlende Werte überprüfen
df.isnull().sum()


# In[599]:


df['Cholesterol'].loc[df['Cholesterol']<=20].sum()


# In[600]:


# Duplikate überprüfen und entfernen
print(df.duplicated().sum())
df = df.drop_duplicates()
print(df.duplicated().sum())


# In[601]:


# Nach Datentypen überprüfen
df.dtypes


# ### Feature Engineering

# We have 2 options for data scaling : 1) Normalization 2) Standardization. As most of the algorithms assume the data to be normally (Gaussian) distributed, Normalization is done for features whose data does not display normal distribution and standardization is carried out for features that are normally distributed where their values are huge or very small as compared to other features.
# Normalization : Oldpeak feature is normalized as it had displayed a right skewed data distribution.
# Standardizarion : Age, RestingBP, Cholesterol and MaxHR features are scaled down because these features are normally distributed.
# 
# 
# https://www.kaggle.com/code/tanmay111999/heart-failure-prediction-cv-score-90-5-models#Feature-Engineering

# #### Altersgruppen bilden

# In[602]:


# Feature Erstellung und Transformation
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle-aged', 'Senior'])


# #### Gruppen für Olpeak

# In[603]:


# Oldpeak in Bins einteilen
bins = [-1, 0, 2, 4, float('inf')]
labels = ['Normal', 'Mild', 'Moderate', 'Severe']
df['OldpeakBinned'] = pd.cut(df['Oldpeak'], bins=bins, labels=labels)


# #### Auswahl der Features

# In[604]:


categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'AgeGroup', 'OldpeakBinned']
numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']


# #### Einstellung des Preprocessing

# In[605]:


# Numerische Features aufteilen
features_to_normalize = ['Oldpeak']
features_to_standardize = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']
other_numeric_features = ['FastingBS']

# Pipeline für zu normalisierende Features
normalize_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('normalizer', MinMaxScaler())
])

# Pipeline für zu standardisierende Features
standardize_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline für andere numerische Features
other_numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# Preprocessing Pipeline für kategorische Daten (unverändert)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Kombination der Preprocessing Schritte
preprocessor = ColumnTransformer(
    transformers=[
        ('normalize', normalize_transformer, features_to_normalize),
        ('standardize', standardize_transformer, features_to_standardize),
        ('other_num', other_numeric_transformer, other_numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# #### Anwendung des preprocessing für Übersicht und Debugging

# In[606]:


# Anwenden des Preprocessings
X = preprocessor.fit_transform(df)

# Erstellen eines neuen DataFrames mit den transformierten Daten
numeric_feature_names = features_to_normalize + features_to_standardize + other_numeric_features
categorical_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features).tolist()
feature_names = numeric_feature_names + categorical_feature_names

df_transformed = pd.DataFrame(X, columns=feature_names)

# Hinzufügen der Zielvariable
df_transformed['HeartDisease'] = df['HeartDisease']


# In[607]:


df_transformed.head()


# #### Preprocessed Features Analyse

# In[608]:


# Berechnung der Korrelationen
corr = df_transformed.corrwith(df_transformed['HeartDisease']).sort_values(ascending=False).to_frame()
corr.columns = ['Correlations']

# Visualisierung
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.4, linecolor='black')
plt.title('Korrelation zu HeartDisease')
plt.tight_layout()
plt.show()


# ### Trainings- und Testdatensätze

# In[609]:


# Aufteilen der Daten in Trainings- und Testdatensätze
X = df_transformed.drop('HeartDisease', axis=1)
y = df_transformed['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Modelle
# ### Definition

# In[626]:


# Modelle definieren
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(probability=True),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': xgb.XGBClassifier()
}


# ### Hyperparameter 

# In[627]:


param_dists = {
    'Logistic Regression': {
        'C': np.logspace(-4, 4, 20),
        'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag']
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Support Vector Machine': {
        'C': np.logspace(-3, 3, 7),
        'gamma': np.logspace(-3, 3, 7),
        'kernel': ['rbf', 'poly', 'sigmoid']
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_child_weight': [1, 3, 5]
    }
}


# ### Random und Grid Search

# In[628]:


def random_grid_search_cv(models, param_dists, X_train, y_train, n_splits=5, n_iter=30):
    best_estimators = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"Processing {name}...")
        
        # Berechne die Gesamtzahl der Parameterkombinationen
        n_combinations = np.prod([len(v) for v in param_dists[name].values()])
        
        # Passe n_iter an, wenn nötig
        actual_n_iter = min(n_iter, n_combinations)
        
        # Randomized Search mit Cross-Validation
        random_search = RandomizedSearchCV(model, param_dists[name], n_iter=actual_n_iter, cv=kf, n_jobs=-1, random_state=42, scoring='accuracy')
        random_search.fit(X_train, y_train)
        
        # Grid Search mit den besten Parametern aus Randomized Search
        best_params = random_search.best_params_
        refined_params = {k: [v] if not isinstance(v, list) else v for k, v in best_params.items()}
        grid_search = GridSearchCV(model, refined_params, cv=kf, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        best_estimators[name] = grid_search.best_estimator_
        
        # Cross-Validation Score
        cv_scores = cross_val_score(best_estimators[name], X_train, y_train, cv=kf, scoring='accuracy')
        
        print(f"Best Model: {name}")
        print("Best Parameters:", grid_search.best_params_)
        print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print("\n")
    
    return best_estimators


# ### Training

# In[629]:


best_models = random_grid_search_cv(models, param_dists, X_train, y_train)


# ## Evaluation

# In[630]:


def plot_confusion_matrix_plotly(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    
    # Erstellen der Heatmap mit Plotly
    fig = px.imshow(cm,
                    labels=dict(x="Predicted label", y="True label"),
                    x=['Negative', 'Positive'],
                    y=['Negative', 'Positive'],
                    title=f'Confusion Matrix - {model_name}',
                    color_continuous_scale='Blues',
                    text_auto=True)
    
    fig.update_layout(width=600, height=500)
    return fig


# In[631]:


def plot_model_comparison_plotly(accuracies):
    # Erstellen des Balkendiagramms mit Plotly
    fig = go.Figure(data=[go.Bar(
        x=list(accuracies.keys()),
        y=list(accuracies.values()),
        text=[f'{acc:.2f}' for acc in accuracies.values()],
        textposition='auto',
    )])
    
    fig.update_layout(
        title='Model Comparison - Test Accuracy',
        xaxis_title='Models',
        yaxis_title='Accuracy',
        yaxis_range=[0, 1],  # Setzt die y-Achse von 0 bis 1 für Genauigkeiten
        width=800,
        height=500
    )
    
    return fig


# In[632]:


def plot_roc_curves(models, X_test, y_test):
    fig = go.Figure()
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)
        
        fig.add_trace(go.Scatter(x=fpr, y=tpr,
                                 mode='lines',
                                 name=f'{name} (AUC = {auc_score:.3f})'))
    
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                             mode='lines',
                             name='Random Classifier',
                             line=dict(dash='dash', color='grey')))
    
    fig.update_layout(title='ROC Curves',
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate',
                      legend_title='Models',
                      width=800, height=600)
    
    return fig



# In[633]:


def plot_model_comparison_plotly(accuracies, auc_scores):
    fig = go.Figure()
    
    # Accuracy bars
    fig.add_trace(go.Bar(
        x=list(accuracies.keys()),
        y=list(accuracies.values()),
        name='Accuracy',
        text=[f'{acc:.3f}' for acc in accuracies.values()],
        textposition='auto',
    ))
    
    # AUC bars
    fig.add_trace(go.Bar(
        x=list(auc_scores.keys()),
        y=list(auc_scores.values()),
        name='AUC',
        text=[f'{auc:.3f}' for auc in auc_scores.values()],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Model Comparison - Accuracy and AUC',
        xaxis_title='Models',
        yaxis_title='Score',
        barmode='group',
        yaxis_range=[0, 1],  # Setzt die y-Achse von 0 bis 1 für Scores
        width=900,
        height=500
    )
    
    return fig


# ### Konfusion Matrizen und Klassifikations Report

# In[634]:


def plot_feature_importance(model, X, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print(f"Feature importance not available for {model_name}")
        return None
    
    feature_imp = pd.DataFrame({'feature': X.columns, 'importance': importances})
    feature_imp = feature_imp.sort_values('importance', ascending=False)
    
    fig = px.bar(feature_imp, x='importance', y='feature', orientation='h',
                 title=f'Feature Importance - {model_name}')
    return fig

def plot_shap_summary(model, X, model_name):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type="bar")
    plt.title(f'SHAP Summary - {model_name}')
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(X):
    corr = X.corr()
    
    fig = px.imshow(corr, 
                    labels=dict(color="Correlation"),
                    x=corr.columns,
                    y=corr.columns,
                    title="Feature Correlation Matrix",
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1)
    
    fig.update_layout(width=800, height=800)
    return fig

def plot_pdp(model, X, feature_name, model_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    display = PartialDependenceDisplay.from_estimator(model, X, [feature_name], ax=ax)
    ax.set_title(f'Partial Dependence Plot - {feature_name} ({model_name})')
    plt.tight_layout()
    plt.show()


# In[635]:


# Finale Evaluation und Visualisierungen
accuracies = {}
auc_scores = {}
for name, model in best_models.items():
    print(f"\nFinal Evaluation for {name}:")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    
    # Konfusionsmatrix
    cm_fig = plot_confusion_matrix_plotly(y_test, y_pred, name)
    cm_fig.show()
    
    # Feature Importance
    fi_fig = plot_feature_importance(model, X_train, name)
    if fi_fig:
        fi_fig.show()
    
    # Partial Dependence Plots (für die Top 3 Features)
    if hasattr(model, 'feature_importances_'):
        top_features = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
        top_features = top_features.sort_values('importance', ascending=False).head(3)
        for feature in top_features['feature']:
            plot_pdp(model, X_train, feature, name)
    
    accuracies[name] = model.score(X_test, y_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_scores[name] = auc(fpr, tpr)


# ### ROC

# In[639]:


# ROC Curves
roc_fig = plot_roc_curves(best_models, X_test, y_test)
roc_fig.show()



# #### Accuracy und AUC

# In[640]:


# Gesamtvergleich aller Modelle (Accuracy und AUC)
comparison_fig = plot_model_comparison_plotly(accuracies, auc_scores)
comparison_fig.show()



# ### Korrelationsmatrix

# In[641]:


# Korrelationsmatrix
corr_fig = plot_correlation_matrix(X_train)
corr_fig.show()


# In[ ]:


### Shap Werte
