# SVM workflow refactored into functions (no automatic execution)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# 1) Data loading
def load_data(url=None):
    if url is None:
        url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
    df = pd.read_csv(url)
    return df

# 2) Target and feature preparation
def prepare_target_and_features(df, target_col='Churn', drop_cols=True):
    df = df.copy()
    # Normalize target text, make robust to capitalization/whitespace
    df[target_col] = df[target_col].astype(str).str.strip().str.title()
    y = (df[target_col] == 'Yes').astype(int)
    
    # Count class distribution
    count_0 = (y == 0).sum()
    count_1 = (y == 1).sum()
    print(f"Class 0 (No Churn): {count_0}, Class 1 (Churn): {count_1}")
    
    if drop_cols is True:
        drop_cols = ['customerID', target_col]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)
    return X, y

# 3) Train/test split
def split_data(X, y, test_size=0.2, random_state=42, stratify=True):
    """Split features and target into train and test sets.

    Returns X_train, X_test, y_train, y_test
    """
    strat = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)

# 4) Build preprocessor
def build_preprocessor(X_train):
    """Construct a ColumnTransformer for numeric and categorical preprocessing.

    Returns: preprocessor
    """
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    return preprocessor

# 5) Build model pipelines
def build_pipelines(preprocessor, linear_C=1.0, rbf_C=1.0, rbf_gamma='scale'):
    """Return (linear_pipeline, rbf_pipeline) that include preprocessing."""
    linear_pipeline = Pipeline(steps=[
        ('preproc', preprocessor),
        ('clf', LinearSVC(C=linear_C, random_state=42, max_iter=10000))
    ])

    rbf_pipeline = Pipeline(steps=[
        ('preproc', preprocessor),
        ('clf', SVC(kernel='rbf', C=rbf_C, gamma=rbf_gamma, random_state=42))
    ])

    return linear_pipeline, rbf_pipeline

# 6) Train a pipeline
def train_pipeline(pipeline, X_train, y_train):
    """Fit pipeline on training data and return fitted pipeline."""
    pipeline.fit(X_train, y_train)
    return pipeline

# 7) Evaluate a fitted pipeline
def evaluate_pipeline(pipeline, X_test, y_test):
    """Predict and return accuracy, report, and confusion matrix."""
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, report, cm, y_pred 
# 8) Plot two confusion matrices side-by-side
def plot_confusion_matrices(cm1, cm2, titles=('Model 1', 'Model 2')):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot first confusion matrix
    im1 = axes[0].imshow(cm1, cmap='Blues')
    axes[0].set_title(titles[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    plt.colorbar(im1, ax=axes[0])
    
    # Add text annotations for cm1
    for i in range(cm1.shape[0]):
        for j in range(cm1.shape[1]):
            axes[0].text(j, i, str(cm1[i, j]), 
                        ha='center', va='center', color='white' if cm1[i, j] > cm1.max()/2 else 'black')
    
    # Plot second confusion matrix
    im2 = axes[1].imshow(cm2, cmap='Greens')
    axes[1].set_title(titles[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    plt.colorbar(im2, ax=axes[1])
    
    # Add text annotations for cm2
    for i in range(cm2.shape[0]):
        for j in range(cm2.shape[1]):
            axes[1].text(j, i, str(cm2[i, j]), 
                        ha='center', va='center', color='white' if cm2[i, j] > cm2.max()/2 else 'black')
    
    plt.tight_layout()
    plt.show()

def main():
    df = load_data()
    X, y = prepare_target_and_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    preproc = build_preprocessor(X_train)
    lin_pipe, rbf_pipe = build_pipelines(preproc)
    fitted_lin = train_pipeline(lin_pipe, X_train, y_train)
    fitted_rbf = train_pipeline(rbf_pipe, X_train, y_train)
    acc_lin, report_lin, cm_lin, y_pred_lin = evaluate_pipeline(fitted_lin, X_test, y_test)
    acc_rbf, report_rbf, cm_rbf, y_pred_rbf = evaluate_pipeline(fitted_rbf, X_test, y_test) 
    print("Linear SVM Classification Report:\n", report_lin)
    print("RBF SVM Classification Report:\n", report_rbf)
    plot_confusion_matrices(cm_lin, cm_rbf, ('Linear SVM', 'RBF SVM'))


if __name__ == "__main__":
    main()