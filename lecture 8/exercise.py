#Task 1: Data Preparation
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.datasets import load_breast_cancer
from scipy.stats import uniform, loguniform
import optuna


#Load the breast cancer dataset.
data = load_breast_cancer()
X, y = data.data, data.target

#print(f"Dataset shape: {X.shape}")
#print(f"Number of samples: {X.shape[0]}")
#print(f"Number of features: {X.shape[1]}")
print(f"Class distribution: {np.bincount(y)}")
print(f"Class names: {data.target_names}")
#print(f"Feature names: {data.feature_names}")  # First 5 features

# Plot class distribution
class_counts = np.bincount(y)
plt.figure(figsize=(8, 6))
plt.bar(data.target_names, class_counts, color=['red', 'blue'], edgecolor='black')
plt.title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
plt.xlabel('Class', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.grid(axis='y', alpha=0.3)
for i, count in enumerate(class_counts):
    plt.text(i, count + 5, str(count), ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()



print(f"Missing values in X: {np.isnan(X).sum()}")
print(f"Missing values in y: {np.isnan(y).sum()}")
def datasplit(X, y, test_size=0.2, random_state=42, use_stratify=True):
    """Create training and test splits.
    
    Args:
        use_stratify: If True, maintains class distribution in splits. If False, random split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size,
        random_state=random_state,
        stratify=y if use_stratify else None  # Stratify only if use_stratify=True
    )
    return X_train, X_test, y_train, y_test


#Scale the features using an appropriate scaler (e.g., StandardScaler).
def scaling(X_train, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit on training data
    X_test_scaled = scaler.transform(X_test)  # Only transform test data (no fit!)
    return X_train_scaled, X_test_scaled


# Hyperparameter Optimization Functions

def grid_search_optimization(X_train_scaled, y_train):
    """Grid Search for Logistic Regression"""
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(max_iter=10000, random_state=42),
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=1
    )
    
    start_time = time.time()
    grid_search.fit(X_train_scaled, y_train)
    duration = time.time() - start_time
    
    # Extract CV scores for the best estimator
    best_index = grid_search.best_index_
    cv_scores = np.array([grid_search.cv_results_[f'split{i}_test_score'][best_index] for i in range(5)])
    
    return grid_search.best_params_, grid_search.best_score_, duration, grid_search.best_estimator_, cv_scores


def random_search_optimization(X_train_scaled, y_train):
    """Random Search for Logistic Regression"""
    param_distributions = {
        'C': loguniform(0.001, 100),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    
    random_search = RandomizedSearchCV(
        LogisticRegression(max_iter=10000, random_state=42),
        param_distributions,
        n_iter=30,  # Number of parameter settings sampled
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=1,
        random_state=42
    )
    
    start_time = time.time()
    random_search.fit(X_train_scaled, y_train)
    duration = time.time() - start_time
    
    # Extract CV scores for the best estimator
    best_index = random_search.best_index_
    cv_scores = np.array([random_search.cv_results_[f'split{i}_test_score'][best_index] for i in range(5)])
    
    return random_search.best_params_, random_search.best_score_, duration, random_search.best_estimator_, cv_scores


def objective(trial, X_train_scaled, y_train):
    """Objective function for Bayesian Optimization"""
    params = {
        'C': trial.suggest_loguniform('C', 0.001, 100),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
        'solver': trial.suggest_categorical('solver', ['liblinear', 'saga'])
    }
    
    model = LogisticRegression(max_iter=10000, random_state=42, **params)
    score = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    return score.mean()


def bayesian_optimization(X_train_scaled, y_train):
    """Bayesian Optimization for Logistic Regression using Optuna"""
    start_time = time.time()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X_train_scaled, y_train), n_trials=30, show_progress_bar=True)
    duration = time.time() - start_time
    
    # Train final model with best parameters and get CV scores
    best_model = LogisticRegression(max_iter=10000, random_state=42, **study.best_params)
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    best_model.fit(X_train_scaled, y_train)
    
    return study.best_params, study.best_value, duration, best_model, cv_scores


def evaluate_and_visualize(best_model, best_method_name, best_cv_score, best_cv_scores, X_test_scaled, y_test, 
                           basic_cv_mean, grid_score, random_score, bayesian_score,
                           grid_time, random_time, bayesian_time, data,
                           basic_cv_scores, grid_cv_scores, random_cv_scores, bayesian_cv_scores):
    """Evaluate the best model and create visualizations comparing all methods."""
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    
    # Make predictions with the winning model
    print(f"\nUsing {best_method_name} for final predictions...")
    y_pred_best = best_model.predict(X_test_scaled)
    
    # Calculate evaluation metrics
    accuracy_best = accuracy_score(y_test, y_pred_best)
    
    print(f"\n{best_method_name} Test Set Accuracy: {accuracy_best:.4f} ({accuracy_best*100:.2f}%)")
    print(f"CV Score (training): {best_cv_score:.4f}")
    print(f"Difference (CV - Test): {(best_cv_score - accuracy_best)*100:.2f}%")
    
    print("\nClassification Report (Best Model):")
    print(classification_report(y_test, y_pred_best, target_names=data.target_names))
    
    conf_matrix = confusion_matrix(y_test, y_pred_best)
    print(f"\nConfusion Matrix ({best_method_name}):")
    print(conf_matrix)
    
    # Create visualizations
    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=data.target_names, 
                yticklabels=data.target_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {best_method_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # 2. CV Scores Comparison - All Methods
    plt.figure(figsize=(10, 6))
    methods = ['Basic\nModel', 'Grid\nSearch', 'Random\nSearch', 'Bayesian\nOptimization']
    cv_scores_all = [basic_cv_mean, grid_score, random_score, bayesian_score]
    colors = ['lightgray', 'skyblue', 'lightcoral', 'lightgreen']
    
    # Highlight the best method
    colors_updated = []
    for i, method in enumerate(['Basic Model', 'Grid Search', 'Random Search', 'Bayesian Optimization']):
        if method.replace('\n', ' ') == best_method_name.replace('\n', ' ') or \
           method.split()[0] in best_method_name:
            colors_updated.append('gold')
        else:
            colors_updated.append(colors[i])
    
    bars = plt.bar(methods, cv_scores_all, color=colors_updated, edgecolor='black', width=0.6)
    plt.title('Cross-Validation Scores Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('CV Accuracy', fontsize=12)
    plt.ylim([min(cv_scores_all) - 0.02, max(cv_scores_all) + 0.02])
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, cv_scores_all):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                 f'{score:.4f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 3. Optimization Methods Comparison: CV Score vs Time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    opt_methods = ['Grid\nSearch', 'Random\nSearch', 'Bayesian\nOpt']
    cv_scores = [grid_score, random_score, bayesian_score]
    times = [grid_time, random_time, bayesian_time]
    
    # CV Scores
    ax1.bar(opt_methods, cv_scores, color=['skyblue', 'lightcoral', 'lightgreen'], edgecolor='black')
    ax1.set_title('Cross-Validation Scores', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    for i, (method, score) in enumerate(zip(opt_methods, cv_scores)):
        ax1.text(i, score + 0.001, f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Computation Time
    ax2.bar(opt_methods, times, color=['skyblue', 'lightcoral', 'lightgreen'], edgecolor='black')
    ax2.set_title('Computation Time', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time (seconds)', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    for i, (method, t) in enumerate(zip(opt_methods, times)):
        ax2.text(i, t + 0.1, f'{t:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 4. Boxplot of CV scores for all methods
    plt.figure(figsize=(10, 6))
    cv_results = {
        'Basic Model': basic_cv_scores,
        'Grid Search': grid_cv_scores,
        'Random Search': random_cv_scores,
        'Bayesian Opt': bayesian_cv_scores
    }
    plt.boxplot(cv_results.values(), labels=cv_results.keys())
    plt.title('Comparison of Cross-validation Methods', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print mean scores
    print(f"\n{'='*70}")
    print("CV Scores Summary (5 folds each):")
    print(f"{'='*70}")
    for method, scores in cv_results.items():
        print(f"{method:20s}: Mean = {scores.mean():.4f} (+/- {scores.std():.4f})")


#Task 2: Model Selection and Training
def main():
    with_or_without_stratify = input("Enable stratification? (True/False): ")  # Set to False to disable stratification
    if with_or_without_stratify.lower() == 'true':
        use_stratify = True
    else:
        use_stratify = False
    # Call the datasplit function
    print(f"stratify={use_stratify}...")
    X_train, X_test, y_train, y_test = datasplit(X, y, use_stratify=use_stratify)
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    # Call the scaling function
    X_train_scaled, X_test_scaled = scaling(X_train, X_test)
    print(f"\nData has been scaled!")
    print(f"Training data shape: {X_train_scaled.shape}")
    print(f"Test data shape: {X_test_scaled.shape}")

    #Create a Logistic Regression model.
    model = LogisticRegression(max_iter=10000)

    # Evaluate the basic model with cross-validation (don't train yet)
    print("\n" + "="*70)
    print("Basic Model - Cross-Validation")
    print("="*70)

    # Perform k-fold cross-validation
    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    basic_cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring='accuracy')
    basic_cv_mean = basic_cv_scores.mean()

    print(f"Cross-Validation Scores (5 folds): {basic_cv_scores}")
    print(f"Mean CV Score: {basic_cv_mean:.4f} (+/- {basic_cv_scores.std():.4f})")

    # NOTE: Basic model will be trained AFTER comparing with optimized models

    #Implement all three hyperparameter optimization methods
    print("\n" + "="*70)
    print("HYPERPARAMETER OPTIMIZATION COMPARISON")
    print("="*70)

    # 1. Grid Search
    print("\n[1/3] Running Grid Search...")
    print("-" * 70)
    grid_params, grid_score, grid_time, grid_model, grid_cv_scores = grid_search_optimization(X_train_scaled, y_train)

    # 2. Random Search
    print("\n[2/3] Running Random Search...")
    print("-" * 70)
    random_params, random_score, random_time, random_model, random_cv_scores = random_search_optimization(X_train_scaled, y_train)

    # 3. Bayesian Optimization
    print("\n[3/3] Running Bayesian Optimization...")
    print("-" * 70)
    bayesian_params, bayesian_score, bayesian_time, bayesian_model, bayesian_cv_scores = bayesian_optimization(X_train_scaled, y_train)

    #Output the best parameters found by each method
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)

    results_df = pd.DataFrame({
        'Method': ['Basic Model', 'Grid Search', 'Random Search', 'Bayesian Optimization'],
        'Best CV Score': [basic_cv_mean, grid_score, random_score, bayesian_score],
        'Time (seconds)': [0.0, grid_time, random_time, bayesian_time]
    })

    print("\n" + results_df.to_string(index=False))

    print("\n" + "-"*70)
    print("Best Parameters Found:")
    print("-"*70)
    print(f"\nBasic Model:          Default parameters (C=1.0, penalty='l2', solver='lbfgs')")
    print(f"Grid Search:          {grid_params}")
    print(f"Random Search:        {random_params}")
    print(f"Bayesian Optimization: {bayesian_params}")

    # Select the best overall model (highest CV score) - including basic model
    best_methods = [('Basic Model', model, basic_cv_mean, basic_cv_scores),
                    ('Grid Search', grid_model, grid_score, grid_cv_scores), 
                    ('Random Search', random_model, random_score, random_cv_scores),
                    ('Bayesian Optimization', bayesian_model, bayesian_score, bayesian_cv_scores)]
    best_method_name, best_model, best_cv_score, best_cv_scores = max(best_methods, key=lambda x: x[2])

    print(f"\n{'='*70}")
    print(f"WINNER: {best_method_name} with CV Score: {best_cv_score:.4f}")
    print(f"{'='*70}")

    # NOW train the winning model on full training set for test predictions
    best_model.fit(X_train_scaled, y_train)
    print(f"\n{best_method_name} trained on full training set!")

    #Task 3: Model Evaluation

    #Task 3: Model Evaluation
    print(f"\n{'='*70}")
    print("MODEL EVALUATION ON TEST SET")
    print(f"{'='*70}")

    # Call the evaluation and visualization function
    evaluate_and_visualize(
        best_model, best_method_name, best_cv_score, best_cv_scores,
        X_test_scaled, y_test,
        basic_cv_mean, grid_score, random_score, bayesian_score,
        grid_time, random_time, bayesian_time, data,
        basic_cv_scores, grid_cv_scores, random_cv_scores, bayesian_cv_scores
    )


if __name__ == "__main__":
    main()