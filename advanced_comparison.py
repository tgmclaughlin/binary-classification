import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures
from imblearn.over_sampling import SMOTE
import time
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from dataset import load_pima_dataset
from metrics import binary_classification_metrics, confusion_matrix_stats, print_classification_report


def add_polynomial_features(X_train, X_test, degree=2):
    """
    Add polynomial features to the dataset.

    Args:
        X_train: Training features
        X_test: Test features
        degree: Degree of polynomial features

    Returns:
        Tuple of (X_train_poly, X_test_poly) with polynomial features
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    return X_train_poly, X_test_poly


def create_interaction_terms(df):
    """
    Create interaction terms between important features.

    Args:
        df: DataFrame with features

    Returns:
        DataFrame with added interaction features
    """
    # Make a copy to avoid modifying the original
    df_new = df.copy()

    # Create interaction between glucose and BMI (two key diabetes indicators)
    df_new['glucose_bmi'] = df_new['glucose'] * df_new['bmi']

    # Create interaction between age and diabetes pedigree
    df_new['age_pedigree'] = df_new['age'] * df_new['diabetes_pedigree']

    # Create ratio of glucose to insulin (insulin sensitivity indicator)
    df_new['glucose_insulin_ratio'] = df_new['glucose'] / (df_new['insulin'] + 1)  # Avoid div by zero

    # Age group as numeric values first, then create interaction
    df_new['age_group'] = pd.cut(df_new['age'], bins=[0, 30, 45, 100], labels=[0, 1, 2])
    df_new['age_group'] = df_new['age_group'].astype(int)  # Convert to integer before multiplication
    df_new['age_group_bmi'] = df_new['age_group'] * df_new['bmi']

    # Create BMI category feature
    df_new['bmi_category'] = pd.cut(df_new['bmi'],
                                    bins=[0, 18.5, 25, 30, 100],
                                    labels=[0, 1, 2, 3])  # Underweight, Normal, Overweight, Obese
    df_new['bmi_category'] = df_new['bmi_category'].astype(int)  # Convert to integer

    return df_new


def plot_roc_curves(models, X_test_orig, X_test_eng, X_test_poly, y_test):
    """
    Plot ROC curves for multiple models.

    Args:
        models: Dictionary of model name to model object
        X_test_orig: Original test features
        X_test_eng: Engineered test features
        X_test_poly: Polynomial test features
        y_test: Test labels
    """
    plt.figure(figsize=(10, 8))

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            # Select appropriate test set based on model name
            if "Feature Engineering" in name:
                X_test = X_test_eng
            elif "Poly Features" in name:
                X_test = X_test_poly
            else:
                X_test = X_test_orig

            if name == "Custom Logistic Regression":
                y_scores = model.predict_proba(X_test)
            else:
                y_scores = model.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC: {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('roc_curve.png')
    plt.close()


def plot_precision_recall_curves(models, X_test_orig, X_test_eng, X_test_poly, y_test):
    """
    Plot precision-recall curves for multiple models.

    Args:
        models: Dictionary of model name to model object
        X_test_orig: Original test features
        X_test_eng: Engineered test features
        X_test_poly: Polynomial test features
        y_test: Test labels
    """
    plt.figure(figsize=(10, 8))

    # Plot iso-f1 curves
    f1_scores = np.linspace(0.2, 0.8, num=4)
    for f1_score in f1_scores:
        x = np.linspace(0.01, 1)
        y = f1_score * x / (2 * x - f1_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate(f'f1={f1_score:0.1f}', xy=(0.9, y[45]), alpha=0.8)

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            # Select appropriate test set based on model name
            if "Feature Engineering" in name:
                X_test = X_test_eng
            elif "Poly Features" in name:
                X_test = X_test_poly
            else:
                X_test = X_test_orig

            if name == "Custom Logistic Regression":
                y_scores = model.predict_proba(X_test)
            else:
                y_scores = model.predict_proba(X_test)[:, 1]

            precision, recall, _ = precision_recall_curve(y_test, y_scores)
            avg_precision = average_precision_score(y_test, y_scores)
            plt.step(recall, precision, lw=2, where='post',
                     label=f'{name} (AP: {avg_precision:.3f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('precision_recall_curve_advanced.png')
    plt.close()


def plot_feature_importance(feature_names, importances, title):
    """
    Plot feature importance.

    Args:
        feature_names: List of feature names
        importances: Array of feature importance values
        title: Plot title
    """
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.close()


def tune_model_hyperparameters(model, param_grid, X_train, y_train):
    """
    Perform grid search to find optimal hyperparameters.

    Args:
        model: Model to tune
        param_grid: Dictionary of parameter grids
        X_train: Training features
        y_train: Training labels

    Returns:
        Best model found by grid search
    """
    # Use stratified k-fold to maintain class distribution
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Grid search with F1 score as the optimization metric
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1
    )

    # Fit the grid search
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Train and evaluate a model, printing metrics.

    Args:
        model: Model object with fit and predict methods
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        model_name: Name of the model for printing

    Returns:
        Tuple of (trained model, metrics dict, confusion dict)
    """
    print(f"\n{'='*20} {model_name} {'='*20}")

    # Training time
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.4f} seconds")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = binary_classification_metrics(y_test, y_pred)
    confusion = confusion_matrix_stats(y_test, y_pred)

    # Print classification report
    print_classification_report(metrics, confusion)

    return model, metrics, confusion


def main():
    """
    Main function to run advanced model comparison.
    """
    print("\n" + "="*30)
    print("ADVANCED MODEL COMPARISON")
    print("="*30)

    print("\nLoading and preprocessing Pima Indians Diabetes dataset...")
    df = load_pima_dataset()

    # Basic dataset info
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution: \n{df['outcome'].value_counts(normalize=True).round(4)}")

    # Create feature engineered dataset
    print("\nApplying feature engineering...")
    df_engineered = create_interaction_terms(df)

    # Original features
    X_orig = df.drop('outcome', axis=1).values
    y = df['outcome'].values
    feature_names_orig = df.drop('outcome', axis=1).columns

    # Engineered features
    X_eng = df_engineered.drop('outcome', axis=1).values
    feature_names_eng = df_engineered.drop('outcome', axis=1).columns

    # Create train and test sets
    X_orig_train, X_orig_test, y_train, y_test = train_test_split(
        X_orig, y, test_size=0.2, random_state=42, stratify=y
    )

    X_eng_train, X_eng_test = train_test_split(
        X_eng, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler_orig = StandardScaler()
    X_orig_train_scaled = scaler_orig.fit_transform(X_orig_train)
    X_orig_test_scaled = scaler_orig.transform(X_orig_test)

    scaler_eng = StandardScaler()
    X_eng_train_scaled = scaler_eng.fit_transform(X_eng_train)
    X_eng_test_scaled = scaler_eng.transform(X_eng_test)

    # Apply SMOTE to handle class imbalance
    print("\nApplying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_orig_train_smote, y_train_smote = smote.fit_resample(X_orig_train_scaled, y_train)
    X_eng_train_smote, y_eng_train_smote = smote.fit_resample(X_eng_train_scaled, y_train)

    # Generate polynomial features
    print("\nGenerating polynomial features...")
    X_poly_train, X_poly_test = add_polynomial_features(X_orig_train_scaled, X_orig_test_scaled, degree=2)

    print(f"\nTraining set: {X_orig_train.shape[0]} samples")
    print(f"Test set: {X_orig_test.shape[0]} samples")
    print(f"After SMOTE: {X_orig_train_smote.shape[0]} samples")
    print(f"Engineered features: {X_eng_train.shape[1]}")
    print(f"Polynomial features: {X_poly_train.shape[1]}")

    # Create and tune models
    print("\nTuning model hyperparameters...")

    # LogisticRegression with grid search
    lr_params = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'class_weight': [None, 'balanced']
    }
    lr = LogisticRegression(random_state=42, max_iter=1000)
    best_lr = tune_model_hyperparameters(lr, lr_params, X_orig_train_scaled, y_train)

    # RandomForest with grid search
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced', 'balanced_subsample']
    }
    rf = RandomForestClassifier(random_state=42)
    best_rf = tune_model_hyperparameters(rf, rf_params, X_orig_train_scaled, y_train)

    # GradientBoosting with grid search
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'min_samples_split': [2, 5]
    }
    gb = GradientBoostingClassifier(random_state=42)
    best_gb = tune_model_hyperparameters(gb, gb_params, X_orig_train_scaled, y_train)

    # Create model dictionary
    models = {
        "Tuned Logistic Regression": best_lr,
        "Tuned Random Forest": best_rf,
        "Tuned Gradient Boosting": best_gb,
        "GB with SMOTE": GradientBoostingClassifier(random_state=42).fit(X_orig_train_smote, y_train_smote),
        "GB with Feature Engineering": GradientBoostingClassifier(random_state=42).fit(X_eng_train_scaled, y_train),
        "GB with Poly Features": GradientBoostingClassifier(random_state=42).fit(X_poly_train, y_train)
    }

    # Dictionary to store results
    results = {}
    trained_models = {}

    # Evaluate all models
    for name, model in models.items():
        if "SMOTE" in name:
            trained_model, metrics, confusion = evaluate_model(
                models[name], X_orig_train_smote, X_orig_test_scaled, y_train_smote, y_test, name
            )
        elif "Feature Engineering" in name:
            trained_model, metrics, confusion = evaluate_model(
                models[name], X_eng_train_scaled, X_eng_test_scaled, y_train, y_test, name
            )
        elif "Poly Features" in name:
            trained_model, metrics, confusion = evaluate_model(
                models[name], X_poly_train, X_poly_test, y_train, y_test, name
            )
        else:
            trained_model, metrics, confusion = evaluate_model(
                models[name], X_orig_train_scaled, X_orig_test_scaled, y_train, y_test, name
            )

        results[name] = {"metrics": metrics, "confusion": confusion}
        trained_models[name] = trained_model

    # Plot ROC and precision-recall curves
    plot_roc_curves(
        trained_models,
        X_orig_test_scaled,
        X_eng_test_scaled,
        X_poly_test,
        y_test
    )

    plot_precision_recall_curves(
        trained_models,
        X_orig_test_scaled,
        X_eng_test_scaled,
        X_poly_test,
        y_test
    )

    # Plot feature importance for random forest and gradient boosting
    if hasattr(trained_models["Tuned Random Forest"], "feature_importances_"):
        plot_feature_importance(
            feature_names_orig,
            trained_models["Tuned Random Forest"].feature_importances_,
            "Random Forest Feature Importance"
        )

    if hasattr(trained_models["Tuned Gradient Boosting"], "feature_importances_"):
        plot_feature_importance(
            feature_names_orig,
            trained_models["Tuned Gradient Boosting"].feature_importances_,
            "Gradient Boosting Feature Importance"
        )

    # Feature importance for the engineered features
    if hasattr(trained_models["GB with Feature Engineering"], "feature_importances_"):
        plot_feature_importance(
            feature_names_eng,
            trained_models["GB with Feature Engineering"].feature_importances_,
            "Feature Engineered Gradient Boosting Importance"
        )

    # Compare across metrics in a nice table
    comparison_table = pd.DataFrame({
        model_name: {
            'Accuracy': results[model_name]['metrics']['accuracy'],
            'Precision': results[model_name]['metrics']['precision'],
            'Recall': results[model_name]['metrics']['recall'],
            'F1 Score': results[model_name]['metrics']['f1']
        } for model_name in results.keys()
    }).round(4) * 100  # Convert to percentages

    print("\n=== Advanced Model Comparison (%) ===")
    print(comparison_table)

    # Save comparison to CSV
    comparison_table.to_csv('advanced_model_comparison.csv')
    print("Comparison saved to advanced_model_comparison.csv")

    # Print conclusions and insights
    print("\n=== Key Insights ===")

    # Find best model by F1 score
    best_model = max(results.items(), key=lambda x: x[1]['metrics']['f1'])[0]
    print(f"Best overall model (by F1 score): {best_model}")

    # Find best model by recall
    best_recall_model = max(results.items(), key=lambda x: x[1]['metrics']['recall'])[0]
    print(f"Best model for maximizing recall: {best_recall_model}")

    # Find best model by precision
    best_precision_model = max(results.items(), key=lambda x: x[1]['metrics']['precision'])[0]
    print(f"Best model for maximizing precision: {best_precision_model}")

    # Calculate improvement over basic logistic regression
    print("\nImprovement over basic logistic regression:")
    baseline_f1 = 0.6364  # From your previous output
    best_f1 = results[best_model]['metrics']['f1']
    improvement = (best_f1 - baseline_f1) / baseline_f1 * 100
    print(f"F1 score improvement: {improvement:.2f}%")


if __name__ == "__main__":
    main()