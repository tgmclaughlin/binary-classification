import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import precision_recall_curve
import time

# Import our modules
from dataset import load_pima_dataset
from metrics import binary_classification_metrics, confusion_matrix_stats, print_classification_report
from logistic_regression import LogisticRegression as CustomLogisticRegression


def plot_precision_recall_curves(models, X_test, y_test):
    """
    Plot precision-recall curves for multiple models.

    Args:
        models: Dictionary of model name to model object
        X_test: Test features
        y_test: Test labels
    """
    plt.figure(figsize=(10, 8))

    for name, model in models.items():
        if name == "Custom Logistic Regression":
            y_scores = model.predict_proba(X_test)
        else:
            y_scores = model.predict_proba(X_test)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
        plt.plot(recall, precision, lw=2, label=f'{name} (AUC: {np.trapz(precision, recall):.3f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('precision_recall_curve.png')
    plt.close()


def plot_feature_importance(feature_names, importance, title):
    """
    Plot feature importance.

    Args:
        feature_names: List of feature names
        importance: Array of feature importance values
        title: Plot title
    """
    indices = np.argsort(importance)
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importance[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title(title)
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.close()


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
    Main function to compare different models on the Pima Indians diabetes dataset.
    """
    print("\nLoading and preprocessing Pima Indians Diabetes dataset...")
    df = load_pima_dataset()

    # Basic dataset info
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution: \n{df['outcome'].value_counts(normalize=True).round(4)}")

    # Split features and target
    X = df.drop('outcome', axis=1).values
    y = df['outcome'].values
    feature_names = df.drop('outcome', axis=1).columns

    # Create train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Create models
    models = {
        "Custom Logistic Regression": CustomLogisticRegression(learning_rate=0.01, num_iterations=1000),
        "Sklearn Logistic Regression": SklearnLogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            class_weight='balanced'
        )
    }

    # Dictionary to store results
    results = {}
    trained_models = {}

    # Evaluate models
    for name, model in models.items():
        # Use scaled data for all models
        trained_model, metrics, confusion = evaluate_model(
            model, X_train_scaled, X_test_scaled, y_train, y_test, name
        )
        results[name] = {"metrics": metrics, "confusion": confusion}
        trained_models[name] = trained_model

    # Plot precision-recall curves
    plot_precision_recall_curves(trained_models, X_test_scaled, y_test)

    # Plot feature importance
    if hasattr(trained_models["Random Forest"], "feature_importances_"):
        plot_feature_importance(
            feature_names,
            trained_models["Random Forest"].feature_importances_,
            "Random Forest Feature Importance"
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

    print("\n=== Model Comparison (%) ===")
    print(comparison_table)

    # Save comparison to CSV
    comparison_table.to_csv('model_comparison.csv')
    print("Comparison saved to model_comparison.csv")


if __name__ == "__main__":
    main()