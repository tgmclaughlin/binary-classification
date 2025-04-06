import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import our modules
from dataset import load_pima_dataset
from metrics import binary_classification_metrics, confusion_matrix_stats, print_classification_report
from logistic_regression import LogisticRegression


def plot_training_history(history):
    """Plot the training cost over iterations."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, len(history['cost']) * 100, 100), history['cost'], marker='o')
    plt.title('Training Cost over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.close()


def main():
    """Main function to train and evaluate the logistic regression model."""
    print("Loading and preprocessing Pima Indians Diabetes dataset...")
    df = load_pima_dataset()

    print(f"Dataset shape: {df.shape}")
    print("\nFeature statistics:")
    print(df.describe().round(2))

    # Split the data
    X = df.drop('outcome', axis=1).values
    y = df['outcome'].values

    # Create train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Train logistic regression model
    print("\nTraining logistic regression model...")
    model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
    history = model.fit(X_train, y_train, verbose=True)

    # Plot training history
    plot_training_history(history)

    # Make predictions
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = binary_classification_metrics(y_test, y_pred)
    confusion = confusion_matrix_stats(y_test, y_pred)

    # Print classification report
    print_classification_report(metrics, confusion)

    print("\nMost important features (absolute weight values):")
    feature_names = df.drop('outcome', axis=1).columns
    feature_importance = np.abs(model.weights)

    for name, importance in sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True):
        print(f"{name}: {importance:.4f}")


if __name__ == "__main__":
    main()