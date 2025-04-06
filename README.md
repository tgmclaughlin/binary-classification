# Binary Classification Model Comparison

This project implements and compares different binary classification models using the Pima Indians Diabetes dataset to predict diabetes onset. It includes a custom logistic regression implementation from scratch, as well as scikit-learn models for comparison.

## Project Structure

- `dataset.py`: Data loading and preprocessing
- `metrics.py`: Evaluation metrics (precision, recall, F1)
- `logistic_regression.py`: Custom implementation of logistic regression 
- `main.py`: Basic script to run the logistic regression pipeline
- `model_comparison.py`: Advanced script comparing multiple classification models

## Getting Started

### Prerequisites

- Python 3.13
- NumPy
- Pandas
- Matplotlib
- scikit-learn (only for data preprocessing and comparison)

## Installation

To install the required dependencies:

```bash
uv sync
```

## Running Experiments

### Basic Logistic Regression

Run the main script for the basic logistic regression implementation:

```bash
python main.py
```

This will:
1. Load and preprocess the Pima Indians Diabetes dataset
2. Split the data into training and test sets
3. Train a custom logistic regression model
4. Evaluate the model on test data
5. Print evaluation metrics (accuracy, precision, recall, F1)
6. Show feature importance

### Advanced Model Comparison

For state-of-the-art results with advanced techniques:

```bash
python advanced_comparison.py
```

This enhanced script includes:
1. Advanced models:
   - Tuned Logistic Regression (with grid search)
   - Tuned Random Forest (with grid search)
   - Gradient Boosting (with grid search)
2. Feature engineering techniques:
   - Polynomial features
   - Interaction terms
   - Feature transformations
3. Class imbalance handling with SMOTE
4. Advanced visualizations:
   - ROC curves with AUC
   - Precision-recall curves with iso-F1 lines
   - Feature importance for all models
5. Detailed metrics and cross-validated results

## Understanding Evaluation Metrics

### Precision, Recall, and F1 Score

These metrics are crucial for evaluating binary classification models, especially when dealing with imbalanced datasets:

- **Precision**: Measures how many of the positively predicted instances are actually positive. 
  - Formula: TP / (TP + FP)
  - Interpretation: "When the model predicts positive, how often is it correct?"

- **Recall**: Measures how many of the actual positive instances were correctly predicted.
  - Formula: TP / (TP + FN)
  - Interpretation: "What percentage of actual positives did the model catch?"

- **F1 Score**: Harmonic mean of precision and recall, providing a single metric that balances both.
  - Formula: 2 * (Precision * Recall) / (Precision + Recall)
  - Interpretation: "A balanced measure between precision and recall"

Where:
- TP = True Positives
- FP = False Positives
- TN = True Negatives
- FN = False Negatives

### Why These Metrics Matter

- Accuracy alone can be misleading, especially with imbalanced classes.
- Precision focuses on minimizing false positives (Type I errors).
- Recall focuses on minimizing false negatives (Type II errors).
- F1 provides a balance when both types of errors are important.

In medical diagnoses like diabetes prediction, recall is often more important than precision, as missing a positive case (false negative) can be more harmful than a false alarm (false positive).

## Why This Approach?

### Traditional ML vs. Deep Learning

This project uses scikit-learn rather than deep learning frameworks like TensorFlow or PyTorch for several reasons:

1. **Simplicity**: Traditional ML algorithms are more interpretable and require less computational resources
2. **Appropriateness**: For tabular data with relatively few features, classical ML often performs just as well as neural networks
3. **Educational value**: Understanding these fundamental algorithms provides a strong foundation before moving to more complex approaches
4. **Efficiency**: Training and inference are much faster, especially on limited hardware
5. **Less data dependency**: Classical ML models typically need less training data to perform well

### Random Forest Benefits

The model comparison includes Random Forest because it:
- Handles non-linear relationships that logistic regression cannot capture
- Is less prone to overfitting with proper tuning
- Provides useful feature importance metrics
- Often performs well with minimal hyperparameter tuning
- Handles class imbalance better than logistic regression

## Next Steps

After exploring this project, you can:

1. Implement more advanced models (Gradient Boosting, SVM, etc.)
2. Try feature engineering and selection techniques
3. Implement proper cross-validation for more robust evaluation
4. Explore hyperparameter tuning with grid search or random search
5. Handle class imbalance using techniques like SMOTE
6. Deploy the best model as a simple web service
7. Try deep learning approaches with TensorFlow or PyTorch to compare performance