# Binary Classification Model Progression

This project demonstrates a progressive approach to binary classification, from a custom logistic regression implementation to advanced ensemble methods. It uses the Pima Indians Diabetes dataset to predict diabetes risk and focuses on evaluation metrics, model optimization, and feature engineering.

## Project Structure

- `dataset.py`: Data loading and preprocessing
- `metrics.py`: Implementation of evaluation metrics (precision, recall, F1)
- `logistic_regression.py`: Custom implementation of logistic regression
- `main.py`: Basic implementation with custom logistic regression
- `model_comparison.py`: Comparison of multiple classification algorithms
- `advanced_comparison.py`: Advanced techniques with feature engineering and hyperparameter tuning

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

### Model Comparison

For a comparison of different models:

```bash
python model_comparison.py
```

This script compares:
- Custom logistic regression implementation
- Scikit-learn's logistic regression
- Random forest classifier

### Advanced Model Comparison

For state-of-the-art results with advanced techniques:

```bash
python advanced_comparison.py
```

This enhanced script includes:
1. Advanced models:
   - Tuned Logistic Regression (with grid search)
   - Tuned Random Forest (with grid search)
   - Gradient Boosting Classifier (with grid search)
2. Feature engineering techniques:
   - Polynomial features
   - Interaction terms
   - Feature transformations
3. Class imbalance handling with SMOTE
4. Advanced visualizations:
   - ROC curves with AUC
   - Precision-recall curves with iso-F1 lines
   - Feature importance plots

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

## Why This Approach?

### Traditional ML vs. Deep Learning

This project uses scikit-learn rather than deep learning frameworks like TensorFlow or PyTorch for several reasons:

1. **Appropriateness**: For tabular data with relatively few features, classical ML often performs just as well as neural networks
2. **Data efficiency**: Traditional ML models typically need less training data to perform well
3. **Interpretability**: Models like Random Forest provide clear feature importance rankings
4. **Simplicity**: Traditional ML algorithms are more interpretable and require less computational resources
5. **Educational value**: Understanding these fundamental algorithms provides a strong foundation

### Model Progression

The project demonstrates a natural progression in model sophistication:

1. **Custom Logistic Regression**: Understanding the basics from scratch
2. **Scikit-learn Models**: Leveraging optimized implementations
3. **Ensemble Methods**: Capturing complex patterns with Random Forest
4. **Gradient Boosting**: Achieving higher performance with sequential learning
5. **Advanced Techniques**: Feature engineering, hyperparameter tuning, and handling class imbalance

