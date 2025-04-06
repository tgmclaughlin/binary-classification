import numpy as np
from typing import Tuple


class LogisticRegression:
    """
    Simple implementation of Logistic Regression for binary classification.
    """

    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000):
        """
        Initialize the logistic regression model.

        Args:
            learning_rate: Step size for gradient descent
            num_iterations: Number of iterations for optimization
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.

        Args:
            z: Input to the sigmoid function

        Returns:
            Output of sigmoid function
        """
        # Clip z to avoid overflow in exp
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))

    def _initialize_parameters(self, n_features: int) -> None:
        """
        Initialize model parameters.

        Args:
            n_features: Number of input features
        """
        self.weights = np.zeros(n_features)
        self.bias = 0

    def _compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the binary cross-entropy loss.

        Args:
            X: Features matrix
            y: Target vector

        Returns:
            Loss value
        """
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        h = self._sigmoid(z)

        # Compute binary cross-entropy loss
        epsilon = 1e-15  # Small value to avoid log(0)
        h = np.clip(h, epsilon, 1 - epsilon)
        cost = -1/m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

        return cost

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute gradients for weight and bias parameters.

        Args:
            X: Features matrix
            y: Target vector

        Returns:
            Gradients for weights and bias
        """
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        h = self._sigmoid(z)

        # Compute gradients
        dw = 1/m * np.dot(X.T, (h - y))
        db = 1/m * np.sum(h - y)

        return dw, db

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> dict:
        """
        Train the logistic regression model using gradient descent.

        Args:
            X: Training features
            y: Training labels
            verbose: Whether to print training progress

        Returns:
            Dictionary with training history
        """
        # Get number of features and samples
        m, n = X.shape

        # Initialize parameters
        self._initialize_parameters(n)

        # Training history
        history = {'cost': []}

        # Gradient descent
        for i in range(self.num_iterations):
            # Compute gradients
            dw, db = self._compute_gradients(X, y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Compute and store cost
            if i % 100 == 0 or i == self.num_iterations - 1:
                cost = self._compute_cost(X, y)
                history['cost'].append(cost)

                if verbose and (i % 100 == 0):
                    print(f"Iteration {i}: Cost = {cost:.4f}")

        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features matrix

        Returns:
            Predicted probabilities for positive class
        """
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features matrix
            threshold: Classification threshold

        Returns:
            Predicted class labels (0 or 1)
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)