import pandas as pd
import numpy as np
from typing import Tuple
from urllib.request import urlretrieve

def fetch_pima_dataset(cache_path: str = "pima_indians_diabetes.csv") -> pd.DataFrame:
    """
    Fetch Pima Indians Diabetes dataset and return as DataFrame.

    Args:
        cache_path: Local path to save the dataset

    Returns:
        DataFrame with the dataset
    """
    # Define column names
    column_names = [
        'pregnancies',
        'glucose',
        'blood_pressure',
        'skin_thickness',
        'insulin',
        'bmi',
        'diabetes_pedigree',
        'age',
        'outcome'
    ]

    try:
        # Try to read from local cache first
        df = pd.read_csv(cache_path)
        # Check if headers are numeric (0, 1, 2...) and replace if needed
        if df.columns.dtype == "int64":
            df.columns = column_names
        return df
    except FileNotFoundError:
        # Download if not available locally
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        urlretrieve(url, cache_path)
        return pd.read_csv(cache_path, header=None, names=column_names)

def preprocess_pima_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the Pima Indians Diabetes dataset.

    Args:
        df: Raw DataFrame

    Returns:
        Preprocessed DataFrame
    """
    # Ensure we have the correct column names
    column_names = [
        'pregnancies',
        'glucose',
        'blood_pressure',
        'skin_thickness',
        'insulin',
        'bmi',
        'diabetes_pedigree',
        'age',
        'outcome'
    ]

    # Handle the case where we might have numeric columns from pandas
    if df.columns.dtype == "int64" or all(c.isdigit() for c in df.columns[0]):
        df.columns = column_names

    # Handle missing values (marked as 0 in original dataset)
    zero_not_acceptable = ['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi']

    # Replace inappropriate zeros with NaN
    for column in zero_not_acceptable:
        if column in df.columns:  # Check if column exists
            df[column] = df[column].replace(0, np.nan)

    # Fill missing values with median
    for column in zero_not_acceptable:
        if column in df.columns:  # Check if column exists
            median_value = df[column].median(skipna=True)
            df[column] = df[column].fillna(median_value)

    return df

def load_pima_dataset() -> pd.DataFrame:
    """
    Load and preprocess the Pima Indians Diabetes dataset.

    Returns:
        Preprocessed DataFrame ready for model training
    """
    df = fetch_pima_dataset()
    return preprocess_pima_dataset(df)

def train_test_split_pima(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the Pima dataset into train and test sets.

    Args:
        df: Preprocessed DataFrame
        test_size: Proportion of the dataset to include in the test split
        random_state: Seed for random number generator

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Shuffle the data
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Split features and target
    X = df.drop('outcome', axis=1).values
    y = df['outcome'].values

    # Calculate split indices
    test_idx = int(len(df) * (1 - test_size))

    # Split the data
    X_train, X_test = X[:test_idx], X[test_idx:]
    y_train, y_test = y[:test_idx], y[test_idx:]

    return X_train, X_test, y_train, y_test