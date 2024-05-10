from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from buycycle.data import DataStoreBase, sql_db_read
from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor

from xgboost import XGBClassifier

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score, make_scorer, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model: Callable,
    parameters: Optional[Dict[str, Union[int, float]]] = None,
    scoring: Callable = accuracy_score,
) -> Tuple[Callable, Callable]:
    """
    Trains the model.
    Args:
        X_train: Dataframe of transformed training data.
        y_train: Series of target training data.
        model: Model to train.
        parameters: Parameters for the model. Default is None.
        scoring: Scoring function. Default is accuracy_score.
    Returns:
        classifier: Trained model.
    """
    classifier = model()
    if parameters:
        classifier.set_params(**parameters)
    classifier.fit(X_train, y_train)
    preds = classifier.predict(X_train)
    # If the model is a classifier and supports predict_proba, calculate log loss
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(X_train)
        # Get the class labels that the classifier is trained on
        classes = classifier.classes_
        print("{} Train log loss: {}".format(model.__name__, log_loss(y_train, probs, labels=classes)))
    else:
        print("Model does not support probability predictions. Log loss cannot be calculated.")

    return classifier


def predict(X_transformed: pd.DataFrame, classifier: Callable) -> np.ndarray:
    """
    Transform X and predicts target variable.
    Args:
        X_transformed: Transformed Features.
        classifier: Trained model.
    Returns:
        preds: Predictions.
    """
    preds = classifier.predict(X_transformed)
    return preds


def test(
    X_transformed: pd.DataFrame, y: pd.Series, classifier: Callable, scoring: Callable = accuracy_score,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Tests the model.
    Args:
        X_transformed: Transformed Test features.
        y: Test target.
        classifier: Trained model.
        scoring: Scoring function. Default is accuracy_score.
    Returns:
        preds: Predictions.
        error
    """
    preds = predict(X_transformed, classifier)
    pd.DataFrame(preds).to_pickle("data/preds.pkl")
    score = scoring(y, preds)
    print("Test accuracy: {}".format(score))
    return preds


def predict_probabilities(X_transformed: pd.DataFrame, classifier: Callable) -> np.ndarray:
    """
    Predicts class probabilities.
    Args:
        X_transformed: Transformed Features.
        classifier: Trained model.
    Returns:
        probs: Class probabilities.
    """
    probs = classifier.predict_proba(X_transformed)
    return probs


def test_probability(
    X_transformed: pd.DataFrame, y_encoded: pd.Series, classifier: Callable, label_encoder: LabelEncoder,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Tests the model's predicted probabilities using log loss and returns true labels and per-class log loss.
    Args:
        X_transformed: Transformed Test features.
        y_encoded: Encoded Test target.
        classifier: Trained model.
        label_encoder: Fitted LabelEncoder instance used to encode the labels.
    Returns:
        log_loss_value: Total log loss of the predicted probabilities.
        per_class_log_loss: Log loss for each class.
        y_true: True labels (original class names).
    """
    # Predict class probabilities
    probs = classifier.predict_proba(X_transformed)
    # Calculate the total log loss using the encoded labels
    log_loss_value = log_loss(y_encoded, probs)

    # Calculate per-class log loss
    per_class_log_loss = []
    class_labels = label_encoder.classes_  # Get the class names

    print("Test total log loss of probabilities: {}".format(log_loss_value))
    print("Per-class log loss:")
    for i, label in enumerate(class_labels):
        # Create a binary array for each class
        y_true_binary = (y_encoded == i).astype(int)
        # Calculate log loss for the current class
        class_log_loss = log_loss(y_true_binary, probs[:, i])
        per_class_log_loss.append(class_log_loss)
        print(f"  {label}: {class_log_loss}")

    # Decode the encoded labels back to the original class names
    y_true = label_encoder.inverse_transform(y_encoded)

    return log_loss_value, np.array(per_class_log_loss), y_true


# X_train, X_test, y_train, y_test= create_data(query=main_query, query_dtype="", numerical_features=numerical_features, categorical_features=categorical_features, target="status")
# X_train, X_test, data_transform_pipeline = fit_transform(X_train,
# X_test, categorical_features, numerical_features)
# y_train, y_test, labelencoder = encode_target(y_train, y_test)
# classifier = train(X_train, y_train, model = RandomForestClassifier)
# probs = predict_probabilities(X_test, classifier)
# test_probability(X_test, y_test, classifier, labelencoder)
