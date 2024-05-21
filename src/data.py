import os
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    RandomForestClassifier,
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from joblib import dump, load
from buycycle.data import sql_db_read, DataStoreBase
from quantile_forest import RandomForestQuantileRegressor, ExtraTreesQuantileRegressor

import threading  # for data read lock


def get_data(
    main_query: str, main_query_dtype: str, index_col: str = "status", config_paths: str = "config/config.ini"
) -> pd.DataFrame:
    """
    Fetches data from SQL database.
    Args:
        main_query: SQL query for main data.
        main_query_dtype: Data type for main query.
        index_col: Index column for DataFrame. Default is 'sales_price'.
        config_paths: Path to configuration file. Default is 'config/config.ini'.
    Returns:
        DataFrame: Main data.
    """
    df = sql_db_read(query=main_query, DB="DB_BIKES", config_paths=config_paths, dtype=main_query_dtype, index_col=index_col)
    return df


def clean_data(
    df: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str],
    target: str = "status",
    iqr_limit: float = 2,
) -> pd.DataFrame:
    """
    Cleans data by removing outliers and unnecessary data.
    Args:
        df: DataFrame to clean.
        numerical_features: List of numerical feature names.
        categorical_features: List of categorical feature names.
        target: Target column. Default is 'sales_price'.
        iqr_limit: IQR limit for outlier detection. Default is 2.
    Returns:
        DataFrame: Cleaned data.
    """
    # only keep categorical and numerical features
    df = df[categorical_features + numerical_features + [target]]
    # remove custom template idf and where target = NA
    df = df.loc[~df.index.duplicated(keep="last")]
    df.dropna(inplace=True)

    # Remove outliers from numerical features
    for feature in numerical_features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Filter out the outliers
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # replace bike_created_at_month with a sinusoidal transformation
    df["bike_created_at_month_sin"] = np.sin(2 * np.pi * df["bike_created_at_month"] / 12)
    df["bike_created_at_month_cos"] = np.cos(2 * np.pi * df["bike_created_at_month"] / 12)
    # create bike age from bike_year
    df["bike_age"] = pd.to_datetime("today").year - df["bike_year"]

    return df

def create_data(
    query: str, query_dtype: str, numerical_features: List[str], categorical_features: List[str], target: str, path: str = "data/"
):
    """
    Fetches, cleans, splits, and saves data.
    Args:
        query: SQL query for main data.
        query_dtype: Data type for main query.
        numerical_features: numerical_features.
        arget: Target column.
        path: Path to save data. Default is 'data/'.
    """
    df = get_data(query, query_dtype, index_col="id")
    print(f"Dimensions of df after get_data: {df.shape}")
    df = clean_data(df, numerical_features, categorical_features, target=target).sample(frac=1)
    print(f"Dimensions of df after clean_data and sampling: {df.shape}")
    df = feature_engineering(df)
    print(f"Dimensions of df after feature_engineering: {df.shape}")
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[target]), df[target], test_size=0.2, random_state=0)
    categorical_feature_indices = [df.columns.get_loc(c) for c in categorical_features if c in df]

    smote_nc = SMOTENC(categorical_features=categorical_feature_indices, random_state=0)
    X_train, y_train = smote_nc.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test


def read_data(path: str = "data/"):
    """
    Reads saved data from disk.
    Args:
        path: Path to saved data. Default is 'data/'.
    Returns:
        X_train, y_train, X_test, y_test: Training and test sets.
    """
    X_train = pd.read_pickle(path + "X_train.pkl")
    y_train = pd.read_pickle(path + "y_train.pkl")
    X_test = pd.read_pickle(path + "X_test.pkl")
    y_test = pd.read_pickle(path + "y_test.pkl")
    return X_train, y_train, X_test, y_test


class DummyCreator(BaseEstimator, TransformerMixin):
    """
    Class for one-hot encoding
    """

    def __init__(self, categorical_features: Optional[List[str]] = None):
        """
        Initialize DummyCreator.
        Parameters: categorical_features : list of str (optional, default encodes all)
        """
        self.categorical_features = categorical_features

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "DummyCreator":
        """
        Fit DummyCreator on X.
        Parameters: X : DataFrame (input data), y : DataFrame (optional, for API consistency)
        Returns: self : DummyCreator (fitted instance)
        """
        self.encoder_ = (
            ce.OneHotEncoder(cols=self.categorical_features, handle_unknown="indicator", use_cat_names=True)
            if self.categorical_features
            else ce.OneHotEncoder(handle_unknown="indicator", use_cat_names=True)
        )
        self.encoder_.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode categorical features.
        Parameters: X : DataFrame (input data)
        Returns: DataFrame (transformed data)
        """
        return self.encoder_.transform(X)


class Scaler(BaseEstimator, TransformerMixin):
    """
    Class for scaling features using MinMaxScaler
    """

    def __init__(self, numerical_features: Optional[List[str]] = None):
        """
        Initialize Scaler.
        Parameters: numerical_features : list of str (optional, default scales all)
        """
        self.scaler_ = MinMaxScaler()
        self.numerical_features = numerical_features

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> "Scaler":
        """
        Fit Scaler on X.
        Parameters: X : DataFrame (input data), y : DataFrame (optional, for API consistency)
        Returns: self : Scaler (fitted instance)
        """
        self.scaler_.fit(X[self.numerical_features]) if self.numerical_features else self.scaler_.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features.
        Parameters: X : DataFrame (input data)
        Returns: DataFrame (transformed data)
        """
        X.loc[:, self.numerical_features] = (
            self.scaler_.transform(X[self.numerical_features]) if self.numerical_features else self.scaler_.transform(X)
        )
        return X


def encode_target(y_train: pd.Series, y_test: pd.Series) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Encode the target variable.
    Args:
    ----------
    y_train : Series
        Training target data.
    y_test : Series
        Testing target data.
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, LabelEncoder]
        Encoded training target, encoded testing target, and fitted LabelEncoder.
    """
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    return y_train_encoded, y_test_encoded, label_encoder


def create_data_transform_pipeline(
    categorical_features: Optional[List[str]] = None, numerical_features: Optional[List[str]] = None
) -> Pipeline:
    """
    Create a pipeline for data preprocessing.
    Args:
    ----------
    categorical_features : list of str, optional
        Categorical features to process. If None, all are processed.
    numerical_features : list of str, optional
        Numerical features to process. If None, all are processed.
    Returns
    -------
    Pipeline
        Constructed pipeline.
    """

    # One-hot encoding for categorical features
    categorical_encoder = DummyCreator(categorical_features)

    # Scaler for numerical features
    numerical_scaler = Scaler(numerical_features)

    # Create the pipeline with the custom imputer, one-hot encoder, and scaler
    data_transform_pipeline = Pipeline(steps=[("dummies", categorical_encoder), ("scale", numerical_scaler),])
    return data_transform_pipeline


def fit_transform(
    X_train: pd.DataFrame, X_test: pd.DataFrame, categorical_features: List[str], numerical_features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    """
    Fit and transform the data using the pipeline.
    Args:
    ----------
    X_train : DataFrame
        Training data.
    X_test : DataFrame
        Testing data.
    categorical_features : list of str, optional
        Categorical features to process. If None, all are processed.
    numerical_features : list of str, optional
        Numerical features to process. If None, all are processed.
    Returns
    -------
    Tuple[DataFrame, DataFrame, Pipeline]
        Transformed training data, transformed testing data, and fitted pipeline.
    """
    data_transform_pipeline = create_data_transform_pipeline(categorical_features, numerical_features)
    X_train = data_transform_pipeline.fit_transform(X_train)
    X_test = data_transform_pipeline.transform(X_test)

    print(
        f"Model has been fit and transformed with numerical features: {numerical_features} "
        f"and categorical features: {categorical_features}"
    )
    return X_train, X_test, data_transform_pipeline


def write_model_pipeline(regressor: Callable, data_transform_pipeline: Callable, path: str) -> None:
    """
    Save the regressor and the data transformation pipeline to a file.
    Args:
        regressor: A trained model that can be used to make predictions.
        data_transform_pipeline: A pipeline that performs data transformations.
        path: The file path where the model and pipeline should be saved.
    Returns:
        None
    """

    model_file_path = os.path.join(path, "model.joblib")
    pipeline_file_path = os.path.join(path, "pipeline.joblib")

    # Save the regressor to the model file
    dump(regressor, model_file_path)

    # Save the pipeline to the pipeline file
    dump(data_transform_pipeline, pipeline_file_path)


def read_model_pipeline(path: Optional[str] = "./data/") -> Tuple[Callable, Callable]:
    """
    Load the regressor and the data transformation pipeline from a file.
    Args:
        path: The file path from which the model and pipeline should be loaded.
    Returns:
        regressor: The loaded model that can be used to make predictions.
        data_transform_pipeline: The loaded pipeline that performs data transformations.
    """
    # Construct file paths for the model and pipeline
    model_file_path = os.path.join(path, "model.joblib")
    pipeline_file_path = os.path.join(path, "pipeline.joblib")

    # Load the regressor from the model file
    regressor = load(model_file_path)

    # Load the pipeline from the pipeline file
    data_transform_pipeline = load(pipeline_file_path)

    return regressor, data_transform_pipeline


def create_data_model(
    path: str,
    main_query: str,
    main_query_dtype: Dict[str, Any],
    numerical_features: List[str],
    categorical_features: List[str],
    model: BaseEstimator,
    target: str,
    months: int,
    parameters: Optional[Dict[str, Union[int, float]]] = None,
) -> None:
    """
    Create data and model
    write the data, model and datapipeline to a given path.
    Args:
    - path: The path where the model and pipeline will be written.
    - main_query: The main query to create data.
    - main_query_dtype: Data types for the main query.
    - numerical_features: List of numerical feature names.
    - categorical_features: List of categorical feature names.
    - model: The machine learning model to be trained.
    - target: The target variable name.
    - months: The number of months to consider in the data.
    - parameters: Optional dictionary of hyperparameters for the model.
    Side effects:
    - Writes the trained model and data transformation pipeline to the specified path.
    Returns:
    - None
    """
    X_train, X_test, y_train, y_test = create_data(
        main_query, main_query_dtype, numerical_features, categorical_features, target, months
    )

    X_train, X_test, data_transform_pipeline = fit_transform(X_train, X_test, categorical_features, numerical_features)

    regressor = train(X_train, y_train, model, target, parameters, scoring=mean_absolute_percentage_error,)

    X_train.to_pickle(path + "X_train.pkl")
    y_train.to_pickle(path + "y_train.pkl")
    X_test.to_pickle(path + "X_test.pkl")
    y_test.to_pickle(path + "y_test.pkl")
    write_model_pipeline(regressor, data_transform_pipeline, path)

    return categorical_features, numerical_features


class ModelStore(DataStoreBase):
    def __init__(self):
        super().__init__()
        self.regressor = None
        self.data_transform_pipeline = None
        self._lock = threading.Lock()  # Initialize a lock object

    def read_data(self):
        with self._lock:  # Acquire the lock before reading data
            self.regressor, self.data_transform_pipeline = read_model_pipeline()

    def get_logging_info(self):
        return {"regressor_info": str(self.regressor), "data_transform_pipeline": str(self.data_transform_pipeline)}
