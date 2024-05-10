"""
    Test data realted functions, queries the DB for test data
    Unittest class for testing the DB read-in
    Rest in Pytest
"""

import time

import pytest
import unittest

from tests.test_fixtures import inputs, testdata_content

from src.driver_content import (
    main_query,
    main_query_dtype,
    popularity_query,
    categorical_features,
    numerical_features,
    prefilter_features,
)


from src.data_content import get_data


class TestData(unittest.TestCase):
    """unittest used to check how long DB read takes"""

    def setUp(self):
        self.main_query = main_query
        self.main_query_dtype = main_query_dtype
        self.popularity_query = popularity_query
        self.config_paths = "config/config.ini"
        self.prefilter_features = prefilter_features
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.df, self.df_popularity = get_data(
            main_query=self.main_query,
            main_query_dtype=self.main_query_dtype,
            popularity_query=self.popularity_query,
            config_paths=self.config_paths,
        )

    def test_get_data_time(self):
        start_time = time.time()
        get_data(
            main_query=self.main_query,
            main_query_dtype=self.main_query_dtype,
            popularity_query=self.popularity_query,
            config_paths=self.config_paths,
        )
        end_time = time.time()
        self.assertLess(end_time - start_time, 120, "get_data() took more than 2 minutes to execute")


def test_get_data_length(testdata_content):

    df, df_status_masked, df_popularity, similarity_matrix = testdata_content

    assert len(df) >= 10000, f"df has {len(df)} rows, which is less than 10000 rows"
    assert len(df_popularity) >= 7000, f"df_popularity has {len(df_popularity)} rows, which is less than 7000 rows"


def test_columns_in_get_data(testdata_content):
    """All features are in the DB read-in"""

    df, df_status_masked, df_popularity, similarity_matrix = testdata_content

    features = prefilter_features + categorical_features + numerical_features

    for feature in prefilter_features:
        assert feature in df.columns, f"{feature} is not in the dataframe"


def test_get_data_na_drop_ratio(testdata_content):

    df, df_status_masked, df_popularity, similarity_matrix = testdata_content

    na_ratio = 1 - (len(df) / len(df_popularity))

    assert na_ratio < 0.15, f"NA ratio is {na_ratio:.2f}, which is above 0.15"
