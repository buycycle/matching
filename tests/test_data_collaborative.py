"""
    Test data realted functions, queries the DB for test data
    Unittest class for testing the DB read-in
    Rest in Pytest
"""

import pytest

import datetime

from tests.test_fixtures import inputs, testdata_collaborative

from src.data_content import get_data


def test_dataset_length(testdata_collaborative, limit=1000):
    """number of users and items is at least limit"""

    df, metadata, model, dataset = testdata_collaborative

    n_users = dataset.interactions_shape()[0]
    n_items = dataset.interactions_shape()[1]

    assert n_users >= limit, f"dataset has n_users {n_users}, which is less than {limit}"
    assert n_items >= limit, f"dataset has n_items {n_users}, which is less than {limit}"


def test_dataset_current(testdata_collaborative):
    """dataset contains at least yesterday's data"""

    df, metadata, model, dataset = testdata_collaborative

    # check if datediff current date to metadate is > 3 days
    data_age = datetime.date.today() - metadata

    assert data_age.days >= 1, f"dataset age {data_age}, which is higher than 1 days"
