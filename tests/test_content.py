"""
 test content recommendation provision realted functions
 test with prepacked test DB read-in, as fixtures
"""

import time

from tests.test_fixtures import inputs, testdata_content

import numpy as np
from src.driver_content import prefilter_features

from src.strategies import MixedStrategy


def test_time_MixedStrategy(inputs, testdata_content):
    """test time of recommendation for a predifined bike_id with DB read-in"""
    bike_id, distinct_id, family_id, price, frame_size_code, n, ratio, app, logger = inputs

    df, df_status_masked, df_popularity, similarity_matrix = testdata_content

    start_time = time.time()

    mixed_strategy = MixedStrategy(df, df_status_masked, df_popularity, similarity_matrix, prefilter_features, logger)
    recommendation, len_prefiltered, error = mixed_strategy.get_recommendations(bike_id, family_id, price, frame_size_code, n)

    end_time = time.time()
    assert end_time - start_time < 0.1, "MixedStrategy took more than 100 ms to execute"


def test_len_sample_(inputs, testdata_content, n_test=100):
    """test length of recommendation list for a random subset of bike_ids in the similarity_matrix rows
    similarity_matrix rows are bike_ids with all statuses
    """

    bike_id, distinct_id, family_id, price, frame_size_code, n, ratio, app, logger = inputs

    df, df_status_masked, df_popularity, similarity_matrix = testdata_content

    for i in similarity_matrix.sample(n_test).index:
        bike_id = i

        mixed_strategy = MixedStrategy(df, df_status_masked, df_popularity, similarity_matrix, prefilter_features, logger)
        recommendation, len_prefiltered, error = mixed_strategy.get_recommendations(bike_id, family_id, price, frame_size_code, n)

        assert (
            len(recommendation) == n
        ), f"MixedStrategy recommendation has {len(recommendation)} rows for bike_id {bike_id}, which is not {n} rows"


def test_len_random_MixedStrategy(inputs, testdata_content, n_test=100):
    """test length of recommendation list for radnom bike ids
    similarity_matrix rows are bike_ids with all statuses
    """

    bike_id, distinct_id, family_id, price, frame_size_code, n, ratio, app, logger = inputs

    df, df_status_masked, df_popularity, similarity_matrix = testdata_content

    # do n_test times for i between 0 and 50000
    for i in range(n_test):

        # i is a random number between 0 and 50000
        bike_id = int(50000 * np.random.random_sample())

        mixed_strategy = MixedStrategy(df, df_status_masked, df_popularity, similarity_matrix, prefilter_features, logger)
        recommendation, len_prefiltered, error = mixed_strategy.get_recommendations(bike_id, family_id, price, frame_size_code, n)

        assert (
            len(recommendation) == n
        ), f"MixedStrategy recommendation has {len(recommendation)} rows for bike_id {bike_id}, which is not {n} rows"
