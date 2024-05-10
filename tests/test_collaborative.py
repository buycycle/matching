"""
 test collaborative recommendation provision realted functions
"""

import time
import random

from tests.test_fixtures import inputs, testdata_collaborative

from src.strategies import CollaborativeStrategy


def test_time_CollaborativeStrategy(inputs, testdata_collaborative):
    """test time of recommendation for a predifined bike_id with DB read-in"""
    bike_id, distinct_id, family_id, price, frame_size_code, n, ratio, app, logger = inputs

    df, metadata, model, dataset = testdata_collaborative

    start_time = time.time()
    collaborative_strategy = CollaborativeStrategy("./data/", n, logger)

    recommendation, error = collaborative_strategy.get_recommendations(user_id=distinct_id, n=n)

    end_time = time.time()
    assert end_time - start_time < 0.1, "CollaborativeStrategy took more than 100 ms to execute"


def test_len_CollaborativeStrategy(inputs, testdata_collaborative, n_test=100):
    """test length of recommendation list for a random subset of user in dataset"""
    bike_id, distinct_id, family_id, price, frame_size_code, n, ratio, app, logger = inputs

    df, metadata, model, dataset = testdata_collaborative

    collaborative_strategy = CollaborativeStrategy("./data/", n, logger)

    users = dataset.mapping()[0].keys()
    users = list(users)

    # subsample
    for i in random.sample(users, n_test):
        distinct_id = i

        recommendation, error = collaborative_strategy.get_recommendations(user_id=distinct_id, n=n)

        assert (
            len(recommendation) == n
        ), f"recommendation has {len(recommendation)} rows for distinct_id {distinct_id}, which is not {n} rows"
