"""test fixutres used in the tests"""

import os
import pytest

from flask import Flask
from flask.logging import create_logger


from src.data_content import read_data_content

# for create_data
from src.data_content import create_data_model_content
from src.driver_content import (
    main_query,
    main_query_dtype,
    popularity_query,
    categorical_features,
    numerical_features,
    prefilter_features,
    numerical_features_to_overweight,
    numerical_features_overweight_factor,
    categorical_features_to_overweight,
    categorical_features_overweight_factor,
)

from src.driver_collaborative import *
from src.collaborative import create_data_model_collaborative, update_model, read_data_model
from src.data_collaborative import (
    s3_client,
    s3_credentials,
    get_last_date_S3,
    get_last_date_local,
    get_multi_day_data,
    write_data,
    combine_data,
    read_data_collaborative,
)


@pytest.fixture(scope="package")
def inputs():
    bike_id = 14394
    distinct_id = "1234"
    family_id = 1101
    price = 1200
    frame_size_code = "56"
    n = 12
    ratio = 0.5
    app = Flask(__name__)
    logger = create_logger(app)

    return bike_id, distinct_id, family_id, price, frame_size_code, n, ratio, app, logger


@pytest.fixture(scope="package")
def testdata_content():

    # make folder data if not exists
    if not os.path.exists("./data/"):
        os.makedirs("./data/")

    create_data_model_content(
        main_query,
        main_query_dtype,
        popularity_query,
        categorical_features,
        numerical_features,
        prefilter_features,
        numerical_features_to_overweight,
        numerical_features_overweight_factor,
        categorical_features_to_overweight,
        categorical_features_overweight_factor,
        status=["active"],
        metric="euclidean",
        path="./data/",
    )
    df, df_status_masked, df_popularity, similarity_matrix = read_data_content()

    return df, df_status_masked, df_popularity, similarity_matrix


@pytest.fixture(scope="package")
def testdata_collaborative():

    # make folder data if not exists
    if not os.path.exists("./data/"):
        os.makedirs("./data/")

    create_data_model_collaborative(
        s3_client=s3_client,
        s3_credentials=s3_credentials,
        get_last_date_S3=get_last_date_S3,
        get_last_date_local=get_last_date_local,
        get_multi_day_data=get_multi_day_data,
        implicit_feedback=implicit_feedback,
        features=features,
        user_features=user_features,
        item_features=item_features,
        user_id=user_id,
        bike_id=bike_id,
        combine_data=combine_data,
        write_data=write_data,
        update_model=update_model,
        path="./data/",
        limit=3,
        fraction=1,
    )
    model, dataset = read_data_model(path="./data/")
    df, metadata = read_data_collaborative(path="./data/")

    return df, metadata, model, dataset
