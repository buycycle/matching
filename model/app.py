"""App for matching"""

# get env variable
import os

# flask
from flask import Flask, request, jsonify

# periodical data read-in
from threading import Thread

import pandas as pd

# config file
import configparser

# get loggers
from src.logger import Logger
from src.logger import KafkaLogger


# sql queries and feature selection
from src.driver_content import prefilter_features
from src.driver_collaborative import user_id, bike_id, features, item_features, user_features, implicit_feedback

# import the function from src
from src.data_content import DataStoreContent
from src.data_collaborative import (
    s3_client,
    s3_credentials,
    get_last_date_S3,
    get_last_date_local,
    get_multi_day_data,
    write_data,
    combine_data,
)
from src.collaborative import DataStoreCollaborative
from src.strategies import MixedStrategy, CollaborativeStrategy

from src.helper import get_field_value

config_paths = "config/config.ini"

config = configparser.ConfigParser()
config.read(config_paths)

path = "data/"

app = Flask(__name__)
# read the environment from the docker environment variable
environment = os.getenv("ENVIRONMENT")
ab = os.getenv("AB")
app_name = "matching"
app_version = "0.0.4"

logger = Logger.configure_logger(environment, ab, app_name, app_version)
# logger = KafkaLogger(environment, ab, app_name, app_version)

logger.info("Flask app started")

# create data stores and load periodically
data_store_content = DataStoreContent()
data_store_collaborative = DataStoreCollaborative()

# read the data periodically
data_loader_content = Thread(target=data_store_content.read_data_periodically, args=(22, logger))
data_loader_collaborative = Thread(target=data_store_collaborative.read_data_periodically, args=(720, logger))

data_loader_content.start()
data_loader_collaborative.start()


@app.route("/")
def home():
    html = f"<h3>matching</h3>"
    return html.format(format)


@app.route("/matching", methods=["POST"])
def matching():
    """take in user_id and bike_id and return a recommendation
    the payload should be in the following format:

    {
        "user_id": int,
        "distinct_id": str,
        "bike_id": int,
        "family_id": int,
        "price": int,
        "frame_size_code": str,
        "n": int
    }

    """

    # Logging the input payload
    json_payload = request.json

    recommendation_payload = pd.DataFrame(json_payload, index=[0])

    # read from request, assign default if missing or null
    user_id = get_field_value(recommendation_payload, "user_id", 0)

    distinct_id = get_field_value(recommendation_payload, "distinct_id", "NA", dtype=str)
    bike_id = get_field_value(recommendation_payload, "bike_id", 0)

    family_id = get_field_value(recommendation_payload, "family_id", 1101)
    price = get_field_value(recommendation_payload, "price", 1200)
    frame_size_code = get_field_value(recommendation_payload, "frame_size_code", "56", dtype=str)

    n = get_field_value(recommendation_payload, "n", 12)

    # try the collaboration filtering first
    collaborative_strategy = CollaborativeStrategy(path, n, logger)

    recommendation, error = collaborative_strategy.get_recommendations(user_id=distinct_id, n=n)

    # convert the recommendation to int
    recommendation = [int(i) for i in recommendation]

    if len(recommendation) == n:
        logger.info(
            "CollaborativeStrategy",
            extra={
                "user_id": user_id,
                "distinct_id": distinct_id,
                "bike_id": bike_id,
                "family_id": family_id,
                "price": price,
                "frame_size_code": frame_size_code,
                "n": n,
                "recommendation": recommendation,
            },
        )
    if len(recommendation) != n:

        # Get recommendations using the MixedStrategy object
        mixed_strategy = MixedStrategy(
            data_store_content.df,
            data_store_content.df_status_masked,
            data_store_content.df_popularity,
            data_store_content.similarity_matrix,
            prefilter_features,
            logger,
        )
        recommendation, len_prefiltered, error = mixed_strategy.get_recommendations(bike_id, family_id, price, frame_size_code, n)

        # Log the output prediction value
        logger.info(
            "Content based MixedStrategy",
            extra={
                "user_id": user_id,
                "distinct_id": distinct_id,
                "bike_id": bike_id,
                "family_id": family_id,
                "price": price,
                "frame_size_code": frame_size_code,
                "n": n,
                "len_prefiltered": len_prefiltered,
                "recommendation": recommendation,
            },
        )

    if error:
        # Return error response if it exists
        logger.error("Error no recommendation available, exception: " + error)
        return (
            jsonify({"status": "error", "message": "Recommendation not available"}),
            404,
        )

    else:
        # Return success response with recommendation data and 200 OK
        return (jsonify({"status": "success", "recommendation": recommendation}), 200)


# Error handling for 400 Bad Request
@app.errorhandler(400)
def bad_request_error(e):
    # Log the error details using the provided logger
    logger.error(
        "400 Bad Request:", extra={"info": "user_id, bike_id and n must be convertable to integers",},
    )

    return (
        jsonify({"status": "error", "message": "Bad Request, user_id, bike_id and n must be convertable to integers"}),
        400,
    )


# add 500 error handling

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=80)
