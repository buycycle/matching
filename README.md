# matching

Develop a system that effectively matches bikes with potential buyers to increase the likelihood of a sale and reduce the probability of item deletion. Find optimal synergistic strategies to in-cooperate boosted listings. Use this system in the ranking and the landing page. Also test if adding this as a threshold into the recommendations gives better results.



## Requirements

* [Docker version 20 or later](https://docs.docker.com/install/#support)

## Setup development environment

We setup the conda development environment with the following command.

- `make setup`

Install requirements

- `make install`

## Lint and formatting

- `make lint`

- `make format`


## Docker

when creating an docker image the data is downloaded and prepared. Build test and production stages and runs tests.

- `docker compose build`

Run app.

- `docker compose up app`


## Driver and Config

src/driver.py defines the SQL queries, categorical and numerical features as well as the prefilter_features.
config/config.ini holdes DB credentials.

## Endpoint

REST API

### Get 'add get command'



## known issues

### content
	
### model
	

## To-does


### Dev 


