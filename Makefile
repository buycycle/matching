# Local and Development

setup:
	conda create  --name matching --file requirements.txt python=3.12

activate:
	conda activate .env/matching

juypter:
	@cd notebook; PYTHONPATH=".." jupyter notebook notebook.ipynb

test:
	## run test cases in tests directory
	python -m unittest discover
	pytest -v tests

lint:
	pylint --disable=R,C,W1203,W1202 src/
	pylint --disable=R,C,W1203,W1202 model/



	# check style with flake8
	mypy .
	flake8 .

format:
	# Check Python formatting
	black --line-length 130 .



build-docker:
	## Run Docker locally

	docker compose build

	docker compose up app


all: install lint test

