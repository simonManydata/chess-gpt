# README.md

## FastAPI MLflow Application

This is a FastAPI based application that provides an API for interacting with MLflow experiments, runs, and models. It allows you to search experiments, register models, and retrieve experiment runs.

### Features

- Searching experiments based on keywords and date range
- Registering a model with a specific name
- Retrieving a list of runs for a given experiment
- Generating OpenAPI YAML schema for the application

### Installation

1. Ensure you have Python 3.7 or later installed on your system.
2. Install the required dependencies using the following command:

```sh
pip install fastapi uvicorn mlflow pydantic python-dotenv
```

### Usage

To run the application, execute the following command:

```sh
python app.py
```

The application will start on port 5555 by default. You can access the API by navigating to `http://localhost:5555`.

### API Endpoints

- `POST /experiments`: Search experiments based on keyword, minimum date, and maximum date.
- `POST /register_model/{run_id}`: Register a model from a specific run with a given name.
- `POST /runs/`: Retrieve a list of runs for a given experiment.
- `GET /generate-openapi-yaml`: Generate OpenAPI YAML schema for the application.

### Configuration

You can customize the port by modifying the `PORT` variable in the script. The default is 5555. CORS is also configured to allow requests from `http://localhost:{PORT}` and `https://chat.openai.com`.




## Mlflow server 
mlflow server --default-artifact-root s3://mlflow-server-sm-gpt  --artifacts-destination s3://mlflow-server-sm-gpt/artifacts