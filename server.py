import os
from fuzzywuzzy import fuzz, process
import chess.pgn
import json
import numpy as np
from fastapi.responses import JSONResponse
import boto3
from typing import Any, Optional
from fastapi import FastAPI
from dotenv import load_dotenv
# import BaseModel
from pydantic import BaseModel
from mlflow.tracking import MlflowClient
from starlette.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from datetime import datetime
from typing import List
import uvicorn
# import MlflowException
# import HTTPException
import mlflow
from fastapi import HTTPException
from mlflow.exceptions import MlflowException
app = FastAPI()
# import HTTPException
PORT = 5555
load_dotenv()

origins = [
    f"http://localhost:{PORT}",
    "https://chat.openai.com",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



origins = [
    f"http://localhost:{PORT}",
    "https://chat.openai.com",
]


@app.route("/.well-known/ai-plugin.json")
async def get_manifest(request):
    file_path = "./.well-known/ai-plugin.json"
    return FileResponse(file_path, media_type="text/json")


@app.route("/.well-known/logo.png")
async def get_logo(request):
    file_path = "./.well-known/logo.png"
    return FileResponse(file_path, media_type="text/json")


@app.route("/.well-known/openapi.yaml")
async def get_openapi(request):
    print(f"request: {request} for openapi.yaml")
    file_path = "./.well-known/openapi.yaml"
    return FileResponse(file_path, media_type="text/json")

import matplotlib.pyplot as plt
from pydantic import Field


class SearchRequest(BaseModel):
    name: str


def is_name_match(name, player_name):
    # Adjust this threshold as needed
    return fuzz.token_set_ratio(name, player_name) >= 80


def get_games_from_pgn(pgn_file, name):
    games = []
    with open(pgn_file) as file:
        while True:
            game = chess.pgn.read_game(file)
            if game is None:
                break

            white = game.headers.get("White", "")
            black = game.headers.get("Black", "")

            if is_name_match(name, white) or is_name_match(name, black):
                game_info = {
                    "event": game.headers.get("Event", ""),
                    "site": game.headers.get("Site", ""),
                    "date": game.headers.get("Date", ""),
                    "round": game.headers.get("Round", ""),
                    "white": white,
                    "black": black,
                    "result": game.headers.get("Result", ""),
                }
                games.append(game_info)
    return games

@app.post("/games")
async def search_games(request: SearchRequest):
    pgns_path = "pgns"
    pgns = os.listdir(pgns_path)

    all_games = []
    for pgn in pgns:
        pgn_file = os.path.join(pgns_path, pgn)
        games = get_games_from_pgn(pgn_file, request.name)
        all_games.extend(games)

    if not all_games:
        raise HTTPException(status_code=404, detail="Name not found")

    return all_games





@app.get("/generate-openapi-yaml")
async def generate_openapi_yaml():
    import yaml  # Import the pyyaml library

    import os
    from fastapi.openapi.utils import get_openapi

    # Get the OpenAPI JSON schema
    openapi_schema = get_openapi(
        title="My Application",
        version="1.0.0",
        routes=app.routes,
    )

    # Convert the JSON schema to YAML
    openapi_yaml = yaml.dump(openapi_schema)

    # Create the .well-known directory if it doesn't exist
    os.makedirs(".well-known", exist_ok=True)

    # Write the YAML schema to the .well-known/openapi.yaml file
    print("Writing OpenAPI YAML schema to .well-known/openapi.yaml")
    with open(".well-known/openapi.yaml", "w") as yaml_file:
        yaml_file.write(openapi_yaml)

    return {"detail": "OpenAPI YAML schema has been generated and stored in .well-known/openapi.yaml"}


@app.on_event("startup")
async def on_startup():
    # Call the generate_openapi_yaml function during the startup event
    await generate_openapi_yaml()

def start():
    uvicorn.run("server:app",
                host="0.0.0.0", port=PORT, reload=True)
    

if __name__ == "__main__":
    start()