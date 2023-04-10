import requests
import pandas as pd
import os
from typing import Optional
from fuzzywuzzy import fuzz
import chess.pgn
from fastapi import FastAPI
from dotenv import load_dotenv
# import BaseModel
from pydantic import BaseModel
from starlette.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from io import StringIO
from fastapi import HTTPException
import glob
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


def get_games_from_chess_com(username):
    url = f"https://api.chess.com/pub/player/{username}/games/archives"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(
            f"Failed to get games for user {username}: {response.text}")

    archives = response.json()["archives"]

    games = []
    for archive_url in archives:
        month_games = requests.get(archive_url)
        if month_games.status_code != 200:
            raise Exception(
                f"Failed to get games for user {username}: {month_games.text}")

        pgn_data = month_games.text
        pgn_file = StringIO(pgn_data)
        print(f"pgn_file: {month_games.text} for {username}")

        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            game_info = {
                "event": game.headers.get("Event", ""),
                "site": game.headers.get("Site", ""),
                "date": game.headers.get("Date", ""),
                "round": game.headers.get("Round", ""),
                "white": game.headers.get("White", ""),
                "black": game.headers.get("Black", ""),
                "result": game.headers.get("Result", ""),
            }
            games.append(game_info)
    return games


def get_opening_name(board, openings_df):
    moves = board.move_stack
    closest_opening = ""
    closest_distance = float("inf")

    for index, row in openings_df.iterrows():
        opening_pgn = row["pgn"].split(" ")
        distance = len(moves) - len(opening_pgn)
        if distance < 0:
            continue

        same_moves = all(
            [board.move_stack[i].uci() == opening_pgn[i]
             for i in range(len(opening_pgn))]
        )

        if same_moves and distance < closest_distance:
            closest_opening = row["eco_name"]
            closest_distance = distance
            if distance == 0:
                break

    return closest_opening

class SearchRequest(BaseModel):
    name: Optional[str] = None
    username_chess_com: Optional[str] = None
    color: Optional[str] = None


def is_name_match(name, player_name):
    # Adjust this threshold as needed
    return fuzz.token_set_ratio(name, player_name) >= 80



openings = glob.glob("chess-openings/*.tsv")

openings_df = pd.concat([pd.read_csv(opening, sep="\t") for opening in openings])

import gen
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
                # Get the first 10 moves
                moves = []
                node = game
                for _ in range(10):
                    node = node.variations[0] if node.variations else None
                    if node is None:
                        break
                    moves.append(node.move.uci())

                # Get the opening name
                board = game.board()
                for move in moves:
                    board.push(chess.Move.from_uci(move))
                opening_name = get_opening_name(board, openings_df)

                game_info = {
                    "event": game.headers.get("Event", ""),
                    "site": game.headers.get("Site", ""),
                    "date": game.headers.get("Date", ""),
                    "round": game.headers.get("Round", ""),
                    "white": white,
                    "black": black,
                    "result": game.headers.get("Result", ""),
                    "moves": moves,
                    "opening_name": opening_name
                }
                games.append(game_info)
    return games

@app.post("/games")
async def search_games(request: SearchRequest):
    pgns_path = "pgns"
    pgns = os.listdir(pgns_path)

    all_games = []
    username_chess_com = request.username_chess_com
    if username_chess_com:
        games = get_games_from_chess_com(username_chess_com)[:15]
        all_games.extend(games)

    if request.name:
        for pgn in pgns:
            pgn_file = os.path.join(pgns_path, pgn)
            games = get_games_from_pgn(pgn_file, request.name)[:15]
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