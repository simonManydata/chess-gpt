import boto3
from fastapi.responses import JSONResponse
import chess.engine
import requests
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

from helpers import add_score_change, classify_opening, extract_moves_from_game, extract_moves_from_pgn, hash_pgn_to_8_characters, moves_to_pgn

bucket_name = "chess-gpt"
expiration = 3600
s3_client = boto3.client('s3')

app = FastAPI()

# Constants
ERROR_THRESHOLD = {
    'BLUNDER': -300,
    'MISTAKE': -150,
    'DUBIOUS': -75,
}
NEEDS_ANNOTATION_THRESHOLD = 7.5

# import HTTPException
PORT = 5555
load_dotenv()
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
TIME_ANALYSIS = 0.1

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

from enum import Enum
# Define an enumeration for the allowed color values


class ChessColor(str, Enum):
    white = "white"
    black = "black"


class SearchRequest(BaseModel):
    name: Optional[str] = None
    color: Optional[ChessColor] = None
    opening: Optional[str] = None


def is_name_match(name, player_name):
    # Adjust this threshold as needed
    return fuzz.token_set_ratio(name, player_name) >= 80



openings = glob.glob("chess-openings/*.tsv")



GAMES = {
}

def get_games_from_pgn(pgn_file, name, number_moves=300): # 300 moves as upper bound
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
                if is_name_match(name, white):
                    white = name
                if is_name_match(name, black):
                    black = name
                moves = []
                node = game
                for _ in range(number_moves):
                    node = node.variations[0] if node.variations else None
                    if node is None:
                        break
                    moves.append(node.move.uci())

                # # Get the opening name
                # board = game.board()
                # for move in moves:
                #     board.push(chess.Move.from_uci(move))
                
                game, root_node, ply_count = classify_opening(game)
                moves = extract_moves_from_game(game)
                pgn_moves = moves_to_pgn(moves)

                opening = game.headers["Opening"]

                pgn_id = hash_pgn_to_8_characters(pgn_moves)
                GAMES[pgn_id] = pgn_moves

                game_info = {
                    # "event": game.headers.get("Event", ""),
                    # "site": game.headers.get("Site", ""),
                    "date": game.headers.get("Date", ""),
                    # "round": game.headers.get("Round", ""),
                    "white": white,
                    "black": black,
                    "result": game.headers.get("Result", ""),
                    # "mainline_moves": pgn_moves,
                    "opening": opening,
                    "pgn_id": pgn_id,
                }
                games.append(game_info)
    return games

@app.post("/games")
async def search_games(request: SearchRequest):
    pgns_path = "pgns"
    pgns = os.listdir(pgns_path)

    all_games = []
    LIMIT_GAMES = 15

    if request.name:
        for pgn in pgns:
            pgn_file = os.path.join(pgns_path, pgn)
            games = get_games_from_pgn(pgn_file, request.name)#[:LIMIT_GAMES]
            all_games.extend(games)


    all_games_filtered = []
    for game in all_games:
        if request.color:
            if request.color == ChessColor.white and game["white"] == request.name:
                all_games_filtered.append(game)
            elif request.color == ChessColor.black and game["black"] == request.name:
                all_games_filtered.append(game)
        elif request.opening:
            if request.opening.lower() in game["opening"].lower():
                all_games_filtered.append(game)
    if not all_games:
        raise HTTPException(status_code=404, detail="Name not found")

    return all_games


# Define a data model for the request body
class EvaluateMovesRequest(BaseModel):
    pgn_id: Optional[str]

# Define a data model for the response


class Evaluation(BaseModel):
    pgn_with_score_change: str

# Define the endpoint for evaluating moves

@app.post("/evaluate_moves", response_model=Evaluation)
async def evaluate_moves(request: EvaluateMovesRequest, description="Evaluate score change for each moves. After getting the response, return a markdown table of the worst moves made by each player (player|move|description|how to improve)") -> Evaluation:
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    # Parse the PGN
    limit_per_move = 0.1
    try:
        pgn_data = GAMES[request.pgn_id]
        pgn_data = add_score_change(pgn_data, engine, limit_per_move)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid PGN")
    # Initialize the chess engine (replace "path/to/stockfish" with the actual path to the Stockfish binary)
    engine.close()
    return Evaluation(pgn_with_score_change=pgn_data)




class GenerateGifGameRequest(BaseModel):
    pgn_data: Optional[str]
    pgn_id: Optional[str]
    reverse: bool = False

@app.post("/generate-gif-game/", description="Generate a gif game from a PGN file. Use reverse if player is black. Embed the gif in markdown with ![]({url})")
async def generate_gif_game(request: GenerateGifGameRequest):
    if request.pgn_id:
        pgn_data = GAMES[request.pgn_id]
    else:
        pgn_data = request.pgn_data
    reverse: bool = request.reverse
    import uuid
    import tempfile
    from pgn_to_gif import PgnToGifCreator
    creator = PgnToGifCreator(reverse=reverse, duration=1.3)

    with tempfile.TemporaryDirectory() as temp_dir:
        input_pgn_path = os.path.join(temp_dir, "input.pgn")
        random_filename = str(uuid.uuid4())
        output_gif_path = os.path.join(temp_dir, f"{random_filename}.gif")

        with open(input_pgn_path, "w") as pgn_file:
            pgn_file.write(pgn_data)

        creator.create_gif(input_pgn_path, out_path=output_gif_path)

        # Set the appropriate values for your S3 bucket and object key
        
        object_key = f"gif/{random_filename}.gif"

        # Upload the file to S3
        s3 = boto3.client("s3")
        s3.upload_file(output_gif_path, bucket_name, object_key)

        # Generate a presigned URL
        signed_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': object_key},
            ExpiresIn=expiration
        )

        return JSONResponse(content={"url": signed_url})

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