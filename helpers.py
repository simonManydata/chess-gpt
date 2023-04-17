import chess
import json
import os
import io
import chess.engine
import chess.pgn


def add_score_change(pgn_string, engine, limit_per_move=0.1):
    # Initialize chess engine
    # Parse the PGN string
    pgn = chess.pgn.read_game(io.StringIO(pgn_string))

    # Iterate through the moves, evaluate the position, and add score change to the PGN
    node = pgn
    board = node.board()
    previous_score = None
    while not node.is_end():
        next_node = node.variations[0]
        move = next_node.move
        board.push(move)

        # Evaluate the position
        info = engine.analyse(board, chess.engine.Limit(time=limit_per_move),)
        current_score = info["score"].white().score()

        if previous_score is not None:
            # Calculate the change in score and add it as a comment
            score_change = current_score - previous_score

            if score_change > 15: # show big changes
                next_node.comment = f"{score_change}"


        # Update the previous score and move to the next position
        previous_score = current_score
        node = next_node

    # Close the chess engine
    # engine.quit()

    # Export the modified PGN
    exporter = chess.pgn.StringExporter(
        headers=True, variations=True, comments=True)
    modified_pgn_string = pgn.accept(exporter)
    return modified_pgn_string


def classify_fen(fen, ecodb):
    """
    Searches a JSON file with Encyclopedia of Chess Openings (ECO) data to
    check if the given FEN matches an existing opening record
    Returns a classification
    A classfication is a dictionary containing the following elements:
        "code":         The ECO code of the matched opening
        "desc":         The long description of the matched opening
        "path":         The main variation of the opening
    """
    classification = {}
    classification["code"] = ""
    classification["desc"] = ""
    classification["path"] = ""

    for opening in ecodb:
        if opening['f'] == fen:
            classification["code"] = opening['c']
            classification["desc"] = opening['n']
            classification["path"] = opening['m']

    return classification


def eco_fen(board):
    """
    Takes a board position and returns a FEN string formatted for matching with
    eco.json
    """
    board_fen = board.board_fen()
    castling_fen = board.castling_xfen()

    to_move = 'w' if board.turn else 'b'

    return "{} {} {}".format(board_fen, to_move, castling_fen)


def classify_opening(game):
    """
    Takes a game and adds an ECO code classification for the opening
    Returns the classified game and root_node, which is the node where the
    classification was made
    """
    ecopath = os.path.join(os.path.dirname("./"), 'eco/eco.json')
    with open(ecopath, 'r') as ecofile:
        ecodata = json.load(ecofile)

        ply_count = 0

        root_node = game.root()
        node = game.end()

        while not node == game.root():
            prev_node = node.parent

            fen = eco_fen(node.board())
            classification = classify_fen(fen, ecodata)

            if classification["code"] != "":
                # Add some comments classifying the opening
                node.root().headers["ECO"] = classification["code"]
                node.root().headers["Opening"] = classification["desc"]
                node.comment = "{} {}".format(classification["code"],
                                              classification["desc"])
                # Remember this position so we don't analyze the moves
                # preceding it later
                root_node = node
                # Break (don't classify previous positions)
                break

            ply_count += 1
            node = prev_node

        return node.root(), root_node, ply_count
