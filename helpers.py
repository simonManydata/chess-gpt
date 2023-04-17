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

