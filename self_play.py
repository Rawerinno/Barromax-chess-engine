import json
import math
import os
import random
import time

import chess

from engine import (
    find_best_move_timed,
    clear_engine_state,
    order_moves,
    evaluate_board,
)


DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "selfplay_positions.jsonl")

GAMES_TO_GENERATE = 50
TIME_PER_MOVE = 0.15
MAX_DEPTH = 6
MAX_PLIES = 160

RANDOM_OPENING_PLIES = 12
RANDOM_TOP_MOVES = 6
RANDOM_PROBABILITY = 0.75


def choose_move(board, ply):
    """
    Escolhe uma jogada para self-play.

    Nas primeiras jogadas há alguma aleatoriedade para criar variedade.
    Depois usa a engine normalmente.
    """

    legal_moves = list(board.legal_moves)

    if not legal_moves:
        return None

    if ply < RANDOM_OPENING_PLIES and random.random() < RANDOM_PROBABILITY:
        ordered = order_moves(board)
        candidates = ordered[:RANDOM_TOP_MOVES]

        if candidates:
            return random.choice(candidates)

    move, score, depth = find_best_move_timed(
        board,
        time_limit=TIME_PER_MOVE,
        max_depth=MAX_DEPTH,
    )

    if move is None:
        return random.choice(legal_moves)

    return move


def get_final_target(board):
    """
    Resultado do ponto de vista das brancas.

    +1 = brancas ganharam
     0 = empate
    -1 = pretas ganharam

    Se a partida chegou ao limite de lances sem acabar,
    usamos a avaliação clássica da engine como alvo suave.
    """

    outcome = board.outcome(claim_draw=True)

    if outcome is not None:
        if outcome.winner == chess.WHITE:
            return 1.0

        if outcome.winner == chess.BLACK:
            return -1.0

        return 0.0

    final_eval = evaluate_board(board)

    # Transforma centipawns num valor entre -1 e +1.
    return math.tanh(final_eval / 800.0)


def save_positions(game_number, positions, target, final_fen):
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(DATA_PATH, "a", encoding="utf-8") as f:
        for ply, fen in enumerate(positions):
            row = {
                "game": game_number,
                "ply": ply,
                "fen": fen,
                "target": target,
                "final_fen": final_fen,
            }

            f.write(json.dumps(row) + "\n")


def play_one_game(game_number):
    board = chess.Board()
    positions = []

    clear_engine_state()

    start_time = time.perf_counter()

    for ply in range(MAX_PLIES):
        if board.is_game_over(claim_draw=True):
            break

        positions.append(board.fen())

        move = choose_move(board, ply)

        if move is None:
            break

        board.push(move)

    target = get_final_target(board)
    final_fen = board.fen()

    save_positions(game_number, positions, target, final_fen)

    elapsed = time.perf_counter() - start_time

    result_text = board.result(claim_draw=True)

    print(
        f"Jogo {game_number}: "
        f"{len(positions)} posições, "
        f"resultado {result_text}, "
        f"target {target:.3f}, "
        f"tempo {elapsed:.1f}s"
    )


def main():
    print("A gerar partidas self-play da Barromax")
    print(f"Jogos: {GAMES_TO_GENERATE}")
    print(f"Tempo por jogada: {TIME_PER_MOVE}s")
    print(f"Ficheiro de saída: {DATA_PATH}")
    print()

    os.makedirs(DATA_DIR, exist_ok=True)

    for game_number in range(1, GAMES_TO_GENERATE + 1):
        play_one_game(game_number)

    print()
    print("Self-play terminado.")
    print(f"Dados guardados em: {DATA_PATH}")


if __name__ == "__main__":
    main()