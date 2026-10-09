import chess

from engine import (
    get_board_key,
    order_moves_from_list,
    pop_search_move,
    prepare_search_repetitions,
    push_search_move,
    repetition_count,
)


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"{name:<42} {status}")
    return bool(condition)


def main():
    passed = 0
    total = 0

    board = chess.Board()
    prepare_search_repetitions(board)

    total += 1
    passed += check(
        "Repetition count inicial = 1",
        repetition_count(get_board_key(board)) == 1,
    )

    # Repete a posição inicial três vezes no histórico:
    # Nf3 Nf6 Ng1 Ng8, duas voltas.
    for uci in (
        "g1f3", "g8f6", "f3g1", "f6g8",
        "g1f3", "g8f6", "f3g1", "f6g8",
    ):
        board.push_uci(uci)

    prepare_search_repetitions(board)
    current_key = get_board_key(board)

    total += 1
    passed += check(
        "Repetition counter reconhece 3 ocorrências",
        repetition_count(current_key) >= 3,
    )

    before_fen = board.fen()
    before_count = repetition_count(current_key)
    move = next(iter(board.legal_moves))
    child_key = push_search_move(board, move)
    child_count = repetition_count(child_key)
    pop_search_move(board, child_key)

    total += 1
    passed += check(
        "push/pop restaura posição",
        board.fen() == before_fen,
    )

    total += 1
    passed += check(
        "push/pop restaura repetition counter",
        repetition_count(current_key) == before_count and child_count >= 1,
    )

    legal_moves = list(board.legal_moves)
    infos = order_moves_from_list(
        board,
        legal_moves,
        ply=0,
        board_key=current_key,
    )

    total += 1
    passed += check(
        "Move picker mantém todos os lances legais",
        len(infos) == len(legal_moves)
        and {info.move for info in infos} == set(legal_moves),
    )

    print("=" * 58)
    print(f"Resultado: {passed}/{total}")


if __name__ == "__main__":
    main()
