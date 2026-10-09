import chess

from engine import static_exchange_eval


def check(name, fen, uci, expected_sign):
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)

    if move not in board.legal_moves:
        raise AssertionError(f"{name}: lance ilegal: {uci}")

    score = static_exchange_eval(board, move)

    if expected_sign < 0:
        passed = score < 0
    elif expected_sign > 0:
        passed = score > 0
    else:
        passed = score == 0

    print(f"{name:<32} SEE={score:>5}  {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    tests = [
        (
            "Cavalo por peao defendido",
            "4k3/5p2/4p3/8/5N2/8/8/4K3 w - - 0 1",
            "f4e6",
            -1,
        ),
        (
            "Peao captura dama",
            "4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1",
            "e4d5",
            1,
        ),
    ]

    passed = sum(check(*test) for test in tests)
    print()
    print(f"Resultado SEE: {passed}/{len(tests)}")


if __name__ == "__main__":
    main()
