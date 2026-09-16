import time
import chess

from engine import find_best_move_timed, clear_engine_state, get_search_stats


TIME_PER_TEST = 1.0
MAX_DEPTH = 64


TESTS = [
    {
        "name": "Mate em 1 para brancas",
        "fen": "7k/8/5KQ1/8/8/8/8/8 w - - 0 1",
        "expected": {"g6g7"},
    },
    {
        "name": "Mate em 1 para pretas",
        "fen": "8/8/8/8/8/5kq1/8/7K b - - 0 1",
        "expected": {"g3g2"},
    },
    {
        "name": "Brancas ganham a dama",
        "fen": "4k3/8/8/3q4/8/8/3Q4/4K3 w - - 0 1",
        "expected": {"d2d5"},
    },
    {
        "name": "Pretas ganham a dama",
        "fen": "4k3/3q4/8/8/3Q4/8/8/4K3 b - - 0 1",
        "expected": {"d7d4"},
    },
    {
        "name": "Promoção branca para dama",
        "fen": "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "expected": {"a7a8q"},
    },
    {
        "name": "Promoção preta para dama",
        "fen": "4k3/8/8/8/8/8/p7/4K3 b - - 0 1",
        "expected": {"a2a1q"},
    },

    # posiçoes medias

    {
        "name": "Capturar torre",
        "fen": "rn1qkbn1/pppppppp/8/3b1r2/4P3/8/PPPP1PPP/RNBQKBNR w KQq - 0 1",
        "expected": {"e4f5"},
    },

]


def move_to_san(board, move):
    if move is None:
        return "nenhuma"

    try:
        return board.san(move)
    except Exception:
        return move.uci()


def run_test(test_number, test):
    board = chess.Board(test["fen"])

    print("=" * 60)
    print(f"Teste {test_number}: {test['name']}")
    print(f"FEN: {test['fen']}")

    if not board.is_valid():
        print("AVISO: posição FEN pode não ser totalmente válida.")
        print()

    clear_engine_state()

    start = time.perf_counter()

    best_move, score, depth = find_best_move_timed(
        board,
        time_limit=TIME_PER_TEST,
        max_depth=MAX_DEPTH,
    )

    elapsed = time.perf_counter() - start
    stats = get_search_stats()

    if best_move is None:
        best_uci = "nenhuma"
    else:
        best_uci = best_move.uci()

    expected_moves = test["expected"]

    passed = best_uci in expected_moves

    print(f"Jogada esperada: {', '.join(sorted(expected_moves))}")
    print(f"Jogada da engine: {best_uci} ({move_to_san(board, best_move)})")
    print(f"Avaliação: {score}")
    print(f"Profundidade atingida: {depth}")
    print(f"Tempo: {elapsed:.2f}s")
    print(f"Nós normais: {stats['nodes']}")
    print(f"Nós quiescence: {stats['qnodes']}")
    print(f"TT hits: {stats['tt_hits']}")
    print(f"Cutoffs: {stats['cutoffs']}")

    if passed:
        print("Resultado: PASS")
    else:
        print("Resultado: FAIL")

    print()

    return passed


def main():
    print("Benchmark / testes da Barromax")
    print(f"Tempo por teste: {TIME_PER_TEST}s")
    print()

    passed_count = 0

    for index, test in enumerate(TESTS, start=1):
        if run_test(index, test):
            passed_count += 1

    total = len(TESTS)

    print("=" * 60)
    print(f"Resultado final: {passed_count}/{total} testes certos")

    if passed_count == total:
        print("Excelente. A engine passou todos os testes.")
    else:
        print("Alguns testes falharam. Isso não quer dizer que a engine esteja partida.")
        print("Pode significar que precisa de mais tempo, mais profundidade ou melhor avaliação.")


if __name__ == "__main__":
    main()