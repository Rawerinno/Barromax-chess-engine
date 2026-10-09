import time
import chess

from engine import (
    find_best_move_timed,
    clear_engine_state,
    get_search_stats,
    set_hash_size,
)


HASH_VALUES = [64, 128, 256, 512]
TIME_PER_TEST = 1.0
MAX_DEPTH = 64


TESTS = [
    {
        "name": "Mate em 1 brancas",
        "fen": "7k/8/5KQ1/8/8/8/8/8 w - - 0 1",
        "expected": {"g6g7"},
    },
    {
        "name": "Mate em 1 pretas",
        "fen": "8/8/8/8/8/5kq1/8/7K b - - 0 1",
        "expected": {"g3g2"},
    },
    {
        "name": "Brancas ganham dama",
        "fen": "4k3/8/8/3q4/8/8/3Q4/4K3 w - - 0 1",
        "expected": {"d2d5"},
    },
    {
        "name": "Pretas ganham dama",
        "fen": "4k3/3q4/8/8/3Q4/8/8/4K3 b - - 0 1",
        "expected": {"d7d4"},
    },
    {
        "name": "Promoção branca",
        "fen": "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "expected": {"a7a8q"},
    },
    {
        "name": "Promoção preta",
        "fen": "4k3/8/8/8/8/8/p7/4K3 b - - 0 1",
        "expected": {"a2a1q"},
    },
    {
        "name": "Capturar torre",
        "fen": "rn1qkbn1/pppppppp/8/3b1r2/4P3/8/PPPP1PPP/RNBQKBNR w KQq - 0 1",
        "expected": {"e4f5"},
    },
]


def run_hash_test(hash_mb):
    print()
    print("=" * 70)
    print(f"HASH = {hash_mb} MB")
    print("=" * 70)

    set_hash_size(hash_mb)

    total_depth = 0
    total_nodes = 0
    total_tt_hits = 0
    total_time = 0.0
    passed = 0

    for test in TESTS:
        board = chess.Board(test["fen"])

        clear_engine_state()

        start = time.perf_counter()

        move, score, depth = find_best_move_timed(
            board,
            time_limit=TIME_PER_TEST,
            max_depth=MAX_DEPTH,
        )

        elapsed = time.perf_counter() - start

        stats = get_search_stats()

        nodes = stats["nodes"] + stats["qnodes"]
        tt_hits = stats["tt_hits"]

        move_uci = move.uci() if move else "nenhuma"

        if move_uci in test["expected"]:
            result = "PASS"
            passed += 1
        else:
            result = "FAIL"

        total_depth += depth
        total_nodes += nodes
        total_tt_hits += tt_hits
        total_time += elapsed

        print(
            f"{test['name']:<24} "
            f"depth={depth:<2} "
            f"nodes={nodes:<8} "
            f"tt={tt_hits:<6} "
            f"{result}"
        )

    count = len(TESTS)

    return {
        "hash": hash_mb,
        "passed": passed,
        "avg_depth": total_depth / count,
        "nodes": total_nodes,
        "tt_hits": total_tt_hits,
        "time": total_time,
    }


def main():
    results = []

    for hash_mb in HASH_VALUES:
        results.append(run_hash_test(hash_mb))

    print()
    print("=" * 70)
    print("RESUMO")
    print("=" * 70)

    print(
        f"{'Hash':>6} "
        f"{'Testes':>8} "
        f"{'Depth médio':>12} "
        f"{'Nodes':>12} "
        f"{'TT hits':>10}"
    )

    for result in results:
        print(
            f"{result['hash']:>6} "
            f"{result['passed']:>7}/7 "
            f"{result['avg_depth']:>12.2f} "
            f"{result['nodes']:>12} "
            f"{result['tt_hits']:>10}"
        )


if __name__ == "__main__":
    main()