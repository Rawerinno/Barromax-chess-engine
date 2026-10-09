import time
import chess

from engine import (
    USE_AI,
    clear_engine_state,
    find_best_move_timed,
    get_search_stats,
)


TIME_PER_POSITION = 5.0
MAX_DEPTH = 64


POSITIONS = [
    (
        "Italiana",
        [
            "e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6",
            "d2d3", "f8c5", "c2c3", "d7d6", "e1g1", "e8g8",
            "b2b4", "c5b6", "a2a4", "a7a6", "b1d2",
        ],
    ),
    (
        "Gambito da Dama",
        [
            "d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6",
            "c1g5", "f8e7", "e2e3", "e8g8", "g1f3", "b7b6",
            "c4d5", "e6d5", "f1d3", "c8b7", "e1g1", "b8d7",
        ],
    ),
    (
        "Siciliana",
        [
            "e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4",
            "f3d4", "g8f6", "b1c3", "a7a6", "c1e3", "e7e6",
            "f2f3", "b7b5", "d1d2", "c8b7", "e1c1", "b8d7",
        ],
    ),
    (
        "Francesa",
        [
            "e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6",
            "e4e5", "f6d7", "f2f4", "c7c5", "g1f3", "b8c6",
            "c1e3", "a7a6", "d1d2", "b7b5",
        ],
    ),
    (
        "India do Rei",
        [
            "d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7",
            "e2e4", "d7d6", "g1f3", "e8g8", "f1e2", "e7e5",
            "e1g1", "b8c6", "d4d5", "c6e7", "b2b4",
        ],
    ),
    (
        "Inglesa",
        [
            "c2c4", "e7e5", "b1c3", "g8f6", "g2g3", "d7d5",
            "c4d5", "f6d5", "f1g2", "d5b6", "g1f3", "b8c6",
            "e1g1", "f8e7", "d2d3", "e8g8", "c1e3",
        ],
    ),
]


def make_board(moves):
    board = chess.Board()

    for text in moves:
        move = chess.Move.from_uci(text)
        if move not in board.legal_moves:
            raise ValueError(f"Jogada ilegal na posição de benchmark: {text}")
        board.push(move)

    return board


def main():
    print("Benchmark de depth - Barromax 2.3")
    print(f"AI ativa: {USE_AI}")
    print(f"Tempo por posição: {TIME_PER_POSITION:.1f}s")
    print("=" * 96)

    depths = []
    total_nodes = 0
    total_time = 0.0
    total_tt_hits = 0
    total_eval_hits = 0
    total_rfp = 0
    total_lmp = 0
    total_null = 0
    total_rep = 0
    total_movegen = 0

    for name, moves in POSITIONS:
        board = make_board(moves)
        clear_engine_state()

        start = time.perf_counter()
        best_move, score, depth = find_best_move_timed(
            board,
            time_limit=TIME_PER_POSITION,
            max_depth=MAX_DEPTH,
        )
        elapsed = time.perf_counter() - start
        stats = get_search_stats()

        nodes = stats["nodes"] + stats["qnodes"]
        nps = int(nodes / elapsed) if elapsed > 0 else nodes

        depths.append(depth)
        total_nodes += nodes
        total_time += elapsed
        total_tt_hits += stats["tt_hits"]
        total_eval_hits += stats.get("eval_cache_hits", 0)
        total_rfp += stats.get("rfp_prunes", 0)
        total_lmp += stats.get("lmp_prunes", 0)
        total_null += stats.get("null_cutoffs", 0)
        total_rep += stats.get("repetition_draws", 0)
        total_movegen += stats.get("movegen_nodes", 0)

        move_text = best_move.uci() if best_move is not None else "nenhuma"

        print(
            f"{name:<20} "
            f"depth={depth:<2} "
            f"move={move_text:<6} "
            f"score={score:<6} "
            f"nodes={nodes:<9} "
            f"nps={nps:<7} "
            f"tt={stats['tt_hits']:<6} "
            f"evalHit={stats.get('eval_cache_hits', 0):<6} "
            f"rfp={stats.get('rfp_prunes', 0):<5} "
            f"lmp={stats.get('lmp_prunes', 0):<5} "
            f"null={stats.get('null_cutoffs', 0):<5} "
            f"rep={stats.get('repetition_draws', 0):<4}"
        )

    avg_depth = sum(depths) / len(depths) if depths else 0.0
    overall_nps = int(total_nodes / total_time) if total_time > 0 else 0

    print("=" * 96)
    print(f"Depth médio:       {avg_depth:.2f}")
    print(f"Nós totais:        {total_nodes}")
    print(f"NPS global:        {overall_nps}")
    print(f"TT hits:           {total_tt_hits}")
    print(f"Static eval hits:  {total_eval_hits}")
    print(f"RFP prunes:        {total_rfp}")
    print(f"LMP prunes:        {total_lmp}")
    print(f"Null cutoffs:      {total_null}")
    print(f"Repetition draws:  {total_rep}")
    print(f"Movegen nodes:     {total_movegen}")


if __name__ == "__main__":
    main()
