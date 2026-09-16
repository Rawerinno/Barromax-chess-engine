import time
import chess

from engine import (
    find_best_move,
    find_best_move_timed,
    get_search_stats,
    clear_engine_state,
    set_hash_size,
)


ENGINE_NAME = "Barromax"
ENGINE_AUTHOR = "Barro"

board = chess.Board()

DEFAULT_DEPTH = 3
DEFAULT_TIME = 1.0
MAX_THINK_TIME = 3.0
MIN_THINK_TIME = 0.10


def send(message):
    print(message, flush=True)


def send_engine_id():
    send(f"id name {ENGINE_NAME}")
    send(f"id author {ENGINE_AUTHOR}")
    send("option name Hash type spin default 64 min 1 max 1024")
    send("option name MaxThinkTime type spin default 3000 min 100 max 30000")
    send("option name Clear Hash type button")


def get_int_option(parts, name, default=None):
    if name not in parts:
        return default

    index = parts.index(name)

    if index + 1 >= len(parts):
        return default

    try:
        return int(parts[index + 1])
    except ValueError:
        return default


def mate_score_limit():
    return 990_000


def score_to_uci(score):
    score = int(score)

    if score > mate_score_limit():
        plies = 1_000_000 - score
        moves = max(1, (plies + 1) // 2)
        return f"score mate {moves}"

    if score < -mate_score_limit():
        plies = 1_000_000 + score
        moves = max(1, (plies + 1) // 2)
        return f"score mate -{moves}"

    return f"score cp {score}"


def parse_position(command):
    global board

    parts = command.split()

    if "startpos" in parts:
        board = chess.Board()

        if "moves" in parts:
            moves_index = parts.index("moves")
            moves = parts[moves_index + 1:]

            for move_text in moves:
                try:
                    board.push_uci(move_text)
                except ValueError:
                    break

        return

    if "fen" in parts:
        fen_index = parts.index("fen")

        if "moves" in parts:
            moves_index = parts.index("moves")
            fen_parts = parts[fen_index + 1:moves_index]
            moves = parts[moves_index + 1:]
        else:
            fen_parts = parts[fen_index + 1:]
            moves = []

        fen = " ".join(fen_parts)

        try:
            board = chess.Board(fen)
        except ValueError:
            board = chess.Board()
            return

        for move_text in moves:
            try:
                board.push_uci(move_text)
            except ValueError:
                break


def parse_setoption(command):
    global MAX_THINK_TIME

    parts = command.split()

    if "name" not in parts:
        return

    name_index = parts.index("name")

    if "value" in parts:
        value_index = parts.index("value")
        name = " ".join(parts[name_index + 1:value_index]).strip().lower()
        value = " ".join(parts[value_index + 1:]).strip()
    else:
        name = " ".join(parts[name_index + 1:]).strip().lower()
        value = ""

    if name == "hash":
        set_hash_size(value)
        return

    if name == "clear hash":
        clear_engine_state()
        return

    if name == "maxthinktime":
        try:
            milliseconds = int(value)
        except ValueError:
            return

        milliseconds = max(100, min(30000, milliseconds))
        MAX_THINK_TIME = milliseconds / 1000


def choose_time_limit(parts):
    movetime = get_int_option(parts, "movetime")

    if movetime is not None:
        return max(0.03, movetime / 1000 * 0.90)

    if "infinite" in parts:
        return MAX_THINK_TIME

    if board.turn == chess.WHITE:
        remaining = get_int_option(parts, "wtime")
        increment = get_int_option(parts, "winc", 0)
    else:
        remaining = get_int_option(parts, "btime")
        increment = get_int_option(parts, "binc", 0)

    if remaining is None:
        return DEFAULT_TIME

    seconds_remaining = remaining / 1000
    seconds_increment = increment / 1000

    if seconds_remaining <= 0.20:
        return max(0.01, seconds_remaining * 0.50)

    time_limit = seconds_remaining / 35
    time_limit += seconds_increment * 0.60

    time_limit = max(MIN_THINK_TIME, time_limit)
    time_limit = min(MAX_THINK_TIME, time_limit)

    safe_limit = max(0.03, seconds_remaining * 0.80)
    time_limit = min(time_limit, safe_limit)

    return time_limit


def send_search_info(depth, score, elapsed_ms):
    stats = get_search_stats()

    nodes = stats["nodes"] + stats["qnodes"]

    if elapsed_ms <= 0:
        nps = nodes
    else:
        nps = int(nodes * 1000 / elapsed_ms)

    tt_entries = stats["tt_entries"]
    max_tt_entries = max(1, stats["max_tt_entries"])
    hashfull = min(1000, int(tt_entries * 1000 / max_tt_entries))

    send(
        f"info depth {depth} "
        f"{score_to_uci(score)} "
        f"time {elapsed_ms} "
        f"nodes {nodes} "
        f"nps {nps} "
        f"hashfull {hashfull}"
    )


def parse_go(command):
    parts = command.split()

    if board.is_game_over(claim_draw=True):
        send("bestmove 0000")
        return

    start_time = time.perf_counter()

    if "depth" in parts:
        depth = get_int_option(parts, "depth", DEFAULT_DEPTH)

        best_move, score = find_best_move(board, depth=depth)

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        if best_move is None:
            send("bestmove 0000")
        else:
            send_search_info(depth, score, elapsed_ms)
            send(f"bestmove {best_move.uci()}")

        return

    time_limit = choose_time_limit(parts)

    best_move, score, reached_depth = find_best_move_timed(
        board,
        time_limit=time_limit,
        max_depth=64,
    )

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    if best_move is None:
        send("bestmove 0000")
    else:
        send_search_info(reached_depth, score, elapsed_ms)
        send(f"bestmove {best_move.uci()}")


def uci_loop():
    global board

    send_engine_id()
    send("uciok")

    while True:
        try:
            command = input().strip()
        except EOFError:
            break

        if command == "":
            continue

        if command == "uci":
            send_engine_id()
            send("uciok")

        elif command == "isready":
            send("readyok")

        elif command == "ucinewgame":
            board = chess.Board()
            clear_engine_state()

        elif command.startswith("setoption"):
            parse_setoption(command)

        elif command.startswith("position"):
            parse_position(command)

        elif command.startswith("go"):
            parse_go(command)

        elif command == "stop":
            pass

        elif command == "ponderhit":
            pass

        elif command == "quit":
            break


if __name__ == "__main__":
    uci_loop()