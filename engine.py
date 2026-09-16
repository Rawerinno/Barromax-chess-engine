import chess
import chess.polyglot
import math
import time
from dataclasses import dataclass
from typing import Optional

try:
    from ai_eval import evaluate_ai_cp, clear_ai_cache
except Exception:
    def evaluate_ai_cp(board, scale=600):
        return 0

    def clear_ai_cache():
        pass



INFINITY = 10_000_000
MATE_SCORE = 1_000_000
QUIESCENCE_DEPTH = 5

AI_WEIGHT = 0.20
AI_SCALE = 600

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

MAX_TT_ENTRIES = 200_000


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


PHASE_VALUES = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
    chess.KING: 0,
}


PAWN_TABLE = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [5, 10, 10, -20, -20, 10, 10, 5],
    [5, -5, -10, 0, 0, -10, -5, 5],
    [0, 0, 0, 20, 20, 0, 0, 0],
    [5, 5, 10, 25, 25, 10, 5, 5],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [0, 0, 0, 0, 0, 0, 0, 0],
]


KNIGHT_TABLE = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20, 0, 0, 0, 0, -20, -40],
    [-30, 0, 10, 15, 15, 10, 0, -30],
    [-30, 5, 15, 20, 20, 15, 5, -30],
    [-30, 0, 15, 20, 20, 15, 0, -30],
    [-30, 5, 10, 15, 15, 10, 5, -30],
    [-40, -20, 0, 5, 5, 0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50],
]


BISHOP_TABLE = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 10, 10, 5, 0, -10],
    [-10, 5, 5, 10, 10, 5, 5, -10],
    [-10, 0, 10, 10, 10, 10, 0, -10],
    [-10, 10, 10, 10, 10, 10, 10, -10],
    [-10, 5, 0, 0, 0, 0, 5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20],
]


ROOK_TABLE = [
    [0, 0, 0, 5, 5, 0, 0, 0],
    [5, 10, 10, 10, 10, 10, 10, 5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [0, 0, 0, 5, 5, 0, 0, 0],
]


QUEEN_TABLE = [
    [-20, -10, -10, -5, -5, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 5, 5, 5, 0, -10],
    [-5, 0, 5, 5, 5, 5, 0, -5],
    [0, 0, 5, 5, 5, 5, 0, -5],
    [-10, 5, 5, 5, 5, 5, 0, -10],
    [-10, 0, 5, 0, 0, 0, 0, -10],
    [-20, -10, -10, -5, -5, -10, -10, -20],
]


KING_MIDDLEGAME_TABLE = [
    [20, 30, 10, 0, 0, 10, 30, 20],
    [20, 20, 0, 0, 0, 0, 20, 20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
]


KING_ENDGAME_TABLE = [
    [-50, -30, -30, -30, -30, -30, -30, -50],
    [-30, -20, -10, -10, -10, -10, -20, -30],
    [-30, -10, 20, 30, 30, 20, -10, -30],
    [-30, -10, 30, 40, 40, 30, -10, -30],
    [-30, -10, 30, 40, 40, 30, -10, -30],
    [-30, -10, 20, 30, 30, 20, -10, -30],
    [-30, -30, 0, 0, 0, 0, -30, -30],
    [-50, -30, -30, -30, -30, -30, -30, -50],
]


POSITION_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
}


PASSED_PAWN_BONUS = [0, 5, 10, 20, 35, 60, 100, 0]
CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]


@dataclass
class TTEntry:
    depth: int
    score: int
    flag: int
    best_move: Optional[chess.Move]


TRANSPOSITION_TABLE = {}
KILLER_MOVES = {}
HISTORY_HEURISTIC = {}

SEARCH_STATS = {
    "nodes": 0,
    "qnodes": 0,
    "tt_hits": 0,
    "cutoffs": 0,
}


class SearchTimeout(Exception):
    pass


def clear_engine_state():
    TRANSPOSITION_TABLE.clear()
    KILLER_MOVES.clear()
    HISTORY_HEURISTIC.clear()
    clear_ai_cache()

def set_hash_size(megabytes):
    global MAX_TT_ENTRIES

    try:
        megabytes = int(megabytes)
    except ValueError:
        return

    megabytes = max(1, min(1024, megabytes))
    MAX_TT_ENTRIES = megabytes * 3000
    TRANSPOSITION_TABLE.clear()


def reset_search_stats():
    SEARCH_STATS["nodes"] = 0
    SEARCH_STATS["qnodes"] = 0
    SEARCH_STATS["tt_hits"] = 0
    SEARCH_STATS["cutoffs"] = 0


def get_search_stats():
    stats = dict(SEARCH_STATS)
    stats["tt_entries"] = len(TRANSPOSITION_TABLE)
    stats["max_tt_entries"] = MAX_TT_ENTRIES
    return stats


def check_time(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise SearchTimeout()


def get_board_key(board):
    return chess.polyglot.zobrist_hash(board)


def get_game_phase(board):
    phase = 0

    for piece in board.piece_map().values():
        phase += PHASE_VALUES[piece.piece_type]

    return min(24, phase) / 24


def table_value(table, square, color):
    file = chess.square_file(square)
    rank = chess.square_rank(square)

    if color == chess.WHITE:
        return table[rank][file]

    return table[7 - rank][file]


def get_position_value(piece, square, phase):
    if piece.piece_type == chess.KING:
        middlegame = table_value(KING_MIDDLEGAME_TABLE, square, piece.color)
        endgame = table_value(KING_ENDGAME_TABLE, square, piece.color)
        return int(middlegame * phase + endgame * (1 - phase))

    table = POSITION_TABLES[piece.piece_type]
    return table_value(table, square, piece.color)


def is_terminal_draw(board):
    return (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_seventyfive_moves()
        or board.is_fivefold_repetition()
        or board.can_claim_fifty_moves()
    )


def terminal_score_for_side_to_move(board, ply):
    if board.is_checkmate():
        return -MATE_SCORE + ply

    if is_terminal_draw(board):
        return 0

    return None


def count_legal_moves(board, color):
    original_turn = board.turn
    board.turn = color

    count = board.legal_moves.count()

    board.turn = original_turn

    return count


def has_pawn_on_square(board, square, color):
    piece = board.piece_at(square)
    return piece is not None and piece.piece_type == chess.PAWN and piece.color == color


def is_passed_pawn(board, square, color):
    file = chess.square_file(square)
    rank = chess.square_rank(square)

    enemy_pawns = board.pieces(chess.PAWN, not color)

    for f in range(file - 1, file + 2):
        if f < 0 or f > 7:
            continue

        if color == chess.WHITE:
            ranks = range(rank + 1, 8)
        else:
            ranks = range(rank - 1, -1, -1)

        for r in ranks:
            if chess.square(f, r) in enemy_pawns:
                return False

    return True


def is_protected_by_pawn(board, square, color):
    for attacker_square in board.attackers(color, square):
        piece = board.piece_at(attacker_square)

        if piece is not None and piece.piece_type == chess.PAWN:
            return True

    return False


def evaluate_pawn_structure(board, color):
    pawns = list(board.pieces(chess.PAWN, color))

    if not pawns:
        return 0

    score = 0
    files = [0] * 8

    for square in pawns:
        files[chess.square_file(square)] += 1

    for count in files:
        if count > 1:
            score -= 12 * (count - 1)

    for square in pawns:
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        if color == chess.WHITE:
            progress = rank
            front_square = square + 8
        else:
            progress = 7 - rank
            front_square = square - 8

        has_left = file > 0 and files[file - 1] > 0
        has_right = file < 7 and files[file + 1] > 0

        if not has_left and not has_right:
            score -= 10

        if is_passed_pawn(board, square, color):
            score += PASSED_PAWN_BONUS[progress]

            if is_protected_by_pawn(board, square, color):
                score += 8

        if progress >= 5:
            score += 8

        if 0 <= front_square < 64 and board.piece_at(front_square) is not None:
            score -= 5

    return score


def evaluate_rooks(board, color):
    score = 0

    own_pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)

    for rook_square in board.pieces(chess.ROOK, color):
        file = chess.square_file(rook_square)
        rank = chess.square_rank(rook_square)

        own_pawn_on_file = any(chess.square(file, r) in own_pawns for r in range(8))
        enemy_pawn_on_file = any(chess.square(file, r) in enemy_pawns for r in range(8))

        if not own_pawn_on_file and not enemy_pawn_on_file:
            score += 22
        elif not own_pawn_on_file and enemy_pawn_on_file:
            score += 12

        if color == chess.WHITE and rank == 6:
            score += 15

        if color == chess.BLACK and rank == 1:
            score += 15

    return score


def evaluate_king_safety(board, color, phase):
    king_square = board.king(color)

    if king_square is None:
        return 0

    if phase < 0.35:
        return 0

    score = 0

    file = chess.square_file(king_square)
    rank = chess.square_rank(king_square)

    if color == chess.WHITE:
        if king_square in [chess.G1, chess.C1]:
            score += 35
        elif king_square == chess.E1:
            score -= 18

        front_rank = rank + 1
        second_front_rank = rank + 2
    else:
        if king_square in [chess.G8, chess.C8]:
            score += 35
        elif king_square == chess.E8:
            score -= 18

        front_rank = rank - 1
        second_front_rank = rank - 2

    for f in range(file - 1, file + 2):
        if f < 0 or f > 7:
            continue

        found_shield = False

        if 0 <= front_rank < 8:
            front_square = chess.square(f, front_rank)

            if has_pawn_on_square(board, front_square, color):
                score += 9
                found_shield = True

        if 0 <= second_front_rank < 8:
            second_square = chess.square(f, second_front_rank)

            if has_pawn_on_square(board, second_square, color):
                score += 4
                found_shield = True

        if not found_shield:
            score -= 6

    return int(score * phase)


def evaluate_development(board, color, phase):
    if phase < 0.65:
        return 0

    score = 0

    if color == chess.WHITE:
        starting_minors = [chess.B1, chess.G1, chess.C1, chess.F1]
        queen_start = chess.D1
    else:
        starting_minors = [chess.B8, chess.G8, chess.C8, chess.F8]
        queen_start = chess.D8

    for square in starting_minors:
        piece = board.piece_at(square)

        if piece is not None and piece.color == color and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            score -= 12

    queens = list(board.pieces(chess.QUEEN, color))

    if queens:
        queen_square = queens[0]

        if queen_square != queen_start and board.fullmove_number <= 10:
            score -= 10

    return score


def evaluate_center_control(board):
    score = 0

    for square in CENTER_SQUARES:
        white_attackers = len(board.attackers(chess.WHITE, square))
        black_attackers = len(board.attackers(chess.BLACK, square))

        score += 3 * (white_attackers - black_attackers)

    return score


def evaluate_classic_board(board):
    """
    Positivo = vantagem das brancas.
    Negativo = vantagem das pretas.
    """

    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -MATE_SCORE
        return MATE_SCORE

    if is_terminal_draw(board):
        return 0

    score = 0
    phase = get_game_phase(board)

    for square, piece in board.piece_map().items():
        material = PIECE_VALUES[piece.piece_type]
        position = get_position_value(piece, square, phase)

        total = material + position

        if piece.color == chess.WHITE:
            score += total
        else:
            score -= total

    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += 35

    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= 35

    score += evaluate_pawn_structure(board, chess.WHITE)
    score -= evaluate_pawn_structure(board, chess.BLACK)

    score += evaluate_rooks(board, chess.WHITE)
    score -= evaluate_rooks(board, chess.BLACK)

    score += evaluate_king_safety(board, chess.WHITE, phase)
    score -= evaluate_king_safety(board, chess.BLACK, phase)

    score += evaluate_development(board, chess.WHITE, phase)
    score -= evaluate_development(board, chess.BLACK, phase)

    score += evaluate_center_control(board)

    white_mobility = count_legal_moves(board, chess.WHITE)
    black_mobility = count_legal_moves(board, chess.BLACK)

    score += 2 * (white_mobility - black_mobility)

    if board.turn == chess.WHITE:
        score += 8
    else:
        score -= 8

    return int(score)


def evaluate_board(board):
    """
    Avaliação final da Barromax.

    Combina:
    - avaliação clássica feita à mão
    - avaliação neural treinada por self-play
    """

    if board.is_checkmate():
        return evaluate_classic_board(board)

    if is_terminal_draw(board):
        return 0

    classic_score = evaluate_classic_board(board)
    ai_score = evaluate_ai_cp(board, scale=AI_SCALE)

    final_score = (
        (1.0 - AI_WEIGHT) * classic_score
        + AI_WEIGHT * ai_score
    )

    return int(final_score)



def evaluate_for_side_to_move(board):
    score = evaluate_board(board)

    if board.turn == chess.WHITE:
        return score

    return -score


def history_key(move):
    return (move.from_square, move.to_square, move.promotion)


def update_history(move, depth):
    key = history_key(move)
    HISTORY_HEURISTIC[key] = HISTORY_HEURISTIC.get(key, 0) + depth * depth


def add_killer_move(ply, move):
    killers = KILLER_MOVES.get(ply, [])

    if move in killers:
        return

    killers.insert(0, move)
    KILLER_MOVES[ply] = killers[:2]


def capture_score(board, move):
    attacker = board.piece_at(move.from_square)

    if board.is_en_passant(move):
        victim_value = PIECE_VALUES[chess.PAWN]
    else:
        victim = board.piece_at(move.to_square)
        victim_value = PIECE_VALUES[victim.piece_type] if victim else 0

    attacker_value = PIECE_VALUES[attacker.piece_type] if attacker else 0

    return 100_000 + 10 * victim_value - attacker_value


def is_quiet_move(board, move):
    return not board.is_capture(move) and move.promotion is None


def move_order_score(board, move, ply=0, tt_move=None):
    if tt_move is not None and move == tt_move:
        return 1_000_000

    score = 0

    if move.promotion:
        score += 900_000 + PIECE_VALUES.get(move.promotion, 0)

    if board.is_capture(move):
        score += capture_score(board, move)

    elif move in KILLER_MOVES.get(ply, []):
        score += 80_000

    else:
        score += HISTORY_HEURISTIC.get(history_key(move), 0)

    if board.gives_check(move):
        score += 5_000

    if board.is_castling(move):
        score += 2_000

    piece = board.piece_at(move.from_square)

    if piece is not None:
        if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            if piece.color == chess.WHITE and chess.square_rank(move.from_square) == 0:
                score += 200
            elif piece.color == chess.BLACK and chess.square_rank(move.from_square) == 7:
                score += 200

        if move.to_square in CENTER_SQUARES:
            score += 120

    return score


def order_moves(board, ply=0, tt_move=None):
    moves = list(board.legal_moves)

    if tt_move is not None and tt_move not in moves:
        tt_move = None

    moves.sort(
        key=lambda move: move_order_score(board, move, ply, tt_move),
        reverse=True,
    )

    return moves


def is_tactical_move(board, move, include_checks=True):
    if board.is_capture(move):
        return True

    if move.promotion:
        return True

    if include_checks and board.gives_check(move):
        return True

    return False


def order_tactical_moves(board, q_depth):
    include_checks = q_depth >= 3

    moves = [
        move
        for move in board.legal_moves
        if is_tactical_move(board, move, include_checks)
    ]

    moves.sort(
        key=lambda move: move_order_score(board, move),
        reverse=True,
    )

    return moves


def probe_tt(board, depth, alpha, beta):
    key = get_board_key(board)
    entry = TRANSPOSITION_TABLE.get(key)

    if entry is None:
        return None, None

    tt_move = entry.best_move

    if entry.depth >= depth:
        SEARCH_STATS["tt_hits"] += 1

        if entry.flag == TT_EXACT:
            return entry.score, tt_move

        if entry.flag == TT_LOWER and entry.score >= beta:
            return entry.score, tt_move

        if entry.flag == TT_UPPER and entry.score <= alpha:
            return entry.score, tt_move

    return None, tt_move


def store_tt(board, depth, score, flag, best_move):
    if len(TRANSPOSITION_TABLE) >= MAX_TT_ENTRIES:
        TRANSPOSITION_TABLE.clear()

    key = get_board_key(board)

    old_entry = TRANSPOSITION_TABLE.get(key)

    if old_entry is not None and old_entry.depth > depth:
        return

    TRANSPOSITION_TABLE[key] = TTEntry(
        depth=depth,
        score=int(score),
        flag=flag,
        best_move=best_move,
    )


def get_board_key(board):
    return chess.polyglot.zobrist_hash(board)


def quiescence(board, alpha, beta, deadline=None, q_depth=QUIESCENCE_DEPTH, ply=0):
    check_time(deadline)

    SEARCH_STATS["qnodes"] += 1

    terminal = terminal_score_for_side_to_move(board, ply)

    if terminal is not None:
        return terminal

    if q_depth <= 0:
        return evaluate_for_side_to_move(board)

    if board.is_check():
        best_score = -INFINITY

        for move in order_moves(board, ply):
            check_time(deadline)

            board.push(move)

            try:
                score = -quiescence(
                    board,
                    -beta,
                    -alpha,
                    deadline,
                    q_depth - 1,
                    ply + 1,
                )
            finally:
                board.pop()

            if score > best_score:
                best_score = score

            if best_score > alpha:
                alpha = best_score

            if alpha >= beta:
                SEARCH_STATS["cutoffs"] += 1
                return alpha

        return best_score

    stand_pat = evaluate_for_side_to_move(board)

    if stand_pat >= beta:
        return beta

    if stand_pat > alpha:
        alpha = stand_pat

    for move in order_tactical_moves(board, q_depth):
        check_time(deadline)

        board.push(move)

        try:
            score = -quiescence(
                board,
                -beta,
                -alpha,
                deadline,
                q_depth - 1,
                ply + 1,
            )
        finally:
            board.pop()

        if score >= beta:
            SEARCH_STATS["cutoffs"] += 1
            return beta

        if score > alpha:
            alpha = score

    return alpha


def negamax(board, depth, alpha, beta, deadline=None, ply=0):
    check_time(deadline)

    SEARCH_STATS["nodes"] += 1

    terminal = terminal_score_for_side_to_move(board, ply)

    if terminal is not None:
        return terminal

    if depth == 0:
        return quiescence(board, alpha, beta, deadline, QUIESCENCE_DEPTH, ply)

    alpha_original = alpha
    beta_original = beta

    tt_score, tt_move = probe_tt(board, depth, alpha, beta)

    if tt_score is not None:
        return tt_score

    best_score = -INFINITY
    best_move = None

    for move in order_moves(board, ply, tt_move):
        check_time(deadline)

        quiet_before_push = is_quiet_move(board, move)

        board.push(move)

        try:
            score = -negamax(
                board,
                depth - 1,
                -beta,
                -alpha,
                deadline,
                ply + 1,
            )
        finally:
            board.pop()

        if score > best_score:
            best_score = score
            best_move = move

        if score > alpha:
            alpha = score

        if alpha >= beta:
            SEARCH_STATS["cutoffs"] += 1

            if quiet_before_push:
                add_killer_move(ply, move)
                update_history(move, depth)

            break

    if best_score <= alpha_original:
        flag = TT_UPPER
    elif best_score >= beta_original:
        flag = TT_LOWER
    else:
        flag = TT_EXACT

    store_tt(board, depth, best_score, flag, best_move)

    return best_score


def search_root(board, depth, deadline=None, preferred_move=None):
    best_move = None
    best_score = -INFINITY
    alpha = -INFINITY
    beta = INFINITY

    tt_score, tt_move = probe_tt(board, depth, alpha, beta)

    if preferred_move is not None:
        tt_move = preferred_move

    for move in order_moves(board, 0, tt_move):
        check_time(deadline)

        board.push(move)

        try:
            score = -negamax(
                board,
                depth - 1,
                -beta,
                -alpha,
                deadline,
                1,
            )
        finally:
            board.pop()

        if score > best_score:
            best_score = score
            best_move = move

        if score > alpha:
            alpha = score

    store_tt(board, depth, best_score, TT_EXACT, best_move)

    return best_move, best_score


def find_best_move(board, depth=3):
    reset_search_stats()
    return search_root(board, depth)


def find_best_move_timed(board, time_limit=1.0, max_depth=64):
    reset_search_stats()

    moves = order_moves(board)

    if not moves:
        return None, 0, 0

    best_move = moves[0]
    best_score = 0
    completed_depth = 0

    time_limit = max(0.03, time_limit)
    deadline = time.perf_counter() + time_limit

    for depth in range(1, max_depth + 1):
        try:
            move, score = search_root(
                board,
                depth,
                deadline,
                preferred_move=best_move,
            )
        except SearchTimeout:
            break

        if move is not None:
            best_move = move
            best_score = score
            completed_depth = depth

        if time.perf_counter() >= deadline:
            break

    return best_move, best_score, completed_depth