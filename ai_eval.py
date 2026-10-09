from pathlib import Path

import chess
import chess.polyglot
import torch

from ai_model import BoardValueNet, board_to_tensor


MODEL_PATH = Path(__file__).with_name("barromax_ai.pt")

_AI_MODEL = None
_AI_CACHE = {}
_MAX_CACHE_SIZE = 100_000


def clear_ai_cache():
    _AI_CACHE.clear()


def load_ai_model():
    global _AI_MODEL

    if _AI_MODEL is not None:
        return _AI_MODEL

    if not MODEL_PATH.exists():
        return None

    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")

        model = BoardValueNet()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        _AI_MODEL = model
        return _AI_MODEL

    except Exception as error:
        print(f"info string erro ao carregar AI: {error}", flush=True)
        return None


def _ai_cache_key(board, scale):
    # O tensor inclui fullmove_number, por isso o cache também tem de o incluir.
    return (
        chess.polyglot.zobrist_hash(board),
        board.fullmove_number,
        int(scale),
    )


def evaluate_ai_batch_cp(boards, scale=600):
    """
    Avalia várias posições numa única passagem pela rede.

    A rede devolve valor entre -1 e +1:
    +1 = bom para brancas
    -1 = bom para pretas

    O resultado é convertido para centipawns.
    """
    if not boards:
        return []

    model = load_ai_model()

    if model is None:
        return [0] * len(boards)

    scores = [None] * len(boards)
    missing_indices = []
    missing_boards = []
    missing_keys = []

    for index, board in enumerate(boards):
        key = _ai_cache_key(board, scale)

        if key in _AI_CACHE:
            scores[index] = _AI_CACHE[key]
        else:
            missing_indices.append(index)
            missing_boards.append(board)
            missing_keys.append(key)

    if missing_boards:
        batch = torch.stack(
            [
                torch.from_numpy(board_to_tensor(board))
                for board in missing_boards
            ],
            dim=0,
        ).float()

        with torch.inference_mode():
            values = model(batch).cpu().tolist()

        if len(_AI_CACHE) + len(missing_boards) >= _MAX_CACHE_SIZE:
            _AI_CACHE.clear()

        for index, key, value in zip(
            missing_indices,
            missing_keys,
            values,
        ):
            score = int(float(value) * scale)
            scores[index] = score
            _AI_CACHE[key] = score

    return scores


def evaluate_ai_cp(board, scale=600):
    """Avaliação neural de uma única posição em centipawns."""
    return evaluate_ai_batch_cp([board], scale=scale)[0]
