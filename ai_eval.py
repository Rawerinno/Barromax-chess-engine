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


def evaluate_ai_cp(board, scale=600):
    """
    Avaliação neural em centipawns.

    A rede devolve valor entre -1 e +1:
    +1 = bom para brancas
    -1 = bom para pretas

    Depois convertemos para centipawns:
    +1 -> +600
    -1 -> -600
    """

    model = load_ai_model()

    if model is None:
        return 0

    key = chess.polyglot.zobrist_hash(board)

    if key in _AI_CACHE:
        return _AI_CACHE[key]

    x = board_to_tensor(board)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        value = float(model(x).item())

    score = int(value * scale)

    if len(_AI_CACHE) >= _MAX_CACHE_SIZE:
        _AI_CACHE.clear()

    _AI_CACHE[key] = score

    return score