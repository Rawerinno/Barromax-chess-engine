import chess
import numpy as np
import torch
import torch.nn as nn


INPUT_PLANES = 18


PIECE_TO_PLANE = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,

    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}


def board_to_tensor(board):
    """
    Converte uma posição de xadrez num tensor numérico.

    Formato:
    18 planos de 8x8.

    0-11: peças
    12: lado a jogar
    13: roque branco lado rei
    14: roque branco lado dama
    15: roque preto lado rei
    16: roque preto lado dama
    17: número do lance normalizado
    """

    x = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        x[plane, rank, file] = 1.0

    if board.turn == chess.WHITE:
        x[12, :, :] = 1.0
    else:
        x[12, :, :] = 0.0

    if board.has_kingside_castling_rights(chess.WHITE):
        x[13, :, :] = 1.0

    if board.has_queenside_castling_rights(chess.WHITE):
        x[14, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.BLACK):
        x[15, :, :] = 1.0

    if board.has_queenside_castling_rights(chess.BLACK):
        x[16, :, :] = 1.0

    move_number = min(board.fullmove_number, 100) / 100.0
    x[17, :, :] = move_number

    return x


class BoardValueNet(nn.Module):
    """
    Rede neural simples que avalia uma posição.

    Output:
    -1.0 = muito bom para pretas
     0.0 = equilibrado
    +1.0 = muito bom para brancas
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x.squeeze(-1)