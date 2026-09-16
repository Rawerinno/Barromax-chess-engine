import json
import os
import random
from pathlib import Path

import chess
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from ai_model import BoardValueNet, board_to_tensor, INPUT_PLANES


DATA_PATH = Path("data/selfplay_positions.jsonl")
MODEL_PATH = Path("barromax_ai.pt")

EPOCHS = 8
BATCH_SIZE = 128
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.10
MAX_POSITIONS = 200_000


class ChessPositionDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        board = chess.Board(row["fen"])
        x = board_to_tensor(board)
        y = float(row["target"])

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y


def load_rows():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Não encontrei o ficheiro: {DATA_PATH}")

    rows = []

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "fen" not in row or "target" not in row:
                continue

            rows.append(row)

    random.shuffle(rows)

    if len(rows) > MAX_POSITIONS:
        rows = rows[:MAX_POSITIONS]

    return rows


def train():
    rows = load_rows()

    if not rows:
        print("Não há dados para treinar.")
        return

    print(f"Posições carregadas: {len(rows)}")

    dataset = ChessPositionDataset(rows)

    validation_size = max(1, int(len(dataset) * VALIDATION_SPLIT))
    train_size = len(dataset) - validation_size

    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Dispositivo: {device}")

    model = BoardValueNet().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    loss_function = torch.nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss_total = 0.0
        train_batches = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            prediction = model(x)
            loss = loss_function(prediction, y)

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            train_batches += 1

        train_loss = train_loss_total / max(1, train_batches)

        model.eval()

        validation_loss_total = 0.0
        validation_batches = 0

        with torch.no_grad():
            for x, y in validation_loader:
                x = x.to(device)
                y = y.to(device)

                prediction = model(x)
                loss = loss_function(prediction, y)

                validation_loss_total += loss.item()
                validation_batches += 1

        validation_loss = validation_loss_total / max(1, validation_batches)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train loss: {train_loss:.5f} | "
            f"val loss: {validation_loss:.5f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_planes": INPUT_PLANES,
        },
        MODEL_PATH,
    )

    print()
    print(f"Modelo guardado em: {MODEL_PATH}")


if __name__ == "__main__":
    train()