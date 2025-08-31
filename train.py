import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess.pgn
import numpy as np
import os

# -------------------------
# 1. Encode chess board with extra channels
# -------------------------
PIECE_TO_IDX = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}


def encode_board(board):
    planes = np.zeros(
        (18, 64), dtype=np.float32
    )  # 12 piece + 4 castling + 1 en passant + 1 halfmove
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        planes[PIECE_TO_IDX[piece.symbol()], square] = 1.0

    # Castling rights planes
    planes[12, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[13, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[14, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[15, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # En passant possible plane
    planes[16, :] = 1.0 if board.ep_square else 0.0

    # Halfmove clock plane (normalize by 100)
    planes[17, :] = board.halfmove_clock / 100.0

    return planes.flatten()  # shape (18*64=1152,)


# -------------------------
# 2. Dataset
# -------------------------
class ChessDataset(Dataset):
    def __init__(self, positions, labels):
        self.positions = positions
        self.labels = labels

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return torch.tensor(self.positions[idx], dtype=torch.float32), torch.tensor(
            self.labels[idx], dtype=torch.float32
        )


# -------------------------
# 3. MLP model for max speed
# -------------------------
class EvalNet(nn.Module):
    def __init__(self, input_size=1152):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.residual_fc = nn.Linear(512, 256)  # projects for residual
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        # Initialization
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.kaiming_uniform_(self.residual_fc.weight, nonlinearity="relu")

    def forward(self, x):
        x1 = self.relu(self.fc1(x))
        x1 = self.dropout(x1)

        # Residual connection with projection
        x2 = self.relu(self.fc2(x1) + self.residual_fc(x1))
        x2 = self.dropout(x2)

        x3 = self.relu(self.fc3(x2))
        x3 = self.dropout(x3)

        out = torch.tanh(self.fc4(x3))
        return out


# -------------------------
# 4. Training loop
# -------------------------
def train_model(train_loader, val_loader, model, epochs=50, lr=1e-4, device="mps"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.7
    )  # decay every 20 epochs

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.float16):
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0
        val_mae = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()
                val_loss += criterion(outputs, batch_y).item()
                val_mae += torch.mean(torch.abs(outputs - batch_y)).item()
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)

        print(
            f"Epoch {epoch + 1:02d}: Train Loss={train_loss:.4f}, "
            f"Val Loss={val_loss:.4f}, Val MAE={val_mae:.4f}"
        )

        scheduler.step()


# -------------------------
# 5. PGN parser pipeline -> .npz files
# -------------------------
def parse_pgn_folder(pgn_folder, output_file):
    positions = []
    labels = []

    # 1M games to prevent
    max_games = 1000000

    for filename in os.listdir(pgn_folder):
        if not filename.endswith(".pgn"):
            continue
        filepath = os.path.join(pgn_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                board = game.board()
                result = game.headers.get("Result", "1/2-1/2")
                if result == "1-0":
                    outcome = 1.0
                elif result == "0-1":
                    outcome = -1.0
                else:
                    outcome = 0.0

                for move in game.mainline_moves():
                    board.push(move)
                    positions.append(encode_board(board))
                    labels.append(outcome)
                    if len(positions) >= max_games:
                        break
                if len(positions) >= max_games:
                    break

    np.savez_compressed(
        output_file, positions=np.array(positions), labels=np.array(labels)
    )
    print(f"Saved {len(positions)} positions to {output_file}")


# -------------------------
# 6. Fast inference using torch.jit
# -------------------------
def fast_eval(model, board, device="cpu"):
    model.to(device)
    x = torch.tensor(encode_board(board), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad(), torch.inference_mode():
        return float(model(x).item())


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Parse PGNs into .npz dataset
    # parse_pgn_folder("data", "chess_dataset.npz")

    # Load dataset
    data = np.load("chess_dataset.npz")
    positions = data["positions"]
    labels = data["labels"]

    # Shuffle indices
    num_samples = len(positions)
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    # 80% train, 20% validation
    split_idx = int(num_samples * 0.8)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_dataset = ChessDataset(positions[train_indices], labels[train_indices])
    val_dataset = ChessDataset(positions[val_indices], labels[val_indices])

    train_loader = DataLoader(train_dataset, batch_size=2024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2024, shuffle=False)

    # Train small MLP
    model = EvalNet()
    train_model(train_loader, val_loader, model, epochs=100, lr=1e-4, device="mps")

    # Save model and JIT
    torch.save(model.state_dict(), "chess_eval_net.pth")
    example_input = torch.randn(1, 1152).to("mps")
    jit_model = torch.jit.trace(model, example_input)
    jit_model.save("chess_eval_net_jit.pt")
    print("Model saved and JIT compiled.")
