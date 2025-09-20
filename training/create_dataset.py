#!/usr/bin/env python3
"""
Simple dataset chunking script for NNUE training
Creates chunks of 3M positions each from PGN files in data/ folder
"""

import gzip
import random
import struct
from pathlib import Path

import chess
import chess.pgn


def create_chunked_dataset(chunk_size: int = 3_000_000):
    """
    Create chunked dataset from PGN files in data/ folder
    Memory-efficient streaming approach - processes files incrementally
    """
    chunk_dir = Path("chunked_dataset")
    chunk_dir.mkdir(exist_ok=True)

    # Initialize counters and current chunk
    chunk_num = 0
    current_chunk = []
    total_positions = 0
    total_games = 0

    # Get PGN files
    data_path = Path("data")
    pgn_files = list(data_path.glob("*.pgn"))

    if not pgn_files:
        raise FileNotFoundError(f"No PGN files found in {data_path}/")

    print(f"Found {len(pgn_files)} PGN files")
    print("Processing files with memory-efficient streaming...")

    # Process files one by one
    for file_idx, pgn_file in enumerate(pgn_files):  # Process all files
        print(f"Processing file {file_idx + 1}/{len(pgn_files)}: {pgn_file.name}")
        games_in_file = 0

        with open(pgn_file) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                games_in_file += 1
                total_games += 1

                if total_games % 1000 == 0:
                    print(
                        f"  Processed: {total_games:,} games | {total_positions:,} positions | {chunk_num} chunks saved"
                    )

                # Get game result
                result_str = game.headers.get("Result", "*")
                if result_str == "1-0":
                    result = 1.0
                elif result_str == "0-1":
                    result = 0.0
                elif result_str == "1/2-1/2":
                    result = 0.5
                else:
                    continue

                # Process game moves
                board = game.board()
                move_count = 0

                for move in game.mainline_moves():
                    board.push(move)
                    move_count += 1

                    # Skip opening moves, sample positions
                    if move_count < 12 or random.random() > 0.1:
                        continue

                    # Adjust result for side to move
                    position_result = (
                        (1.0 - result) if board.turn == chess.BLACK else result
                    )

                    # Add position to current chunk
                    fen = board.fen()
                    current_chunk.append((fen, position_result))
                    total_positions += 1

                    # Save chunk when full
                    if len(current_chunk) >= chunk_size:
                        save_chunk(chunk_dir, chunk_num, current_chunk)
                        print(
                            f"    → Saved chunk {chunk_num:04d} ({len(current_chunk):,} positions)"
                        )
                        chunk_num += 1
                        current_chunk = []  # Reset for next chunk

                    # Limit positions per game
                    if move_count > 60:
                        break

        print(f"  File complete: {games_in_file:,} games processed")

    # Save final chunk if it has data
    if current_chunk:
        save_chunk(chunk_dir, chunk_num, current_chunk)
        print(
            f"    → Saved final chunk {chunk_num:04d} ({len(current_chunk):,} positions)"
        )
        chunk_num += 1

    print("\n✅ Dataset creation complete!")
    print(f"   Total games processed: {total_games:,}")
    print(f"   Total positions: {total_positions:,}")
    print(f"   Chunks created: {chunk_num}")
    print(
        f"   Average positions per chunk: {total_positions // chunk_num if chunk_num > 0 else 0:,}"
    )

    return chunk_num


def save_chunk(chunk_dir: Path, chunk_num: int, data: list[tuple[str, float]]):
    """Save chunk as compressed binary file"""
    chunk_file = chunk_dir / f"chunk_{chunk_num:04d}.bin.gz"

    with gzip.open(chunk_file, "wb") as f:
        # Write number of positions
        f.write(struct.pack("<I", len(data)))

        for fen, result in data:
            fen_bytes = fen.encode("utf-8")
            f.write(struct.pack("<H", len(fen_bytes)))  # FEN length
            f.write(fen_bytes)  # FEN string
            f.write(struct.pack("<f", result))  # Result (float32)

    print(f"Saved chunk {chunk_num} with {len(data):,} positions -> {chunk_file}")


def load_chunk(chunk_file: str):
    """Load a chunk file and return list of (fen, result) tuples"""
    positions = []

    with gzip.open(chunk_file, "rb") as f:
        num_positions = struct.unpack("<I", f.read(4))[0]

        for _ in range(num_positions):
            fen_len = struct.unpack("<H", f.read(2))[0]
            fen = f.read(fen_len).decode("utf-8")
            result = struct.unpack("<f", f.read(4))[0]

            positions.append((fen, result))

    return positions


def get_chunk_files():
    """Get list of all chunk files"""
    chunk_dir = Path("chunked_dataset")
    if not chunk_dir.exists():
        return []

    return sorted(chunk_dir.glob("chunk_*.bin.gz"))


if __name__ == "__main__":
    import sys

    chunk_size = 3_000_000 if len(sys.argv) < 2 else int(sys.argv[1])

    try:
        num_chunks = create_chunked_dataset(chunk_size)
        print("\nDataset creation complete!")
        print(f"Created {num_chunks} chunks of up to {chunk_size:,} positions each")

        # Test loading first chunk
        chunk_files = get_chunk_files()
        if chunk_files:
            test_positions = load_chunk(str(chunk_files[0]))
            print(f"Test: loaded {len(test_positions)} positions from first chunk")
            if test_positions:
                print(f"Sample position: {test_positions[0]}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
