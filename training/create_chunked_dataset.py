#!/usr/bin/env python3
"""
Create dataset in chunks to avoid memory issues.
Process PGN files and save positions in smaller chunks.
"""

import argparse
import logging
import pickle
import struct
from pathlib import Path

from nnue_dataset import ChessPosition, DatasetGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("dataset_creation.log")],
)
logger = logging.getLogger(__name__)


def create_chunked_dataset(
    data_dir: str = "data/",
    chunk_size: int = 100_000,
    max_positions: int | None = None,
    output_dir: str = "chunked_dataset/",
):
    """Create dataset in memory-efficient chunks"""

    logger.info("🚀 === CHUNKED DATASET CREATION ===")
    logger.info(f"📁 Data directory: {data_dir}")
    logger.info(f"📊 Chunk size: {chunk_size:,} positions")
    logger.info(
        f"🎯 Max positions: {max_positions:,}"
        if max_positions
        else "🎯 Max positions: unlimited"
    )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Process PGN files and collect positions in chunks
    chunk_num = 0
    current_chunk = []
    total_processed = 0

    logger.info("🔄 Processing PGN files...")

    # Get all PGN files
    pgn_files = []
    data_path = Path(data_dir)
    for pgn_file in data_path.glob("*.pgn"):
        pgn_files.append(str(pgn_file))

    logger.info(f"📁 Found {len(pgn_files)} PGN files")

    if not pgn_files:
        logger.error("❌ No PGN files found in data directory")
        return 0, 0

    # Create dataset generator
    generator = DatasetGenerator()

    try:
        for pgn_file in pgn_files:
            logger.info(f"🔄 Processing {pgn_file}")
            # Calculate max positions per file, use unlimited if not specified
            if max_positions:
                max_per_file = max_positions // len(pgn_files)
            else:
                max_per_file = None  # No limit per file - process all positions

            positions = generator.generate_from_pgn(
                pgn_file, max_positions=max_per_file or float("inf")
            )

            for position in positions:
                current_chunk.append(position)
                total_processed += 1

                # Log progress
                if total_processed % 10_000 == 0:
                    logger.info(f"📈 Processed {total_processed:,} positions")

                # Save chunk when it reaches the desired size
                if len(current_chunk) >= chunk_size:
                    chunk_num += 1
                    chunk_file = output_path / f"positions_chunk_{chunk_num:03d}.bin"

                    logger.info(
                        f"💾 Saving chunk {chunk_num} with {len(current_chunk):,} positions (binary format)"
                    )
                    _save_binary_chunk(current_chunk, chunk_file)

                    current_chunk = []
                    logger.info(f"✅ Chunk {chunk_num} saved: {chunk_file}")

                # Check if we've reached max positions
                if max_positions and total_processed >= max_positions:
                    logger.info(f"🎯 Reached max positions limit: {max_positions:,}")
                    break

            # Break outer loop if max reached
            if max_positions and total_processed >= max_positions:
                break

    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        raise

    # Save final chunk if it has positions
    if current_chunk:
        chunk_num += 1
        chunk_file = output_path / f"positions_chunk_{chunk_num:03d}.bin"

        logger.info(
            f"💾 Saving final chunk {chunk_num} with {len(current_chunk):,} positions (binary format)"
        )
        _save_binary_chunk(current_chunk, chunk_file)

        logger.info(f"✅ Final chunk {chunk_num} saved: {chunk_file}")

    # Create metadata file
    metadata = {
        "total_positions": total_processed,
        "total_chunks": chunk_num,
        "chunk_size": chunk_size,
        "position_size": ChessPosition.STRUCT_SIZE,
        "chunks": [f"positions_chunk_{i:03d}.bin" for i in range(1, chunk_num + 1)],
    }

    # Save metadata in both formats for compatibility
    metadata_file = output_path / "dataset_metadata.pkl"
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)

    # Also save binary metadata for efficient reading
    binary_metadata_file = output_path / "dataset_metadata.bin"
    with open(binary_metadata_file, "wb") as f:
        f.write(struct.pack('<IIII',
            metadata['total_chunks'],
            metadata['total_positions'],
            metadata['chunk_size'],
            metadata['position_size']
        ))

    logger.info("🎉 === DATASET CREATION COMPLETED ===")
    logger.info(f"📊 Total positions: {total_processed:,}")
    logger.info(f"📦 Total chunks: {chunk_num}")
    logger.info(f"📁 Output directory: {output_path}")
    logger.info(f"📋 Metadata saved: {metadata_file}")

    return total_processed, chunk_num


def _save_binary_chunk(positions: list[ChessPosition], chunk_file: Path):
    """Save chunk in efficient binary format"""
    with open(chunk_file, 'wb') as f:
        # Write header: position count, position size
        header = struct.pack('<II', len(positions), ChessPosition.STRUCT_SIZE)
        f.write(header)

        # Write positions
        for pos in positions:
            f.write(pos.to_bytes())


def _load_binary_chunk(chunk_file: Path) -> list[ChessPosition]:
    """Load chunk from binary format"""
    positions = []

    with open(chunk_file, 'rb') as f:
        # Read header
        header = f.read(8)
        if len(header) != 8:
            return positions

        position_count, position_size = struct.unpack('<II', header)

        # Read positions
        for _ in range(position_count):
            data = f.read(position_size)
            if len(data) != position_size:
                break
            positions.append(ChessPosition.from_bytes(data))

    return positions


def load_chunk_files(chunk_dir: str) -> list[ChessPosition]:
    """Load all position chunks into a single list (for small datasets)"""

    chunk_path = Path(chunk_dir)
    metadata_file = chunk_path / "dataset_metadata.pkl"

    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)

    logger.info(
        f"📊 Loading {metadata['total_chunks']} chunks with {metadata['total_positions']:,} positions"
    )

    all_positions = []
    for chunk_file in metadata["chunks"]:
        chunk_path_full = chunk_path / chunk_file
        logger.info(f"🔄 Loading {chunk_path_full}")

        # Check if it's binary or pickle format
        if chunk_file.endswith('.bin'):
            chunk_positions = _load_binary_chunk(chunk_path_full)
        else:
            with open(chunk_path_full, "rb") as f:
                chunk_positions = pickle.load(f)

        all_positions.extend(chunk_positions)

    logger.info(f"✅ Loaded {len(all_positions):,} total positions")
    return all_positions


def load_binary_metadata(chunk_dir: str) -> dict:
    """Load metadata from binary format for efficient access"""
    chunk_path = Path(chunk_dir)
    binary_metadata_file = chunk_path / "dataset_metadata.bin"

    if binary_metadata_file.exists():
        with open(binary_metadata_file, 'rb') as f:
            data = f.read(16)  # 4 integers
            return {
                'total_chunks': struct.unpack('<I', data[0:4])[0],
                'total_positions': struct.unpack('<I', data[4:8])[0],
                'chunk_size': struct.unpack('<I', data[8:12])[0],
                'position_size': struct.unpack('<I', data[12:16])[0]
            }
    else:
        # Fallback to pickle metadata
        metadata_file = chunk_path / "dataset_metadata.pkl"
        with open(metadata_file, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create chunked chess dataset")
    parser.add_argument(
        "--data-dir", default="data/", help="Directory containing PGN files"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=2_000_000, help="Positions per chunk"
    )
    parser.add_argument(
        "--max-positions", type=int, help="Maximum positions to process"
    )
    parser.add_argument(
        "--output-dir", default="chunked_dataset/", help="Output directory"
    )

    args = parser.parse_args()

    try:
        create_chunked_dataset(
            args.data_dir, args.chunk_size, args.max_positions, args.output_dir
        )
    except KeyboardInterrupt:
        logger.info("⏹️  Dataset creation interrupted by user")
    except Exception as e:
        logger.error(f"❌ Dataset creation failed: {e}")
        raise
