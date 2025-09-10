#!/usr/bin/env python3
"""
Lightweight Chess AI Endpoint for Raspberry Pi using FastAPI
Fast, efficient endpoint that returns AI moves using deterministic evaluation
"""

import os
import sys

import chess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add the project root to Python path
sys.path.append(os.path.dirname(__file__))

try:
    from pure_cpp_chess_ai import PureCppChessAI

    CPP_AVAILABLE = True
except ImportError:
    print(
        "Warning: C++ engine not available. Install with: cd cpp_search && python setup_pure.py build_ext --inplace"
    )
    CPP_AVAILABLE = False

app = FastAPI(
    title="Chess AI API",
    description="Lightweight chess AI endpoint for Raspberry Pi",
    version="1.0.0",
)

# Initialize AI with deterministic evaluation for Pi efficiency
ai_engine = None


def array_to_board(board_array: list[list[str]], turn: str) -> chess.Board:
    """
    Convert 8x8 board array directly to python-chess Board

    Board pieces should use standard notation:
    - White: K, Q, R, B, N, P
    - Black: k, q, r, b, n, p
    - Empty squares: "" or " " or None
    """
    board = chess.Board(fen=None)  # Create empty board
    board.clear()  # Clear all pieces

    # Place pieces from array (row 0 = rank 8, row 7 = rank 1)
    for row_idx, row in enumerate(board_array):
        rank = 8 - row_idx  # Convert to chess rank (8 to 1)
        for col_idx, piece_str in enumerate(row):
            if piece_str and piece_str.strip():
                file = col_idx  # File a-h (0-7)
                square = chess.square(file, rank - 1)  # Convert to square index

                # Parse piece
                piece_char = piece_str.strip()
                try:
                    piece = chess.Piece.from_symbol(piece_char)
                    board.set_piece_at(square, piece)
                except ValueError:
                    continue  # Skip invalid pieces

    # Set turn
    board.turn = chess.WHITE if turn.lower() == "white" else chess.BLACK

    return board


class MoveRequest(BaseModel):
    board: list[list[str]] = Field(
        ..., description="8x8 board array with piece notation"
    )
    turn: str = Field(..., description="Whose turn: 'white' or 'black'")
    depth: int = Field(5, ge=1, le=8, description="Search depth (1-8)")


class MoveResponse(BaseModel):
    move: str = Field(..., description="Best move in UCI notation")
    score: float = Field(..., description="Position evaluation score")
    status: str = Field(..., description="Response status")
    evaluation_mode: str = Field(..., description="AI evaluation mode used")


class ErrorResponse(BaseModel):
    error: str
    status: str


class HealthResponse(BaseModel):
    status: str
    ai_initialized: bool
    cpp_available: bool


class InfoResponse(BaseModel):
    evaluation_mode: str
    cpp_available: bool
    status: str


@app.on_event("startup")
async def startup_event():
    """Initialize the AI engine on startup"""
    global ai_engine
    if CPP_AVAILABLE:
        ai_engine = PureCppChessAI(evaluation_mode="deterministic")
        print("✅ Chess AI initialized with deterministic evaluation")
    else:
        print("❌ Chess AI not available - C++ engine required")


@app.post(
    "/move",
    response_model=MoveResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def get_move(request: MoveRequest):
    """
    Get AI move for a given board position
    """
    if not ai_engine:
        raise HTTPException(status_code=500, detail="AI engine not initialized")

    try:
        # Validate board array and create board
        try:
            if len(request.board) != 8 or not all(
                len(row) == 8 for row in request.board
            ):
                raise ValueError("Board must be 8x8 array")

            if request.turn.lower() not in ["white", "black"]:
                raise ValueError("Turn must be 'white' or 'black'")

            board = array_to_board(request.board, request.turn)
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid board: {str(e)}"
            ) from e

        # Check if game is over
        if board.is_game_over():
            raise HTTPException(
                status_code=400, detail=f"Game is over. Result: {board.result()}"
            )

        # Get best move from AI
        best_move, score = ai_engine.get_best_move(board, depth=request.depth)

        if best_move is None:
            raise HTTPException(status_code=400, detail="No legal moves available")

        return MoveResponse(
            move=best_move.uci(),
            score=float(score),
            status="success",
            evaluation_mode=ai_engine.get_evaluation_mode(),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}") from e


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        ai_initialized=ai_engine is not None,
        cpp_available=CPP_AVAILABLE,
    )


@app.get(
    "/info", response_model=InfoResponse, responses={500: {"model": ErrorResponse}}
)
async def get_info():
    """Get engine information"""
    if not ai_engine:
        raise HTTPException(status_code=500, detail="AI engine not initialized")

    return InfoResponse(
        evaluation_mode=ai_engine.get_evaluation_mode(),
        cpp_available=CPP_AVAILABLE,
        status="ready",
    )


if __name__ == "__main__":
    import uvicorn

    print("Starting Chess AI FastAPI Endpoint...")

    # Run with lightweight configuration suitable for Pi
    uvicorn.run(
        app,
        host="0.0.0.0",  # Allow external connections
        port=8080,
        workers=1,  # Single worker for Pi
        loop="asyncio",  # Use asyncio event loop
    )
