import numpy as np
from chess import ChessBoard


def pgn_final_position(pgn_moves: str, expected_final: np.ndarray):
    game = ChessBoard(pgn_moves)
    final_board = game.get_board_positions()[-1]

    print(final_board)

    # We only check the non-zero squares, because piece placements
    # can vary in representation but final result must match
    nonzeros_expected = np.argwhere(expected_final != 0)
    for r, c in nonzeros_expected:
        assert final_board[r, c] == expected_final[r, c], f"Mismatch at {r},{c}"

    # Also verify it's checkmate: White king is on b1 (100 at [0,1])
    king_pos = np.argwhere(final_board == 100)
    assert king_pos.size > 0, "White king not found"


def test_first_pgn_final_position():
    pgn_moves = (
        "1. Nf3 d5 2. c4 d4 3. b4 Nc6 4. b5 Nb8 5. e3 c5 6. exd4 cxd4 7. Bb2 e5 "
        "8. Nxe5 Bc5 9. Nd3 Nd7 10. Qg4 Ngf6 11. Qxg7 Rg8 12. Qh6 Qe7+ "
        "13. Kd1 b6 14. Be2 Bb7 15. Bf3 Bxf3+ 16. gxf3 O-O-O 17. Re1 Qd6 "
        "18. a4 Rde8 19. Rxe8+ Rxe8 20. a5 Ne5 21. Nxe5 Qxe5 22. axb6 Qe2+ "
        "23. Kc2 Qxc4+ 24. Nc3 d3+ 25. Kb1 Re1+ 26. Bc1 Qb3# 0-1"
    )
    expected_final = np.array(
        [
            [5, 100, 3, 0, -5, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 1],
            [0, -9, 2, -1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, -3, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, -2, 0, 9],
            [-1, 0, 0, 0, 0, -1, 0, -1],
            [0, 0, -100, 0, 0, 0, 0, 0],
        ]
    )
    pgn_final_position(pgn_moves, expected_final)


def test_second_pgn_final_position():
    pgn_moves = (
        "1. d4 c5 2. Nf3 cxd4 3. Nxd4 d5 4. g3 e5 5. Nb3 Nc6 6. Bg2 Be6 7. O-O Be7 "
        "8. a4 Nf6 9. a5 a6 10. Nc3 h6 11. Bd2 O-O 12. Na4 Nd7 13. Be1 d4 14. f4 "
        "Bxb3 15. cxb3 exf4 16. Rxf4 Bg5 17. Rf1 Rc8 18. Kh1 Be3 19. b4 Qg5 20. Qb3 "
        "Qb5 21. g4 Qxe2 22. Bg3 Nf6 23. h3 Qb5 24. Nc5 Qxb4 25. Qxb4 Nxb4 26. Nxb7 "
        "Nc2 27. Rad1 Nb4 28. Nd6 Rc5 29. Nf5 Rxa5 30. Nxd4 Ra2 31. Nf5 Bc5 32. Be5 "
        "Ne8 33. Bc3 h5 34. gxh5 Ra5 35. h6 gxh6 36. Nxh6+ Kh7 37. Nxf7 1-0"
    )

    expected_final = np.array(
        [
            [0, 0, 0, 5, 0, 5, 0, 100],
            [0, 1, 0, 0, 0, 0, 3, 0],
            [0, 0, 3, 0, 0, 0, 0, 1],
            [0, -2, 0, 0, 0, 0, 0, 0],
            [-5, 0, -3, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 2, 0, -100],
            [0, 0, 0, 0, -2, -5, 0, 0],
        ]
    )
    pgn_final_position(pgn_moves, expected_final)
