import numpy as np
from custom_class import ChessBoard


def pgn_final_position(pgn_moves: str, expected_final: np.ndarray):
    game = ChessBoard(pgn_moves)
    final_board = game.get_board_positions()[-1]

    print(final_board)
    print(game.print_board())

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


def test_third_pgn_final_position():
    pgn_moves = (
        "1. Nf3 d5 2. c4 e6 3. g3 f5 4. Bg2 Nf6 5. O-O c6 6. d3 Bd6 7. b3 O-O 8. Bb2 "
        "Nbd7 9. Nbd2 e5 10. e4 fxe4 11. dxe4 d4 12. Ne1 Nc5 13. b4 Ne6 14. c5 Bc7 "
        "15. Nd3 a5 16. a3 axb4 17. axb4 Rxa1 18. Bxa1 Qe8 19. Kh1 Qg6 20. Qb3 Kh8 "
        "21. f4 exf4 22. gxf4 Ng4 23. h3 Ne3 24. Rf2 Nxf4 25. Bf1 Qg3 26. Rf3 Qh4 "
        "27. Nf2 Ng6 28. Rxf8+ Nxf8 29. Qf7 Qf6 30. Qxf6 gxf6 31. Bxd4 Nxf1 32. "
        "Bxf6+ Kg8 33. Nxf1 Ne6 34. Nd3 Nf4 35. Nxf4 Bxf4 36. Kg2 Be6 37. Bc3 Kf7 "
        "38. Bd2 Be5 39. Ne3 Ke7 40. Ng4 Bxg4 41. hxg4 Ke6 42. Kf3 Bf6 43. Ke3 Bg5+ "
        "44. Kd3 Bf6 45. Bc3 Bg5 46. Bd4 Bf4 47. Kc4 Bc7 48. b5 Bd8 49. bxc6 bxc6 "
        "50. Bc3 Be7 51. e5 Kd7 52. Kd4 Ke6 53. Ke4 Bd8 54. Bd2 Bc7 55. Be3 Bd8 56. "
        "g5 Be7 57. Bf4 Bxc5 58. g6 Be7 59. Bg5 Bxg5 60. g7 h5 61. g8=Q+ Kd7 62. "
        "Qxg5 Kc7 63. Qxh5 Kb6 64. Kd4 c5+ 65. Kc4 Kc6 66. e6 Kd6 67. Qxc5+ Kxe6 68. "
        "Qd5+ Kf6 69. Kd4 Ke7 70. Ke4 Kf6 71. Kf4 Ke7 72. Qc6 Kf7 73. Ke5 1-0"
    )

    expected_final = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 100, 0, 0, 0],
            [0, 0, 9, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -100, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    pgn_final_position(pgn_moves, expected_final)


def test_fourth_pgn_final_position():
    pgn_moves = (
        "1. d4 d5 2. c4 c6 3. Nc3 Nf6 4. Nf3 e6 5. b3 Nbd7 6. Qc2 Be7 7. Bd2 O-O 8. "
        "e4 dxe4 9. Nxe4 Nxe4 10. Qxe4 Nf6 11. Qd3 c5 12. Be2 b6 13. Bc3 Bb7 14. Rd1 "
        "cxd4 15. Bxd4 Bb4+ 16. Bc3 Qxd3 17. Rxd3 Bxc3+ 18. Rxc3 Rfd8 19. O-O Ne4 "
        "20. Re3 Rd6 21. h4 Rad8 22. Ng5 Nd2 23. Rd1 h6 24. Nh3 Nf3+ 25. Bxf3 Rxd1+ "
        "26. Bxd1 Rxd1+ 27. Kh2 Rd2 28. a4 a5 29. Kg3 Kf8 30. Nf4 Ke7 31. Rd3 Rb2 "
        "32. f3 g5 33. hxg5 hxg5 34. Nh3 f6 35. Nf2 f5 36. Nd1 Rc2 37. Nc3 g4 38. "
        "Nb5 gxf3 39. gxf3 e5 40. Nd6 f4+ 41. Kg4 Bc6 42. Kf5 Re2 43. Nc8+ Ke8 44. "
        "Nxb6 e4 45. fxe4 Bxe4+ 46. Kxf4 Bxd3 47. Nd5 Rb2 48. c5 Rxb3 49. c6 Kd8 50. "
        "Ke5 Bc4 51. Nf6 Kc7 52. Kd4 Kxc6 53. Kxc4 Rb4+ 54. Kc3 Rxa4 55. Nh5 Rg4 56. "
        "Nf6 Rh4 57. Ne4 Kb5 58. Nd6+ Kc6 59. Nc4 a4 60. Ne5+ Kb5 61. Nd3 1-0"
    )
    expected_final = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 100, 2, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0, 0, -5],
            [0, -100, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    pgn_final_position(pgn_moves, expected_final)
