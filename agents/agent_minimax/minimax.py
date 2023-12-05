import numpy as np

from game_utils import (
    BoardPiece, PlayerAction,
    SavedState, valid_columns,
    connected_four, apply_player_action,
    PLAYER1, PLAYER2)
from typing import Optional, Callable


def is_last_step(board: np.ndarray) -> bool:
    """
    Returns whether a given step is last step of the game or no.
    """
    return connected_four(board, 1) or connected_four(board, 2) or not (board == 0).any()

def evaluate_window(window: np.ndarray, player: BoardPiece) -> int:
    """
    Returns a score that is calculated for the given window
    """
    opponent = 3 - player
    score = 0

    if np.count_nonzero(window == player) == 4:
        score += 100
    elif np.count_nonzero(window == player) == 3 and np.count_nonzero(window == 0) == 1:
        score += 5
    elif np.count_nonzero(window == player) == 2 and np.count_nonzero(window == 0) == 2:
        score += 2

    if np.count_nonzero(window == opponent) == 3 and np.count_nonzero(window == 0) == 1:
        score -= 4

    return score

def evaluate_board(board: np.ndarray, player: BoardPiece) -> int:
    """
    Returns a total score based on scanning the board and calcuating
    partial scores for each window (vertical, horizontal and diangonal)
    """
    score = 0

    # Score center column
    center_array = board[:, 3]
    center_count = np.count_nonzero(center_array == player)
    score += center_count * 3

    # Score horizontal
    for row in range(board.shape[0]):
        for col in range(board.shape[1] - 3):
            window = board[row, col:col + 4]
            score += evaluate_window(window, player)

    # Score vertical
    for col in range(board.shape[1]):
        for row in range(board.shape[0] - 3):
            window = board[row:row + 4, col]
            score += evaluate_window(window, player)

    # Score diagonal (top left -> bottom right)
    for row in range(board.shape[0] - 3):
        for col in range(board.shape[1] - 3):
            window = board[row:row + 4, col:col + 4].diagonal()
            score += evaluate_window(window, player)

    # Score diagonal (top right -> bottom left)
    for row in range(board.shape[0] - 3):
        for col in range(3, board.shape[1]):
            window = np.fliplr(board[row:row + 4, col - 3:col + 1]).diagonal()
            score += evaluate_window(window, player)

    return score

def get_opponent(player: BoardPiece) -> BoardPiece:
    """
    Returns the opponent.
    """

    opponent = PLAYER2 if (player == PLAYER1) else PLAYER1
    return opponent

def run_minmax(
    board: np.ndarray,
    player: BoardPiece,
    depth: int = 4,
    alpha: float = -np.Inf,
    beta: float = np.Inf,
    maximizing_player: bool = True
):
    """
    Returns result of running minmax algorithm.

    maximizing_player is a choice of which player to start with
    """
    # 1. Get all columns where action is possible
    current_valid_columns = valid_columns(board)

    # 2. Determine player and opponent
    opponent = get_opponent(player)

    if depth == 0 or is_last_step(board):
        return None, evaluate_board(board, 2)

    # 3. Go over the valid columns, calculate score and apply minmax recursivly
    if maximizing_player:
        value = -np.Inf
        best_action = current_valid_columns[0]

        for valid_column in current_valid_columns:
            temp_board = np.copy(board)
            apply_player_action(board, valid_column, player)
            new_score = run_minmax(temp_board, player, depth - 1, alpha, beta, False)[1]

            if new_score > value:
                value = new_score
                best_action = valid_column

            alpha = max(alpha, value)
            if alpha >= beta:
                break

        return best_action, value

    else:
        value = np.Inf
        best_action = current_valid_columns[0]

        for valid_column in current_valid_columns:
            temp_board = np.copy(board)
            apply_player_action(board, valid_column, opponent)
            new_score = run_minmax(temp_board, opponent, depth - 1, alpha, beta, True)[1]

            if new_score < value:
                value = new_score
                best_action = valid_column

            beta = min(beta, value)
            if alpha >= beta:
                break

        return best_action, value


def generate_move_minmax(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> tuple[PlayerAction, Optional[SavedState]]:
    """
    Finds the best move for a player using the minmax algorithm.

    Args:
        board (np.ndarray): The game board represented as an ndarray.
        player (BoardPiece): The current player.
        saved_state (Optional[SavedState]): Optional saved state.

    Returns:
        tuple[PlayerAction, Optional[SavedState]]: The best move and the optional saved state.

    """
    minmax_action = run_minmax(board, player, 4, float('-inf'), float('inf'), True)[0]
    print("Action: ", minmax_action)

    minmax_action = PlayerAction(minmax_action)
    return minmax_action, saved_state
