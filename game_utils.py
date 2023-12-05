from enum import Enum
import numpy as np
from typing import Callable, Optional

BOARD_COLS = 7
BOARD_ROWS = 6
BOARD_SHAPE = (6, 7)
INDEX_HIGHEST_ROW = BOARD_ROWS - 1
INDEX_LOWEST_ROW = 0

BoardPiece = np.int8  # The data type (dtype) of the board pieces
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiecePrint = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint('X')
PLAYER2_PRINT = BoardPiecePrint('O')

PlayerAction = np.int8  # The column to be played

class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0

class MoveStatus(Enum):
    IS_VALID = 1
    WRONG_TYPE = 'Input is not a number.'
    NOT_INTEGER = ('Input is not an integer, or isn\'t equal to an integer in '
                   'value.')
    OUT_OF_BOUNDS = 'Input is out of bounds.'
    FULL_COLUMN = 'Selected column is full.'

class SavedState:
    pass

GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]

def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape BOARD_SHAPE and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    board = np.zeros(shape=BOARD_SHAPE)
    board = board.astype(BoardPiece)
    return board

def player_symbol(board_piece: BoardPiece) -> str:
    """
    Returns a string -> X if Player 1, 0 if Player 2, '' if NO_PLAYER
    """
    if board_piece == PLAYER1:
        player_symbol = "X"
    elif board_piece == PLAYER2:
        player_symbol = "0"
    else:
        player_symbol = " "
    player_symbol += " "

    return player_symbol

def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] of the array should appear in the lower-left in the printed string representation. Here's an example output, note that we use
    PLAYER1_Print to represent PLAYER1 and PLAYER2_Print to represent PLAYER2):
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """
    separator = "|==============|\n"
    end_str = "|0 1 2 3 4 5 6 |\n"

    board_rows = []
    for i in range(board.shape[0]):
        board_row = "|"
        for j in range(board.shape[1]):
            board_row += player_symbol(board[i][j])
        board_row += "|\n"
        board_rows.append(board_row)

    # Adding |==============|
    str_to_print = separator 
    
    # Adding the inverted rows
    for row in board_rows[::-1]:
        str_to_print += row

    # Adding |==============|
    str_to_print += separator

    # Adding |0 1 2 3 4 5 6 |
    str_to_print += end_str

    return str_to_print


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """
    raise NotImplementedError()


def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece):
    """
    Sets board[i, action] = player, where i is the lowest open row. The input 
    board should be modified in place, such that it's not necessary to return 
    something.
    """
    lowest_open_row = board.T[action].tolist().index(0)
    board[lowest_open_row][action] = player
    return True


def connected_four(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    """
    connected = False

    # Checking horizontal rows
    if any(
        (np.all(row[col:col + 4] == player) for row in board for col in range(board.shape[0] - 3))
    ):
        connected = True

    # Check vertical columns
    if any(
        (np.all(board[row:row + 4, col] == player) for col in range(board.shape[1]) for row in range(board.shape[0] - 3))
    ):
        connected = True

    # Check diagonal (top left -> bottom right)
    for row in range(len(board) - 3):
        for col in range(len(board[0]) - 3):
            if all(board[row + i][col + i] == player for i in range(4)):
                connected = True

    # Check diagonal (top right -> bottom left)
    for row in range(len(board) - 3):
        for col in range(3, len(board[0])):
            if all(board[row + i][col - i] == player for i in range(4)):
                connected = True


    return connected


def check_end_state(board: np.ndarray, player: BoardPiece) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    verdict = GameState.STILL_PLAYING
    if connected_four(board, player):
        verdict = GameState.IS_WIN
    elif np.all(board == NO_PLAYER):
        verdict = GameState.IS_DRAW
    return verdict


def valid_columns(board: np.ndarray) -> np.ndarray:
    """
    Returns the valid columns for a given board
    """
    valid_columns = np.where(~board.all(axis=0))[0]
    return valid_columns

def check_move_status(board: np.ndarray, column: float) -> MoveStatus:
    """
    Returns a MoveStatus indicating whether a move is legal or illegal, and why 
    the move is illegal.
    Any column type is accepted, but it needs to be convertible to a number
    and must result in a whole number.
    Furthermore, the column must be within the bounds of the board and the
    column must not be full.
    """
    try:
        numeric_column = float(column)
    except ValueError:
        return MoveStatus.WRONG_TYPE

    is_integer = np.mod(numeric_column, 1) == 0
    if not is_integer:
        return MoveStatus.NOT_INTEGER

    column = PlayerAction(column)
    is_in_range = PlayerAction(0) <= column <= PlayerAction(6)
    if not is_in_range:
        return MoveStatus.OUT_OF_BOUNDS

    is_open = board[-1, column] == NO_PLAYER
    if not is_open:
        return MoveStatus.FULL_COLUMN

    return MoveStatus.IS_VALID
