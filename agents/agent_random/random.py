import numpy as np

from game_utils import BoardPiece, PlayerAction, SavedState, NO_PLAYER, valid_columns
from typing import Optional, Callable

def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> tuple[PlayerAction, Optional[SavedState]]:
    """
    Returns a valid, non-full column randomly and return it as action
    """
    
    # 1. Get all columns where action is possible
    valid_columns = valid_columns()

    # 2. Select one randomly
    random_action = np.random.choice(valid_columns, size=1)[0]

    # 3. Return this column as action
    random_action = PlayerAction(random_action)
    return random_action, saved_state
