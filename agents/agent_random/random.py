import numpy as np

from game_utils import BoardPiece, PlayerAction, SavedState, NO_PLAYER
from typing import Optional, Callable

def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> tuple[PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column randomly and return it as `action`
    
    # 1. Get all columns where action is possible
    valid_columns = np.where(~board.all(axis=0))[0]

    # 2. Select one randomly
    random_action = np.random.choice(valid_columns, size=1)[0]

    # 3. Return this column as action
    random_action = PlayerAction(random_action)
    return random_action, saved_state
