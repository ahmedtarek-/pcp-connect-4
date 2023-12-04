def is_valid_move(board, col):
    return board[0, col] == 0

def drop_piece(board, col, player):
    row = np.argmax(board[:, col] == 0)
    board[row, col] = player

def is_terminal_node(board):
    return is_winner(board, 1) or is_winner(board, 2) or not (board == 0).any()

def evaluate_window(window, player):
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

def evaluate_board(board, player):
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

    # Score diagonal (from top-left to bottom-right)
    for row in range(board.shape[0] - 3):
        for col in range(board.shape[1] - 3):
            window = board[row:row + 4, col:col + 4].diagonal()
            score += evaluate_window(window, player)

    # Score diagonal (from top-right to bottom-left)
    for row in range(board.shape[0] - 3):
        for col in range(3, board.shape[1]):
            window = np.fliplr(board[row:row + 4, col - 3:col + 1]).diagonal()
            score += evaluate_window(window, player)

    return score

# depth=4, alpha=-np.Inf, beta=np.Inf, maximizing_player=True
def minimax(board, depth, alpha, beta, maximizing_player):
    valid_moves = [col for col in range(board.shape[1]) if is_valid_move(board, col)]

    if depth == 0 or is_terminal_node(board):
        return None, evaluate_board(board, 2)

    if maximizing_player:
        value = -np.Inf
        best_move = valid_moves[0]

        for col in valid_moves:
            temp_board = np.copy(board)
            drop_piece(temp_board, col, 2)
            new_score = minimax(temp_board, depth - 1, alpha, beta, False)[1]

            if new_score > value:
                value = new_score
                best_move = col

            alpha = max(alpha, value)
            if alpha >= beta:
                break

        return best_move, value

    else:
        value = np.Inf
        best_move = valid_moves[0]

        for col in valid_moves:
            temp_board = np.copy(board)
            drop_piece(temp_board, col, 1)
            new_score = minimax(temp_board, depth - 1, alpha, beta, True)[1]

            if new_score < value:
                value = new_score
                best_move = col

            beta = min(beta, value)
            if alpha >= beta:
                break

        return best_move, value
