#!/usr/bin/env python3
"""
DQN Player evaluation script with clean, professional output.
No redundant naming or unnecessary qualifiers.
"""

import numpy as np
import gamerules
import time
import pickle
import os

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class TestPlayer(gamerules.Player):
    """Professional DQN player for evaluation"""

    def __init__(self, name, weights_file=None):
        super().__init__(name)
        self.name = name

        # Import the player components
        try:
            from player import CustomDQN, HeuristicEngine

            # Initialize DQN network
            self.q_network = CustomDQN(
                input_size=200,
                hidden_sizes=[256, 128, 64],
                output_size=7,
                learning_rate=0.0001
            )

            # Initialize heuristics
            self.heuristics = HeuristicEngine()

        except ImportError as e:
            print(f"Error importing components: {e}")
            raise

        # Load weights if provided
        if weights_file and os.path.exists(weights_file):
            if self.q_network.load_weights(weights_file):
                print(f"✅ Loaded weights from {weights_file}")
            else:
                print("⚠️ Failed to load weights, using random initialization")
        else:
            print("ℹ️ No weights file provided, using random initialization")

    def getName(self):
        return self.name

    def newGame(self, new_opponent):
        pass

    def getAction(self, board, startValue):
        """Hybrid decision system: heuristics + DQN"""
        possibleActions = self.getPossibleActions(board.board)

        if len(possibleActions) == 0:
            return 0

        try:
            # Critical moves (always override)
            action, _ = self.heuristics.find_immediate_win(board, startValue)
            if action is not None:
                return action

            action, _ = self.heuristics.find_must_block(board, startValue)
            if action is not None:
                return action

            # Tactical moves (high priority)
            action, _ = self.heuristics.find_winning_threat(board, startValue)
            if action is not None:
                return action

            action, _ = self.heuristics.find_dangerous_block(board, startValue)
            if action is not None:
                return action

            # Strategic DQN decision
            state = self._encode_state_enhanced(board, startValue)
            q_values = self.q_network.predict(state)[0]

            # Mask invalid actions
            q_values_masked = q_values.copy()
            for i in range(7):
                if i not in possibleActions:
                    q_values_masked[i] = float('-inf')

            action = np.argmax(q_values_masked)
            return int(action) if action in possibleActions else possibleActions[0]

        except Exception as e:
            print(f"Error in decision system: {e}")
            # Fallback: center preference
            center_preferences = [3, 2, 4, 1, 5, 0, 6]
            for col in center_preferences:
                if col in possibleActions:
                    return col
            return possibleActions[0]

    def _encode_state_enhanced(self, board, startValue):
        """Enhanced state encoding with comprehensive features"""
        features = []

        # Board state (42 features)
        board_normalized = board.board * startValue
        features.extend(board_normalized.flatten())

        # Component analysis (84 features)
        components_normalized = np.sign(board.components) * startValue
        features.extend(components_normalized.flatten())

        components4_normalized = np.sign(board.components4) * startValue
        features.extend(components4_normalized.flatten())

        # Strategic features (74 features)
        strategic_features = self._extract_strategic_features(board, startValue)
        features.extend(strategic_features)

        return np.array(features, dtype=np.float32)

    def _extract_strategic_features(self, board, startValue):
        """Extract comprehensive strategic features"""
        features = []

        # Basic metrics (16 features)
        column_heights = []
        for col in range(7):
            height = 6 - len(np.where(board.board[:, col] == 0)[0])
            column_heights.append(height)
            features.append(height / 6.0)

        center_control = sum(np.sum(board.board[:, col] == startValue) for col in [2, 3, 4])
        features.append(center_control / 18.0)

        edge_control = sum(np.sum(board.board[:, col] == startValue) for col in [0, 6])
        features.append(edge_control / 12.0)

        possible_actions = len(self.getPossibleActions(board.board))
        features.append(possible_actions / 7.0)

        total_pieces = np.sum(board.board != 0)
        features.append(total_pieces / 42.0)

        # Game phase
        if total_pieces < 14:
            features.extend([1.0, 0.0, 0.0])
        elif total_pieces < 28:
            features.extend([0.0, 1.0, 0.0])
        else:
            features.extend([0.0, 0.0, 1.0])

        features.append(1.0 if startValue == 1 else 0.0)

        # Balance analysis
        left_pieces = np.sum(board.board[:, :3] == startValue)
        right_pieces = np.sum(board.board[:, 4:] == startValue)
        total_own = left_pieces + right_pieces
        balance = 1.0 - abs(left_pieces - right_pieces) / max(total_own, 1)
        features.append(balance)

        # Threat analysis (21 features)
        for col in range(7):
            can_win = self._can_win_in_column(board, col, startValue)
            features.append(1.0 if can_win else 0.0)

        for col in range(7):
            must_block = self._can_win_in_column(board, col, -startValue)
            features.append(1.0 if must_block else 0.0)

        for col in range(7):
            threat_level = self._analyze_column_threats(board, col, startValue)
            features.append(threat_level)

        # Formation analysis (14 features)
        own_formations = self._analyze_formations(board, startValue)
        features.extend(own_formations)

        opp_formations = self._analyze_formations(board, -startValue)
        features.extend(opp_formations)

        # Positional evaluation (14 features)
        corners = [(0, 0), (0, 6), (5, 0), (5, 6)]
        for row, col in corners:
            if board.board[row, col] == startValue:
                features.append(1.0)
            elif board.board[row, col] == -startValue:
                features.append(-1.0)
            else:
                features.append(0.0)

        edge_positions = [(0, 3), (5, 3), (2, 0), (2, 6)]
        for row, col in edge_positions:
            if board.board[row, col] == startValue:
                features.append(1.0)
            elif board.board[row, col] == -startValue:
                features.append(-1.0)
            else:
                features.append(0.0)

        connectivity = self._calculate_connectivity(board, startValue)
        opp_connectivity = self._calculate_connectivity(board, -startValue)
        features.extend([connectivity / 50.0, opp_connectivity / 50.0])

        center_pieces = np.sum(board.board[:, 3] == startValue)
        center_opp = np.sum(board.board[:, 3] == -startValue)
        features.extend([center_pieces / 6.0, center_opp / 6.0])

        # Diagonal analysis
        main_diag = sum(1 for i in range(min(6, 7)) if i < 6 and board.board[i, i] == startValue)
        anti_diag = sum(1 for i in range(min(6, 7)) if i < 6 and (5 - i) < 7 and board.board[i, 5 - i] == startValue)
        features.extend([main_diag / 6.0, anti_diag / 6.0])

        # Tactical features (9 features)
        for col in range(7):
            pressure = self._calculate_column_pressure(board, col, startValue, column_heights[col])
            features.append(pressure)

        mobility = len(self.getPossibleActions(board.board)) / 7.0
        tempo = self._calculate_tempo(board, startValue)
        features.extend([mobility, tempo])

        return features

    def _can_win_in_column(self, board, col, player_value):
        """Check immediate win possibility"""
        if col not in self.getPossibleActions(board.board):
            return False

        temp_board = board.board.copy()
        empty_rows = np.where(temp_board[:, col] == 0)[0]
        if len(empty_rows) == 0:
            return False

        row = np.max(empty_rows)
        temp_board[row, col] = player_value

        temp_board_obj = gamerules.Board()
        temp_board_obj.board = temp_board
        temp_board_obj.components = board.components.copy()
        temp_board_obj.components4 = board.components4.copy()
        temp_board_obj.updateComponents(row, col, player_value)
        temp_board_obj.updateComponents4(row, col, player_value)

        return temp_board_obj.checkVictory(col, player_value)

    def _analyze_column_threats(self, board, col, player_value):
        """Analyze threat potential in column"""
        if col not in self.getPossibleActions(board.board):
            return 0.0

        temp_board = board.board.copy()
        empty_rows = np.where(temp_board[:, col] == 0)[0]
        if len(empty_rows) == 0:
            return 0.0

        row = np.max(empty_rows)
        connections = 0

        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 6 and 0 <= nc < 7:
                if temp_board[nr, nc] == player_value:
                    connections += 1
                elif temp_board[nr, nc] == -player_value:
                    connections -= 0.5

        return max(0.0, min(1.0, connections / 8.0))

    def _analyze_formations(self, board, player_value):
        """Analyze formation strength"""
        features = [0.0] * 7

        components = board.components * (np.sign(board.components) == np.sign(player_value))
        if np.any(components):
            unique_components = components[components != 0]
            if len(unique_components) > 0:
                sizes = np.bincount(np.abs(unique_components.astype(int)))
                for size in sizes[1:]:
                    if size <= 7:
                        features[size - 1] = min(1.0, features[size - 1] + 0.2)

        return features

    def _calculate_connectivity(self, board, player_value):
        """Calculate piece connectivity"""
        connectivity = 0

        for row in range(6):
            for col in range(7):
                if board.board[row, col] == player_value:
                    for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < 6 and 0 <= nc < 7 and board.board[nr, nc] == player_value:
                            connectivity += 1

        return connectivity

    def _calculate_column_pressure(self, board, col, player_value, height):
        """Calculate strategic column pressure"""
        pressure = 0.0

        if height >= 4:
            pressure += 0.3
        if height >= 5:
            pressure += 0.4

        center_bonus = max(0, 1.0 - abs(col - 3) * 0.2)
        pressure += center_bonus * 0.3

        return pressure

    def _calculate_tempo(self, board, player_value):
        """Calculate tempo advantage"""
        own_pieces = np.sum(board.board == player_value)
        opp_pieces = np.sum(board.board == -player_value)

        if own_pieces + opp_pieces == 0:
            return 0.5

        return own_pieces / (own_pieces + opp_pieces)


class RandomPlayer(gamerules.Player):
    """Random opponent for testing"""

    def __init__(self, name="Random Opponent"):
        super().__init__(name)

    def getAction(self, board, startValue):
        possibleActions = self.getPossibleActions(board.board)
        return np.random.choice(possibleActions)

    def newGame(self, new_opponent):
        pass


def play_game(player1, player2, verbose=False):
    """Play single game between two players"""
    board = gamerules.Board()
    startValue = {player1: 1, player2: -1}

    player1.newGame(True)
    player2.newGame(True)

    players = [player1, player2]
    current_player = 0
    max_moves = 42
    moves = 0

    while moves < max_moves:
        player = players[current_player]

        try:
            action = player.getAction(board, startValue[player])
        except Exception as e:
            if verbose:
                print(f"Player {player.getName()} error: {e}")
            return -1 if current_player == 0 else 1

        possible_actions = board.getPossibleActions()
        if action not in possible_actions:
            if verbose:
                print(f"Player {player.getName()} invalid move: {action}")
            return -1 if current_player == 0 else 1

        board.updateBoard(action, startValue[player])

        if verbose:
            print(f"{player.getName()} plays column {action}")

        if board.checkVictory(action, startValue[player]):
            if verbose:
                print(f"{player.getName()} wins!")
            return 1 if current_player == 0 else -1

        if len(board.getPossibleActions()) == 0:
            if verbose:
                print("Draw")
            return 0

        current_player = 1 - current_player
        moves += 1

    if verbose:
        print("Draw - timeout")
    return 0


def evaluate_player(player, num_games=100, verbose=False):
    """Comprehensive player evaluation"""
    print(f"Evaluating: {player.getName()}")
    print(f"Games: {num_games}")
    print(f"Target: {num_games * 0.8:.0f} wins (80%)")
    print("-" * 50)

    opponent = RandomPlayer()

    wins = 0
    draws = 0
    losses = 0
    first_wins = 0
    second_wins = 0
    games_first = 0
    games_second = 0

    if TQDM_AVAILABLE:
        pbar = tqdm(total=num_games, desc="Testing", unit="game")

    for game in range(num_games):
        player_starts = (game % 2 == 0)
        players = [player, opponent] if player_starts else [opponent, player]

        result = play_game(players[0], players[1], verbose)

        if result == 1:  # First player wins
            if player_starts:
                wins += 1
                first_wins += 1
                games_first += 1
            else:
                losses += 1
                games_second += 1
        elif result == -1:  # Second player wins
            if player_starts:
                losses += 1
                games_first += 1
            else:
                wins += 1
                second_wins += 1
                games_second += 1
        else:  # Draw
            draws += 1
            if player_starts:
                games_first += 1
            else:
                games_second += 1

        if TQDM_AVAILABLE:
            pbar.update(1)
            pbar.set_postfix_str(f"Win Rate: {(wins / (game + 1)) * 100:.1f}%")

    if TQDM_AVAILABLE:
        pbar.close()

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Overall performance
    win_rate = wins / num_games * 100
    draw_rate = draws / num_games * 100
    loss_rate = losses / num_games * 100

    print("Overall Performance:")
    print(f"  Wins:   {wins:3d}/{num_games} ({win_rate:5.1f}%)")
    print(f"  Draws:  {draws:3d}/{num_games} ({draw_rate:5.1f}%)")
    print(f"  Losses: {losses:3d}/{num_games} ({loss_rate:5.1f}%)")

    # Positional performance
    first_rate = first_wins / max(1, games_first) * 100
    second_rate = second_wins / max(1, games_second) * 100

    print("\nPositional Performance:")
    print(f"  As first player:  {first_wins:2d}/{games_first} ({first_rate:5.1f}%)")
    print(f"  As second player: {second_wins:2d}/{games_second} ({second_rate:5.1f}%)")

    # Balance analysis
    if first_rate > 0 and second_rate > 0:
        balance_score = min(first_rate, second_rate) / max(first_rate, second_rate) * 100
    else:
        balance_score = 0

    print(f"\nBalance Score: {balance_score:5.1f}%")
    if balance_score >= 85:
        print("  ✅ Excellent balance")
    elif balance_score >= 70:
        print("  ⚡ Good balance")
    else:
        print("  ⚠️ Needs improvement")

    # Performance metrics
    non_loss_rate = (wins + draws) / num_games * 100
    win_efficiency = wins / max(1, wins + draws) * 100

    print(f"\nAdvanced Metrics:")
    print(f"  Non-loss rate: {non_loss_rate:5.1f}%")
    print(f"  Win efficiency: {win_efficiency:5.1f}%")

    # Success evaluation
    meets_requirement = wins >= num_games * 0.8
    excellent_performance = (
            meets_requirement and
            balance_score >= 80 and
            min(first_rate, second_rate) >= 75
    )

    print(f"\nEvaluation:")
    if excellent_performance:
        print("  ✅ EXCELLENT - Exceeds all criteria")
    elif meets_requirement:
        print("  ✅ SUCCESS - Meets minimum requirement")
        print(f"     Required: {num_games * 0.8:.0f} wins, Achieved: {wins}")
    else:
        print("  ❌ INSUFFICIENT - Below requirement")
        print(f"     Required: {num_games * 0.8:.0f} wins, Achieved: {wins}")

        # Specific recommendations
        if balance_score < 70:
            print("     🔧 Improve: Balance between positions")
        if second_rate < 70:
            print("     🔧 Improve: Second player strategy")
        if first_rate < 80:
            print("     🔧 Improve: First player consistency")

    print("=" * 60)

    return wins, draws, losses, first_rate, second_rate, balance_score


def main():
    """Main evaluation function"""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate DQN player performance')
    parser.add_argument('--games', type=int, default=100,
                        help='Number of games (default: 100)')
    parser.add_argument('--weights', type=str, default='weights.pkl',
                        help='Weights file (default: weights.pkl)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show game details')
    parser.add_argument('--name', type=str, default='Taras Demchyna',
                        help='Player name (default: Taras Demchyna)')

    args = parser.parse_args()

    print("DQN Player Evaluation")
    print("=" * 50)

    # Initialize player
    try:
        player = TestPlayer(args.name, args.weights)
        print(f"✅ Player initialized successfully")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return 1

    # Run evaluation
    try:
        results = evaluate_player(player, args.games, args.verbose)
        wins, draws, losses, first_rate, second_rate, balance_score = results

        # Determine success
        meets_requirement = wins >= args.games * 0.8
        excellent = (meets_requirement and
                     balance_score >= 80 and
                     min(first_rate, second_rate) >= 75)

        return 0 if meets_requirement else 1

    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted")
        return 1
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())