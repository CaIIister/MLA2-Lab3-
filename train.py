#!/usr/bin/env python3
"""
Enhanced DQN training system designed to achieve 80-85% win rate.
Addresses key weaknesses in the fast approach.
"""

import numpy as np
import random
from collections import deque
import pickle
import gamerules
import copy
import time
from player import Player, HeuristicEngine, CustomDQN

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class EnhancedPlayer(Player):
    """Enhanced player with improved state encoding and strategy"""

    def __init__(self, name, use_heuristics=True):
        # Initialize without loading weights
        self.name = name
        self.use_heuristics = use_heuristics

        # Use full architecture with enhanced features
        self.q_network = CustomDQN(
            input_size=200,  # Full feature set
            hidden_sizes=[256, 128, 64],  # Full architecture
            output_size=7,
            learning_rate=0.0005,  # Balanced learning rate
            min_learning_rate=1e-6,
            learning_rate_patience=10
        )

        # Enhanced heuristic engine
        self.heuristics = HeuristicEngine()

        # Training parameters
        self.epsilon = 0.95  # Higher initial exploration
        self.training_phase = "exploration"  # Track training phase

    def getName(self):
        return self.name

    def newGame(self, new_opponent):
        pass

    def getAction(self, board, startValue):
        """Enhanced action selection with improved second-player strategy"""
        possibleActions = self.getPossibleActions(board.board)

        if len(possibleActions) == 0:
            return 0

        # Enhanced heuristic checks with second-player focus
        if self.use_heuristics:
            # Immediate win - always take it
            action, reason = self.heuristics.find_immediate_win(board, startValue)
            if action is not None:
                return action

            # Block opponent win - always do it
            action, reason = self.heuristics.find_must_block(board, startValue)
            if action is not None:
                return action

            # Enhanced offensive strategy for second player
            if startValue == -1:  # Second player needs more aggressive tactics
                action = self._get_aggressive_second_player_move(board, startValue, possibleActions)
                if action is not None:
                    return action

            # Create winning threats
            action, reason = self.heuristics.find_winning_threat(board, startValue)
            if action is not None and random.random() < 0.8:  # High probability to take
                return action

        # DQN decision with enhanced exploration
        if random.random() < self.epsilon:
            return self._smart_exploration(board, startValue, possibleActions)

        # Use DQN
        try:
            state = self._encode_state_enhanced(board, startValue)
            q_values = self.q_network.predict(state)[0]

            # Mask invalid actions
            q_values_masked = q_values.copy()
            for i in range(7):
                if i not in possibleActions:
                    q_values_masked[i] = float('-inf')

            # Add small random noise for diversity
            noise = np.random.normal(0, 0.01, size=q_values_masked.shape)
            q_values_masked += noise

            action = np.argmax(q_values_masked)
            return int(action) if action in possibleActions else random.choice(possibleActions)

        except Exception as e:
            return self._smart_exploration(board, startValue, possibleActions)

    def _get_aggressive_second_player_move(self, board, startValue, possibleActions):
        """Enhanced strategy for second player to be more aggressive"""
        # Look for moves that create multiple threats
        best_action = None
        best_score = 0

        for action in possibleActions:
            score = self._evaluate_aggressive_move(board, action, startValue)
            if score > best_score:
                best_score = score
                best_action = action

        # Only return if significantly better than random
        if best_score > 0.6:
            return best_action
        return None

    def _evaluate_aggressive_move(self, board, action, player_value):
        """Evaluate how aggressive/threatening a move is"""
        # Simulate the move
        temp_board = board.board.copy()
        empty_rows = np.where(temp_board[:, action] == 0)[0]
        if len(empty_rows) == 0:
            return 0.0

        row = np.max(empty_rows)
        temp_board[row, action] = player_value

        score = 0.0

        # Count potential winning lines this creates
        winning_potential = self._count_winning_potential(temp_board, row, action, player_value)
        score += winning_potential * 0.4

        # Bonus for central play
        center_bonus = max(0, 1.0 - abs(action - 3) * 0.15)
        score += center_bonus * 0.3

        # Bonus for building on existing structures
        adjacent_own = self._count_adjacent_pieces(temp_board, row, action, player_value)
        score += min(0.3, adjacent_own * 0.1)

        return score

    def _count_winning_potential(self, board, row, col, player_value):
        """Count potential winning lines from this position"""
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, 1)]
        potential = 0

        for dr, dc in directions:
            line_length = 1  # Current piece
            spaces = 0

            # Check positive direction
            nr, nc = row + dr, col + dc
            while 0 <= nr < 6 and 0 <= nc < 7:
                if board[nr, nc] == player_value:
                    line_length += 1
                elif board[nr, nc] == 0:
                    spaces += 1
                    if spaces > 2:  # Too many gaps
                        break
                else:
                    break  # Opponent piece
                nr, nc = nr + dr, nc + dc

            # Check negative direction
            nr, nc = row - dr, col - dc
            while 0 <= nr < 6 and 0 <= nc < 7:
                if board[nr, nc] == player_value:
                    line_length += 1
                elif board[nr, nc] == 0:
                    spaces += 1
                    if spaces > 2:  # Too many gaps
                        break
                else:
                    break  # Opponent piece
                nr, nc = nr - dr, nc - dc

            # Score based on line length and available spaces
            if line_length >= 3:
                potential += 0.8
            elif line_length >= 2:
                potential += 0.4

        return min(1.0, potential)

    def _count_adjacent_pieces(self, board, row, col, player_value):
        """Count adjacent pieces of same color"""
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < 6 and 0 <= nc < 7 and board[nr, nc] == player_value:
                    count += 1
        return count

    def _smart_exploration(self, board, startValue, possibleActions):
        """Intelligent exploration strategy"""
        # Prefer center columns during exploration
        center_actions = [a for a in possibleActions if 1 <= a <= 5]

        # Early game: prefer center
        total_pieces = np.sum(board.board != 0)
        if total_pieces < 10 and center_actions:
            return random.choice(center_actions)

        # Mid game: balanced exploration
        if total_pieces < 25:
            weights = [0.5, 0.7, 0.9, 1.0, 0.9, 0.7, 0.5]  # Center preference
            action_weights = [weights[a] for a in possibleActions]
            action_weights = np.array(action_weights)
            action_weights /= action_weights.sum()
            return np.random.choice(possibleActions, p=action_weights)

        # Late game: more random
        return random.choice(possibleActions)


class RNGPlayer(gamerules.Player):
    """Random player for training"""

    def __init__(self, name="Random"):
        super().__init__(name)

    def getAction(self, board, startValue):
        possibleActions = self.getPossibleActions(board.board)
        return np.random.choice(possibleActions)

    def newGame(self, new_opponent):
        pass


class EnhancedTrainer:
    """Enhanced trainer designed to achieve 80-85% win rate"""

    def __init__(self, mode="enhanced"):
        print(f"🚀 Initializing Enhanced DQN Trainer...")

        # Enhanced training configuration
        self.config = {
            "episodes": 8000,  # More episodes for convergence
            "buffer_size": 100000,  # Larger buffer
            "batch_size": 256,  # Larger batches
            "target_update": 200,  # Less frequent target updates
            "eval_freq": 250,  # Frequent evaluation
            "min_buffer": 2000,  # Larger minimum buffer
            "eval_games": 100,  # More evaluation games for accuracy
            "description": "Enhanced training for 80-85% win rate"
        }

        print(f"📋 Mode: enhanced - {self.config['description']}")

        # Initialize enhanced player
        self.player = EnhancedPlayer("EnhancedDQN", use_heuristics=True)

        # Target network
        self.target_network = CustomDQN(
            input_size=200,
            hidden_sizes=[256, 128, 64],
            output_size=7,
            learning_rate=0.0005
        )
        self.target_network.copy_weights_from(self.player.q_network)

        # Enhanced replay buffer
        self.replay_buffer = deque(maxlen=self.config['buffer_size'])

        # Enhanced training parameters
        self.epsilon_decay = 0.9985  # Slower decay for better exploration
        self.epsilon_min = 0.02  # Lower minimum for continued exploration
        self.gamma = 0.98  # Higher discount for longer-term thinking

        # Training phases
        self.phase_episodes = {
            "exploration": 2000,
            "refinement": 4000,
            "convergence": 8000
        }

        # Statistics
        self.episode_rewards = []
        self.win_rates = []
        self.losses = []
        self.best_win_rate = 0
        self.best_model_state = None

    def play_training_game(self, opponent):
        """Enhanced training game with better experience collection"""
        board = gamerules.Board()

        player_starts = random.choice([True, False])
        player_value = 1 if player_starts else -1

        experiences = []
        states = []
        actions = []

        turn = 1
        max_turns = 42
        turns = 0

        while turns < max_turns:
            current_player_is_dqn = (turn == 1 and player_starts) or (turn == -1 and not player_starts)

            if current_player_is_dqn:
                state = self.player._encode_state_enhanced(board, player_value)
                action = self.player.getAction(board, player_value)
                states.append(state)
                actions.append(action)
            else:
                action = opponent.getAction(board, -player_value)

            # Validate action
            possible_actions = self.player.getPossibleActions(board.board)
            if action not in possible_actions:
                # Invalid move loses immediately
                result = -1 if current_player_is_dqn else 1
                return result, self._create_experiences(states, actions, result)

            # Make move
            board.updateBoard(action, turn)

            # Check victory
            if board.checkVictory(action, turn):
                result = 1 if current_player_is_dqn else -1
                return result, self._create_experiences(states, actions, result)

            # Check draw
            if len(board.getPossibleActions()) == 0:
                return 0, self._create_experiences(states, actions, 0)

            turn *= -1
            turns += 1

        # Timeout draw
        return 0, self._create_experiences(states, actions, 0)

    def _create_experiences(self, states, actions, game_result):
        """Create enhanced experience tuples with better reward shaping"""
        experiences = []
        n_moves = len(states)

        if n_moves == 0:
            return experiences

        for i in range(n_moves):
            # Enhanced reward calculation
            if game_result == 1:  # Win
                # Exponential reward decay from end of game
                moves_from_end = n_moves - i
                base_reward = 10.0
                decay_factor = 0.95 ** (moves_from_end - 1)
                reward = base_reward * decay_factor

                # Bonus for efficient wins
                if n_moves <= 15:  # Quick win bonus
                    reward *= 1.3
                elif n_moves <= 20:
                    reward *= 1.1

            elif game_result == -1:  # Loss
                # Progressive penalty - later mistakes are worse
                move_number = i + 1
                base_penalty = -5.0
                progression_penalty = -(move_number * 0.3)
                reward = base_penalty + progression_penalty

            else:  # Draw
                # Small positive reward that decreases over time
                base_reward = 2.0
                time_penalty = -(i * 0.1)
                reward = base_reward + time_penalty

            # Next state for Q-learning
            if i < n_moves - 1:
                next_state = states[i + 1]
                done = False
            else:
                next_state = np.zeros_like(states[i])
                done = True

            experiences.append({
                'state': states[i],
                'action': actions[i],
                'reward': reward,
                'next_state': next_state,
                'done': done
            })

        return experiences

    def store_experiences(self, experiences):
        """Store experiences in replay buffer"""
        for exp in experiences:
            self.replay_buffer.append(exp)

    def train_step(self):
        """Enhanced training step with better stability"""
        if len(self.replay_buffer) < self.config['min_buffer']:
            return None

        # Sample batch
        batch_size = min(self.config['batch_size'], len(self.replay_buffer))
        batch = random.sample(self.replay_buffer, batch_size)

        # Prepare batch data
        states = np.array([exp['state'] for exp in batch])
        actions = np.array([exp['action'] for exp in batch])
        rewards = np.array([exp['reward'] for exp in batch])
        next_states = np.array([exp['next_state'] for exp in batch])
        dones = np.array([exp['done'] for exp in batch])

        # Double DQN target calculation
        current_q = self.player.q_network.predict(states)
        next_q_main = self.player.q_network.predict(next_states)
        next_q_target = self.target_network.predict(next_states)

        targets = current_q.copy()

        for i in range(batch_size):
            if dones[i]:
                target_value = rewards[i]
            else:
                # Double DQN: use main network to select action, target network to evaluate
                best_action = np.argmax(next_q_main[i])
                target_value = rewards[i] + self.gamma * next_q_target[i, best_action]

            targets[i, actions[i]] = target_value

        # Train the network
        loss = self.player.q_network.train_step(states, targets)

        # Adjust learning rate if needed
        lr_adjusted = self.player.q_network.adjust_learning_rate(loss)
        if lr_adjusted:
            print(f"Learning rate adjusted to: {self.player.q_network.learning_rate:.6f}")

        return loss

    def evaluate_player(self, num_games=100):
        """Comprehensive evaluation"""
        original_epsilon = self.player.epsilon
        self.player.epsilon = 0.0  # No exploration during evaluation

        wins_first = 0
        wins_second = 0
        draws = 0
        games_first = 0
        games_second = 0

        opponent = RNGPlayer("Random")

        for game in range(num_games):
            player_starts = (game % 2 == 0)

            if player_starts:
                games_first += 1
                result, _ = self.play_training_game(opponent)
                if result == 1:
                    wins_first += 1
                elif result == 0:
                    draws += 1
            else:
                games_second += 1
                result, _ = self.play_training_game(opponent)
                if result == 1:
                    wins_second += 1
                elif result == 0:
                    draws += 1

        self.player.epsilon = original_epsilon

        total_wins = wins_first + wins_second
        win_rate = (total_wins / num_games) * 100

        first_rate = (wins_first / games_first) * 100 if games_first > 0 else 0
        second_rate = (wins_second / games_second) * 100 if games_second > 0 else 0

        return win_rate, first_rate, second_rate, total_wins, draws

    def update_training_phase(self, episode):
        """Update training phase and parameters"""
        if episode <= self.phase_episodes["exploration"]:
            self.player.training_phase = "exploration"
        elif episode <= self.phase_episodes["refinement"]:
            self.player.training_phase = "refinement"
        else:
            self.player.training_phase = "convergence"

    def train(self):
        """Enhanced training loop"""
        episodes = self.config['episodes']
        print(f"🚀 Starting Enhanced training for {episodes} episodes...")
        print(f"🎯 Target: 80-85% win rate with balanced first/second player performance")

        start_time = time.time()

        # Progress bar
        if TQDM_AVAILABLE:
            pbar = tqdm(total=episodes, desc="Enhanced Training", unit="ep")

        opponent = RNGPlayer("Random")

        for episode in range(episodes):
            # Update training phase
            self.update_training_phase(episode)

            # Play game
            game_result, experiences = self.play_training_game(opponent)

            # Store experiences
            self.store_experiences(experiences)

            # Track statistics
            reward = 10 if game_result == 1 else (-10 if game_result == -1 else 1)
            self.episode_rewards.append(reward)

            # Train network
            loss = self.train_step()
            if loss is not None:
                self.losses.append(loss)

            # Update target network
            if episode % self.config['target_update'] == 0 and episode > 0:
                self.target_network.copy_weights_from(self.player.q_network)

            # Decay epsilon
            if self.player.epsilon > self.epsilon_min:
                self.player.epsilon *= self.epsilon_decay

            # Evaluation and progress reporting
            if episode % self.config['eval_freq'] == 0 and episode > 0:
                win_rate, first_rate, second_rate, wins, draws = self.evaluate_player(self.config['eval_games'])
                self.win_rates.append(win_rate)

                # Save best model
                if win_rate > self.best_win_rate:
                    self.best_win_rate = win_rate
                    self.best_model_state = self.player.q_network.state_dict()
                    self.player.q_network.save_weights('weights.pkl')

                # Progress reporting
                elapsed = time.time() - start_time
                avg_reward = np.mean(self.episode_rewards[-self.config['eval_freq']:])
                avg_loss = np.mean(self.losses[-50:]) if self.losses else 0

                status = (f"Ep {episode:4d} | "
                          f"Overall: {win_rate:5.1f}% | "
                          f"1st: {first_rate:5.1f}% | "
                          f"2nd: {second_rate:5.1f}% | "
                          f"Best: {self.best_win_rate:5.1f}% | "
                          f"ε: {self.player.epsilon:.3f} | "
                          f"Phase: {self.player.training_phase}")

                if TQDM_AVAILABLE:
                    tqdm.write(status)
                    pbar.set_postfix_str(f"Win: {win_rate:.1f}% (1st: {first_rate:.1f}%, 2nd: {second_rate:.1f}%)")
                else:
                    print(status)

                # Success criteria
                if win_rate >= 80.0 and min(first_rate, second_rate) >= 70.0:
                    print(
                        f"\n🎉 TARGET ACHIEVED! Overall: {win_rate:.1f}%, First: {first_rate:.1f}%, Second: {second_rate:.1f}%")
                    break

            # Update progress bar
            if TQDM_AVAILABLE:
                pbar.update(1)

        if TQDM_AVAILABLE:
            pbar.close()

        # Load best model
        if self.best_model_state:
            self.player.q_network.load_state_dict(self.best_model_state)
            self.player.q_network.save_weights('weights.pkl')

        # Final evaluation
        final_overall, final_first, final_second, _, _ = self.evaluate_player(self.config['eval_games'])

        total_time = time.time() - start_time
        print(f"\n🏁 Enhanced training completed in {total_time / 60:.1f} minutes!")
        print(f"📊 Final Results:")
        print(f"   Overall win rate: {final_overall:.1f}%")
        print(f"   First player: {final_first:.1f}%")
        print(f"   Second player: {final_second:.1f}%")
        print(f"🎯 Best overall: {self.best_win_rate:.1f}%")
        print(f"💾 Weights saved to 'weights.pkl'")

        return self.player


def main():
    """Main enhanced training function"""
    print("=" * 60)
    print("🎯 ENHANCED DQN TRAINING")
    print("=" * 60)
    print("🎯 Goal: 80-85% win rate with balanced performance")
    print("🧠 Architecture: Full features + enhanced strategy")
    print("⚖️ Focus: Balanced first/second player performance")
    print("=" * 60)

    # Create enhanced trainer
    trainer = EnhancedTrainer()

    # Train
    trained_player = trainer.train()

    print("\n🎉 Enhanced training completed!")
    print("🧪 Test with: python test_enhanced.py")


if __name__ == "__main__":
    main()