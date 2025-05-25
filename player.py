import numpy as np
import pickle
import os
import gamerules


class StableDQN:
    """Stable DQN implementation addressing training instability issues"""

    def __init__(self, input_size=120, hidden_sizes=[256, 128], output_size=7, learning_rate=0.0001, min_learning_rate=1e-6, learning_rate_patience=50):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.min_lr = min_learning_rate
        self.lr_schedule_patience = learning_rate_patience

        # Stable initialization
        self.layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            # Xavier initialization for stability
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))

            layer = {
                'weights': np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * std,
                'biases': np.zeros((1, layer_sizes[i + 1])),
                'weights_momentum': np.zeros((layer_sizes[i], layer_sizes[i + 1])),
                'biases_momentum': np.zeros((1, layer_sizes[i + 1])),
                'weights_velocity': np.zeros((layer_sizes[i], layer_sizes[i + 1])),
                'biases_velocity': np.zeros((1, layer_sizes[i + 1]))
            }
            self.layers.append(layer)

        # Stable optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

        # Gradient clipping for stability
        self.max_grad_norm = 1.0

        # Learning rate scheduling
        self.lr_decay_factor = 0.98
        self.loss_history = []
        self.best_loss = float('inf')
        self.lr_plateau_count = 0
        self.plateau_threshold = 0.01

    def leaky_relu(self, x, alpha=0.01):
        """Leaky ReLU for better gradient flow"""
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        """Leaky ReLU derivative"""
        return np.where(x > 0, 1, alpha)

    def forward(self, x):
        """Stable forward pass"""
        self.activations = [x]
        self.z_values = []

        current_input = x
        for i, layer in enumerate(self.layers):
            z = np.dot(current_input, layer['weights']) + layer['biases']
            self.z_values.append(z)

            if i < len(self.layers) - 1:  # Hidden layers
                # Use Leaky ReLU for better gradient flow
                activation = self.leaky_relu(z)
            else:  # Output layer
                activation = z

            self.activations.append(activation)
            current_input = activation

        return self.activations[-1]

    def backward(self, y_true, y_pred):
        """Stable backpropagation with gradient clipping"""
        batch_size = y_true.shape[0]
        delta = (y_pred - y_true) / batch_size

        gradients = []
        total_grad_norm = 0

        for i in reversed(range(len(self.layers))):
            # Compute gradients
            weights_grad = np.dot(self.activations[i].T, delta)
            biases_grad = np.sum(delta, axis=0, keepdims=True)

            # Accumulate gradient norm for clipping
            total_grad_norm += np.sum(weights_grad ** 2)
            total_grad_norm += np.sum(biases_grad ** 2)

            gradients.insert(0, {
                'weights': weights_grad,
                'biases': biases_grad
            })

            if i > 0:
                delta = np.dot(delta, self.layers[i]['weights'].T)
                delta = delta * self.leaky_relu_derivative(self.z_values[i - 1])

        # Gradient clipping
        total_grad_norm = np.sqrt(total_grad_norm)
        if total_grad_norm > self.max_grad_norm:
            clip_factor = self.max_grad_norm / total_grad_norm
            for grad in gradients:
                grad['weights'] *= clip_factor
                grad['biases'] *= clip_factor

        return gradients

    def update_weights(self, gradients):
        """Stable weight updates with Adam optimizer"""
        self.t += 1

        for i, (layer, grad) in enumerate(zip(self.layers, gradients)):
            # Adam updates
            layer['weights_momentum'] = self.beta1 * layer['weights_momentum'] + (1 - self.beta1) * grad['weights']
            layer['biases_momentum'] = self.beta1 * layer['biases_momentum'] + (1 - self.beta1) * grad['biases']

            layer['weights_velocity'] = self.beta2 * layer['weights_velocity'] + (1 - self.beta2) * (
                        grad['weights'] ** 2)
            layer['biases_velocity'] = self.beta2 * layer['biases_velocity'] + (1 - self.beta2) * (grad['biases'] ** 2)

            # Bias correction
            weights_momentum_corrected = layer['weights_momentum'] / (1 - self.beta1 ** self.t)
            biases_momentum_corrected = layer['biases_momentum'] / (1 - self.beta1 ** self.t)
            weights_velocity_corrected = layer['weights_velocity'] / (1 - self.beta2 ** self.t)
            biases_velocity_corrected = layer['biases_velocity'] / (1 - self.beta2 ** self.t)

            # Update parameters
            layer['weights'] -= self.learning_rate * weights_momentum_corrected / (
                        np.sqrt(weights_velocity_corrected) + self.epsilon)
            layer['biases'] -= self.learning_rate * biases_momentum_corrected / (
                        np.sqrt(biases_velocity_corrected) + self.epsilon)

    def adjust_learning_rate(self, loss):
        """Adaptive learning rate adjustment with improved stability"""
        self.loss_history.append(loss)
        
        # Use a longer window for loss averaging
        window_size = 100
        if len(self.loss_history) > window_size:
            avg_loss = np.mean(self.loss_history[-window_size:])
            
            # Update best loss if current average is better
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.lr_plateau_count = 0
            else:
                self.lr_plateau_count += 1
            
            # Decay learning rate if plateaued for patience steps
            if self.lr_plateau_count >= self.lr_schedule_patience:
                old_lr = self.learning_rate
                self.learning_rate = max(self.min_lr, self.learning_rate * 0.95)
                
                # Reset plateau counter if learning rate was adjusted
                if old_lr != self.learning_rate:
                    self.lr_plateau_count = 0
                    print(f"Learning rate adjusted to: {self.learning_rate:.6f}")
                    
            # Learning rate recovery if performance improves significantly
            elif avg_loss < self.best_loss * 0.8:  # 20% improvement
                self.learning_rate = min(self.initial_lr, self.learning_rate * 1.05)
                print(f"Learning rate increased to: {self.learning_rate:.6f}")
                self.best_loss = avg_loss

    def predict(self, x):
        """Prediction with stability checks"""
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        # Check for NaN/inf in input
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            print("Warning: Invalid input detected")
            x = np.nan_to_num(x)

        output = self.forward(x)

        # Check for NaN/inf in output
        if np.any(np.isnan(output)) or np.any(np.isinf(output)):
            print("Warning: Invalid output detected")
            output = np.nan_to_num(output)

        return output

    def train_step(self, state, target):
        """Stable training step"""
        predicted = self.forward(state)
        gradients = self.backward(target, predicted)
        self.update_weights(gradients)

        loss = np.mean((predicted - target) ** 2)

        # Check for training instability
        if np.isnan(loss) or np.isinf(loss):
            print("Warning: Training instability detected, resetting learning rate")
            self.learning_rate = self.initial_lr * 0.1
            loss = 1.0  # Fallback loss value

        return loss

    def copy_weights_from(self, other_network):
        """Copy weights from another network"""
        for i, (self_layer, other_layer) in enumerate(zip(self.layers, other_network.layers)):
            self_layer['weights'] = other_layer['weights'].copy()
            self_layer['biases'] = other_layer['biases'].copy()

    def save_weights(self, filepath):
        """Save network weights"""
        weights_data = {
            'layers': [],
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate
        }

        for layer in self.layers:
            weights_data['layers'].append({
                'weights': layer['weights'],
                'biases': layer['biases']
            })

        with open(filepath, 'wb') as f:
            pickle.dump(weights_data, f)

    def load_weights(self, filepath):
        """Load network weights"""
        try:
            with open(filepath, 'rb') as f:
                weights_data = pickle.load(f)

            if (weights_data['input_size'] != self.input_size or
                    weights_data['output_size'] != self.output_size):
                print("Warning: Architecture mismatch")
                return False

            if len(weights_data['layers']) != len(self.layers):
                print("Warning: Different number of layers")
                return False

            for i, layer_data in enumerate(weights_data['layers']):
                if (layer_data['weights'].shape != self.layers[i]['weights'].shape or
                        layer_data['biases'].shape != self.layers[i]['biases'].shape):
                    print(f"Warning: Layer {i} shape mismatch")
                    return False

                self.layers[i]['weights'] = layer_data['weights']
                self.layers[i]['biases'] = layer_data['biases']

            return True

        except Exception as e:
            print(f"Error loading weights: {e}")
            return False

    def state_dict(self):
        """Get network state"""
        return {
            'layers': [{
                'weights': layer['weights'].copy(),
                'biases': layer['biases'].copy()
            } for layer in self.layers]
        }

    def load_state_dict(self, state_dict):
        """Load network state"""
        for i, layer_state in enumerate(state_dict['layers']):
            self.layers[i]['weights'] = layer_state['weights'].copy()
            self.layers[i]['biases'] = layer_state['biases'].copy()


class ImprovedHeuristics:
    """Improved heuristic engine with better balance"""

    def __init__(self):
        self.column_weights = [0.6, 0.8, 0.9, 1.0, 0.9, 0.8, 0.6]

    def find_immediate_win(self, board, player_value):
        """Find immediate winning move"""
        possible_actions = self._get_possible_actions(board)
        for action in possible_actions:
            if self._can_win_with_move(board, action, player_value):
                return action, "WIN"
        return None, None

    def find_must_block(self, board, player_value):
        """Find move that must block opponent win"""
        possible_actions = self._get_possible_actions(board)
        opponent_value = -player_value
        for action in possible_actions:
            if self._can_win_with_move(board, action, opponent_value):
                return action, "BLOCK"
        return None, None

    def find_best_strategic_move(self, board, player_value):
        """Find best strategic move with improved second-player logic"""
        possible_actions = self._get_possible_actions(board)
        if not possible_actions:
            return None, None

        best_action = None
        best_score = -float('inf')

        for action in possible_actions:
            score = self._evaluate_move_comprehensive(board, action, player_value)

            # Enhanced scoring for second player
            if player_value == -1:
                score = self._enhance_second_player_strategy(board, action, score, player_value)

            if score > best_score:
                best_score = score
                best_action = action

        if best_action is not None and best_score > 0.3:
            return best_action, "STRATEGIC"
        return None, None

    def _enhance_second_player_strategy(self, board, action, base_score, player_value):
        """Enhanced strategy specifically for second player"""
        enhanced_score = base_score

        # More aggressive center play for second player
        if action == 3:  # Center column
            enhanced_score += 0.3
        elif action in [2, 4]:  # Near center
            enhanced_score += 0.2

        # Bonus for disrupting opponent formations
        disruption_bonus = self._calculate_disruption_value(board, action, player_value)
        enhanced_score += disruption_bonus * 0.4

        # Bonus for creating multiple threat opportunities
        threat_creation = self._calculate_multi_threat_potential(board, action, player_value)
        enhanced_score += threat_creation * 0.3

        return enhanced_score

    def _evaluate_move_comprehensive(self, board, action, player_value):
        """Comprehensive move evaluation"""
        if action not in self._get_possible_actions(board):
            return -float('inf')

        score = 0.0

        # Base positional value
        score += self.column_weights[action] * 0.2

        # Threat creation
        threat_value = self._calculate_threat_creation(board, action, player_value)
        score += threat_value * 0.3

        # Formation building
        formation_value = self._calculate_formation_value(board, action, player_value)
        score += formation_value * 0.2

        # Opponent disruption
        disruption_value = self._calculate_disruption_value(board, action, player_value)
        score += disruption_value * 0.2

        # Height considerations
        height_penalty = self._calculate_height_penalty(board, action)
        score -= height_penalty * 0.1

        return score

    def _calculate_threat_creation(self, board, action, player_value):
        """Calculate threat creation potential"""
        temp_board = board.board.copy()
        empty_rows = np.where(temp_board[:, action] == 0)[0]
        if len(empty_rows) == 0:
            return 0.0

        row = np.max(empty_rows)
        temp_board[row, action] = player_value

        threats = 0
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, 1)]

        for dr, dc in directions:
            line_strength = self._calculate_line_strength(temp_board, row, action, dr, dc, player_value)
            if line_strength >= 2:
                threats += 1

        return min(1.0, threats * 0.3)

    def _calculate_formation_value(self, board, action, player_value):
        """Calculate formation building value"""
        temp_board = board.board.copy()
        empty_rows = np.where(temp_board[:, action] == 0)[0]
        if len(empty_rows) == 0:
            return 0.0

        row = np.max(empty_rows)

        # Count adjacent friendly pieces
        adjacent_count = 0
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            nr, nc = row + dr, action + dc
            if 0 <= nr < 6 and 0 <= nc < 7 and board.board[nr, nc] == player_value:
                adjacent_count += 1

        return min(1.0, adjacent_count * 0.2)

    def _calculate_disruption_value(self, board, action, player_value):
        """Calculate opponent disruption value"""
        temp_board = board.board.copy()
        empty_rows = np.where(temp_board[:, action] == 0)[0]
        if len(empty_rows) == 0:
            return 0.0

        row = np.max(empty_rows)
        opponent_value = -player_value

        disruption = 0.0

        # Check if this move breaks opponent formations
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            nr, nc = row + dr, action + dc
            if 0 <= nr < 6 and 0 <= nc < 7 and board.board[nr, nc] == opponent_value:
                # Check line strength in multiple directions from opponent piece
                for odr, odc in [(-1, -1), (-1, 0), (-1, 1), (0, 1)]:
                    line_strength = self._calculate_line_strength(board.board, nr, nc, odr, odc, opponent_value)
                    if line_strength >= 2:
                        disruption += 0.3

        return min(1.0, disruption)

    def _calculate_multi_threat_potential(self, board, action, player_value):
        """Calculate potential for creating multiple threats"""
        temp_board = board.board.copy()
        empty_rows = np.where(temp_board[:, action] == 0)[0]
        if len(empty_rows) == 0:
            return 0.0

        row = np.max(empty_rows)
        temp_board[row, action] = player_value

        # Simulate opponent responses
        threat_count = 0
        possible_opponent_moves = self._get_possible_actions_from_board(temp_board)

        for opp_move in possible_opponent_moves:
            if opp_move == action:
                continue

            temp_board2 = temp_board.copy()
            opp_empty_rows = np.where(temp_board2[:, opp_move] == 0)[0]
            if len(opp_empty_rows) > 0:
                opp_row = np.max(opp_empty_rows)
                temp_board2[opp_row, opp_move] = -player_value

                # Check if we still have winning threats
                for final_move in self._get_possible_actions_from_board(temp_board2):
                    temp_board_obj = gamerules.Board()
                    temp_board_obj.board = temp_board2.copy()
                    if self._would_win_after_move(temp_board_obj, final_move, player_value):
                        threat_count += 1
                        break

        return min(1.0, threat_count * 0.2)

    def _calculate_height_penalty(self, board, action):
        """Calculate penalty for playing in tall columns"""
        height = 6 - len(np.where(board.board[:, action] == 0)[0])
        if height >= 5:
            return 0.8
        elif height >= 4:
            return 0.4
        return 0.0

    def _calculate_line_strength(self, board, row, col, dr, dc, player_value):
        """Calculate line strength in given direction"""
        strength = 1  # Count the piece itself

        # Count in positive direction
        nr, nc = row + dr, col + dc
        while 0 <= nr < 6 and 0 <= nc < 7 and board[nr, nc] == player_value:
            strength += 1
            nr, nc = nr + dr, nc + dc

        # Count in negative direction
        nr, nc = row - dr, col - dc
        while 0 <= nr < 6 and 0 <= nc < 7 and board[nr, nc] == player_value:
            strength += 1
            nr, nc = nr - dr, nc - dc

        return strength

    def _get_possible_actions(self, board):
        """Get possible actions from board object"""
        return np.unique(np.where(board.board == 0)[1]).tolist()

    def _get_possible_actions_from_board(self, board_array):
        """Get possible actions from board array"""
        return np.unique(np.where(board_array == 0)[1]).tolist()

    def _can_win_with_move(self, board, action, player_value):
        """Check if move results in immediate win"""
        temp_board = board.board.copy()
        empty_rows = np.where(temp_board[:, action] == 0)[0]
        if len(empty_rows) == 0:
            return False

        row = np.max(empty_rows)
        temp_board[row, action] = player_value

        temp_board_obj = gamerules.Board()
        temp_board_obj.board = temp_board
        temp_board_obj.components = board.components.copy()
        temp_board_obj.components4 = board.components4.copy()
        temp_board_obj.updateComponents(row, action, player_value)
        temp_board_obj.updateComponents4(row, action, player_value)

        return temp_board_obj.checkVictory(action, player_value)

    def _would_win_after_move(self, board, action, player_value):
        """Check if move would result in win"""
        temp_board = board.board.copy()
        empty_rows = np.where(temp_board[:, action] == 0)[0]
        if len(empty_rows) == 0:
            return False

        row = np.max(empty_rows)
        temp_board[row, action] = player_value

        temp_board_obj = gamerules.Board()
        temp_board_obj.board = temp_board
        temp_board_obj.components = board.components.copy()
        temp_board_obj.components4 = board.components4.copy()
        temp_board_obj.updateComponents(row, action, player_value)
        temp_board_obj.updateComponents4(row, action, player_value)

        return temp_board_obj.checkVictory(action, player_value)


class Player(gamerules.Player):
    """Corrected player implementation addressing fundamental issues"""

    def __init__(self, name, weights_file=None):
        super().__init__(name)
        self.name = name
        self.weights_file = weights_file

        # Use stable DQN
        self.q_network = StableDQN(
            input_size=120,  # Reduced feature size for stability
            hidden_sizes=[256, 128],  # Simpler architecture
            output_size=7,
            learning_rate=0.0001,  # Conservative learning rate
            min_learning_rate=1e-6,
            learning_rate_patience=50
        )

        # Use improved heuristics
        self.heuristics = ImprovedHeuristics()

        # Decision tracking
        self.decision_stats = {
            'dqn_decisions': 0,
            'heuristic_overrides': 0,
            'override_types': {}
        }

        # Load weights
        if weights_file and os.path.exists(weights_file):
            if self.q_network.load_weights(weights_file):
                print(f"✅ Loaded weights from {weights_file}")
            else:
                print("⚠️ Failed to load weights, using random initialization")
        else:
            print("ℹ️ Using random weights")

    def getName(self):
        return self.name

    def newGame(self, new_opponent):
        pass

    def getAction(self, board, startValue):
        """Improved hybrid decision system"""
        possibleActions = self.getPossibleActions(board.board)

        if len(possibleActions) == 0:
            return 0

        try:
            # Critical heuristic overrides
            action, reason = self.heuristics.find_immediate_win(board, startValue)
            if action is not None:
                self._record_decision(reason)
                return action

            action, reason = self.heuristics.find_must_block(board, startValue)
            if action is not None:
                self._record_decision(reason)
                return action

            # Strategic heuristic guidance
            action, reason = self.heuristics.find_best_strategic_move(board, startValue)
            if action is not None:
                # Use heuristic guidance with some probability
                total_pieces = np.sum(board.board != 0)
                if total_pieces < 15:  # Early game - trust heuristics more
                    heuristic_probability = 0.7
                else:  # Later game - trust DQN more
                    heuristic_probability = 0.3

                if np.random.random() < heuristic_probability:
                    self._record_decision(reason)
                    return action

            # DQN decision with stability checks
            state = self._encode_state_stable(board, startValue)

            # Check for valid state encoding
            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                print("Warning: Invalid state encoding, using fallback")
                return self._fallback_action(possibleActions, startValue)

            q_values = self.q_network.predict(state)[0]

            # Check for valid Q-values
            if np.any(np.isnan(q_values)) or np.any(np.isinf(q_values)):
                print("Warning: Invalid Q-values, using fallback")
                return self._fallback_action(possibleActions, startValue)

            # Mask invalid actions
            q_values_masked = q_values.copy()
            for i in range(7):
                if i not in possibleActions:
                    q_values_masked[i] = float('-inf')

            # Add small noise for exploration
            noise = np.random.normal(0, 0.01, size=q_values_masked.shape)
            q_values_masked += noise

            action = np.argmax(q_values_masked)

            if action in possibleActions:
                self._record_decision("DQN")
                return int(action)
            else:
                return self._fallback_action(possibleActions, startValue)

        except Exception as e:
            print(f"Error in decision system: {e}")
            return self._fallback_action(possibleActions, startValue)

    def _fallback_action(self, possibleActions, startValue):
        """Improved fallback action selection"""
        # Intelligent fallback based on position
        if startValue == 1:  # First player
            preferences = [3, 2, 4, 1, 5, 0, 6]  # Center preference
        else:  # Second player
            preferences = [3, 4, 2, 1, 5, 0, 6]  # Slightly right-leaning center preference

        for col in preferences:
            if col in possibleActions:
                self._record_decision("FALLBACK")
                return col

        return possibleActions[0]

    def _encode_state_stable(self, board, startValue):
        """Stable state encoding with reduced dimensionality
        
        Returns a feature vector of size 99:
        - Board state (42 features)
        - Game analysis features (57 features)
        """
        features = []

        # Basic board state (42 features)
        board_normalized = board.board * startValue
        features.extend(board_normalized.flatten())

        # Game analysis features
        analysis_features = self._extract_stable_features(board, startValue)
        features.extend(analysis_features)

        # Ensure no NaN/inf values
        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        return features

    def _extract_stable_features(self, board, startValue):
        """Extract stable game features"""
        features = []

        # Basic game state (10 features)
        total_pieces = np.sum(board.board != 0)
        own_pieces = np.sum(board.board == startValue)
        opp_pieces = np.sum(board.board == -startValue)

        features.extend([
            total_pieces / 42.0,
            own_pieces / 21.0,
            opp_pieces / 21.0,
            (own_pieces - opp_pieces) / 21.0,
            1.0 if startValue == 1 else 0.0
        ])

        # Game phase encoding
        if total_pieces < 10:
            features.extend([1.0, 0.0, 0.0])
        elif total_pieces < 25:
            features.extend([0.0, 1.0, 0.0])
        else:
            features.extend([0.0, 0.0, 1.0])

        # Column analysis (14 features)
        for col in range(7):
            height = 6 - len(np.where(board.board[:, col] == 0)[0])
            features.append(height / 6.0)

        # Center control
        center_own = sum(np.sum(board.board[:, col] == startValue) for col in [2, 3, 4])
        center_total = sum(np.sum(board.board[:, col] != 0) for col in [2, 3, 4])
        center_ratio = center_own / max(center_total, 1)
        features.append(center_ratio)

        # Threat analysis (14 features)
        for col in range(7):
            can_win = self._can_win_in_column(board, col, startValue)
            must_block = self._can_win_in_column(board, col, -startValue)
            features.extend([1.0 if can_win else 0.0, 1.0 if must_block else 0.0])

        # Positional features (25 features)
        # Connectivity
        own_connections = self._count_connections(board, startValue)
        opp_connections = self._count_connections(board, -startValue)
        features.extend([
            own_connections / 20.0,
            opp_connections / 20.0
        ])

        # Pattern strength per column
        for col in range(7):
            pattern_strength = self._calculate_pattern_strength(board, col, startValue)
            features.append(pattern_strength)

        # Diagonal control
        main_diag = sum(1 for i in range(min(6, 7)) if i < 6 and board.board[i, i] == startValue)
        anti_diag = sum(1 for i in range(min(6, 7)) if i < 6 and (5 - i) < 7 and board.board[i, 5 - i] == startValue)
        features.extend([main_diag / 6.0, anti_diag / 6.0])

        # Edge vs center balance
        edge_pieces = np.sum(board.board[:, [0, 6]] == startValue)
        center_pieces = np.sum(board.board[:, [2, 3, 4]] == startValue)
        if edge_pieces + center_pieces > 0:
            balance = center_pieces / (edge_pieces + center_pieces)
        else:
            balance = 0.5
        features.append(balance)

        # Strategic factors (15 features)
        strategic_features = self._calculate_strategic_factors(board, startValue)
        features.extend(strategic_features)

        return features

    def _can_win_in_column(self, board, col, player_value):
        """Stable win detection"""
        if col not in self.getPossibleActions(board.board):
            return False

        try:
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
        except:
            return False

    def _count_connections(self, board, player_value):
        """Count piece connections"""
        connections = 0
        for row in range(6):
            for col in range(7):
                if board.board[row, col] == player_value:
                    for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < 6 and 0 <= nc < 7 and board.board[nr, nc] == player_value:
                            connections += 1
        return connections // 2

    def _calculate_pattern_strength(self, board, col, player_value):
        """Calculate pattern strength in column"""
        strength = 0.0
        consecutive = 0
        max_consecutive = 0

        for row in range(5, -1, -1):
            if board.board[row, col] == player_value:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            elif board.board[row, col] != 0:
                consecutive = 0

        return max_consecutive / 4.0

    def _calculate_strategic_factors(self, board, player_value):
        """Calculate strategic factors"""
        features = []

        # Mobility
        possible_moves = len(self.getPossibleActions(board.board))
        features.append(possible_moves / 7.0)

        # Tempo
        own_pieces = np.sum(board.board == player_value)
        opp_pieces = np.sum(board.board == -player_value)
        if own_pieces + opp_pieces > 0:
            tempo = own_pieces / (own_pieces + opp_pieces)
        else:
            tempo = 0.5
        features.append(tempo)

        # Threat density by region
        regions = [(0, 2), (2, 5), (5, 7)]  # Left, center, right
        for start, end in regions:
            threat_count = 0
            total_cells = 0
            for col in range(start, min(end, 7)):
                for row in range(6):
                    total_cells += 1
                    if board.board[row, col] == player_value:
                        # Simple threat calculation
                        adjacent_empty = 0
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = row + dr, col + dc
                            if 0 <= nr < 6 and 0 <= nc < 7 and board.board[nr, nc] == 0:
                                adjacent_empty += 1
                        if adjacent_empty > 0:
                            threat_count += 1

            threat_density = threat_count / max(total_cells, 1)
            features.append(threat_density)

        # Formation compactness
        if own_pieces > 0:
            # Calculate center of mass
            com_row = 0
            com_col = 0
            for row in range(6):
                for col in range(7):
                    if board.board[row, col] == player_value:
                        com_row += row
                        com_col += col
            com_row /= own_pieces
            com_col /= own_pieces

            # Calculate average distance from center of mass
            total_distance = 0
            for row in range(6):
                for col in range(7):
                    if board.board[row, col] == player_value:
                        distance = np.sqrt((row - com_row) ** 2 + (col - com_col) ** 2)
                        total_distance += distance

            compactness = 1.0 - (total_distance / own_pieces) / 10.0  # Normalize
            features.append(max(0.0, compactness))
        else:
            features.append(0.0)

        # Padding to reach 15 features
        while len(features) < 15:
            features.append(0.0)

        return features[:15]

    def _record_decision(self, decision_type):
        """Record decision statistics"""
        if decision_type == "DQN":
            self.decision_stats['dqn_decisions'] += 1
        else:
            self.decision_stats['heuristic_overrides'] += 1
            if decision_type not in self.decision_stats['override_types']:
                self.decision_stats['override_types'][decision_type] = 0
            self.decision_stats['override_types'][decision_type] += 1

    def getPossibleActions(self, board):
        """Get possible actions"""
        return np.unique(np.where(board == 0)[1])