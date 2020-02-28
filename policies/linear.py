import os

from policies import base_policy as bp
import numpy as np
import tensorflow as tf
# from tensorflow.keras import Model, Sequential
# from tensorflow.keras.layers import Dense, Flatten
import tensorflow.keras as keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

EPSILON = 0.05
MAX_BUFFER_SIZE = 128
GAMMA = 0.1
LEARNING_RATE = 0.01
MAX_BATCH_SIZE = 16
directions_indices = {k: i for i, k in enumerate(bp.Policy.TURNS)}
action_indices = {k: i for i, k in enumerate(bp.Policy.ACTIONS)}

class Linear(bp.Policy):
    """
    A policy which avoids collisions with obstacles and other snakes. It has an epsilon parameter which controls the
    percentag of actions which are randomly chosen.
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args.get('eps', EPSILON))
        policy_args['gamma'] = float(policy_args.get('gamma', MAX_BUFFER_SIZE))
        policy_args['learning_rate'] = float(policy_args.get('lr', LEARNING_RATE))
        policy_args['buffer_size'] = int(policy_args.get('buffer_size', MAX_BUFFER_SIZE))
        policy_args['batch_size'] = int(policy_args.get('batch_size', MAX_BATCH_SIZE))

        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.state_buffer = []
        self.target_buffer = []
        self.Q = QEStimator(len(bp.Policy.ACTIONS), self.learning_rate)
        self.slow_counter = 0

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        try:
            if round % 100 == 0:
                if round > self.game_duration - self.score_scope:
                    self.log("Rewards in last 100 rounds which counts towards the score: " + str(self.r_sum),
                             'VALUE')
                else:
                    self.log("Rewards in last 100 rounds: " + str(self.r_sum), 'VALUE')
                self.r_sum = 0
            else:
                self.r_sum += reward

            if too_slow:
                self.batch_size = int(max(2, self.batch_size / 2))
                self.slow_counter = 0
            else:
                self.slow_counter += 1
                if self.slow_counter > 10:  # try to increase batch size
                    self.batch_size = int(min((MAX_BATCH_SIZE, self.batch_size * 2)))
            if self.state_buffer and self.target_buffer:
                self.Q.fit(np.asarray(self.state_buffer),
                           np.asarray(self.target_buffer),
                           batch_size=self.batch_size,
                           epochs=1, verbose=0)
        except Exception as e:
            self.log("Something Went Wrong...", 'EXCEPTION')
            self.log(e, 'EXCEPTION')

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if prev_state is not None and prev_action is not None:
            self.update_state_buffer(prev_state, prev_action, reward)
            if prev_action in ("L", "R"):
                sym_state, sym_action = get_symmetric_state(prev_state, prev_action)
                self.update_state_buffer(sym_state, sym_action, reward)

        if np.random.rand() < self.epsilon or too_slow or round < 50:
            return np.random.choice(bp.Policy.ACTIONS)

        pred = self.predict(new_state)
        action = bp.Policy.ACTIONS[np.argmax(pred)]

        return action

    def predict(self, new_state):

        pred = self.Q.predict(get_state(new_state)[np.newaxis, ...]).flatten()
        return pred


    def update_state_buffer(self, prev_state, prev_action, reward):
        state_vec, target = get_state_tuple(self.Q, prev_state, prev_action, reward, self.gamma)

        if len(self.state_buffer) >= self.buffer_size:
            self.state_buffer = np.roll(self.state_buffer, -1).tolist()
            self.state_buffer.pop()
            self.target_buffer = np.roll(self.target_buffer, -1).tolist()
            self.target_buffer.pop()
        self.state_buffer.append(state_vec)
        self.target_buffer.append(target)

def get_state(state):
    board, head = state
    head_pos, direction = head
    vec = np.zeros(3)
    for i, a in enumerate(bp.Policy.ACTIONS):
        new_pos = head_pos.move(bp.Policy.TURNS[direction][a])
        vec[i] = board[new_pos[0], new_pos[1]]
    return vec[np.newaxis, ...]

def get_state_tuple(model, state, action, reward, gamma):
    state_vector = get_state(state)
    Q = model.predict(state_vector[np.newaxis, ...]).flatten()
    Q_max_arg = action_indices[action]
    Q_max_val = Q[Q_max_arg]
    target = np.array(Q)
    target[Q_max_arg] = reward + gamma * Q_max_val
    return state_vector, target

def get_symmetric_state(state, action):
    sym_action = "L" if action is "R" else "R"
    idx_orig = action_indices[sym_action]
    idx_sym = action_indices[sym_action]
    sym_state = np.array(state)
    sym_state[idx_orig], sym_state[idx_sym] = state[idx_sym], state[idx_orig]
    return sym_state, sym_action


def QEStimator(num_actions, lr):
    model = Sequential()
    model.add(Dense(num_actions))
    model.compile(loss='mean_squared_error',
                  optimizer=keras.optimizers.Adam(lr))
    return model
