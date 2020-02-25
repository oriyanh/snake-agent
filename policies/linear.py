import os

from policies import base_policy as bp
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

EPSILON = 0.1
MAX_BUFFER_SIZE = 256
GAMMA = 0.99
LEARNING_RATE = 0.001
MAX_BATCH_SIZE = 64
optimizer = tf.keras.optimizers.Adam()
loss_op = tf.keras.losses.MSE
directions_indices = {k: i for i, k in enumerate(bp.Policy.TURNS)}

def vectorize_state(state):
    try:
        board, head = state
        head_pos, direction = head
        flat_board = board.flatten()
        head_dir = [*head_pos, directions_indices[direction]]
        state_vector = np.concatenate([head_dir,
                                       flat_board]).astype(np.float)
        # return state_vector[np.newaxis, ...]
        return state_vector[np.newaxis, ...].T
    except Exception as e:
        print(e)

# @tf.function
# def train_step(model, reward, prev_state, next_state, gamma):
#     prev_Q = model.predict(prev_state)
#     with tf.GradientTape() as tape:
#         new_Q = model.predict(next_state)
#         loss = loss_op(prev_Q, reward + gamma * new_Q)
#     grads = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))

def train_step(model, reward, prev_state, next_state, gamma, lr):
    prev_Q = np.matmul(model, prev_state)
    prev_Q_arg = np.argmax(prev_Q[:, 0])
    new_Q = np.matmul(model, next_state)
    next_Q_val = np.max(new_Q[:, 0])
    delta = lr * (reward + gamma * next_Q_val - prev_Q[prev_Q_arg, 0])
    model[prev_Q_arg, 0] += delta
    model /= np.linalg.norm(model)
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
        # self.Q = QEstimator(len(bp.Policy.ACTIONS))
        # self.Q.compile(optimizer, loss_op)
        self.Q = QEstimator(self.board_size, len(self.ACTIONS))
        self.slow_counter = 0

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if too_slow:
            self.buffer_size = int(max(2, self.buffer_size / 2))
            self.slow_counter = 0
        else:
            if self.slow_counter > 10:
                self.buffer_size = int(min((MAX_BUFFER_SIZE, self.buffer_size * 2)))

        # try:
            if round % 100 == 0:
                if round > self.game_duration - self.score_scope:
                    self.log("Rewards in last 100 rounds which counts towards the score: " + str(self.r_sum),
                             'VALUE')
                else:
                    self.log("Rewards in last 100 rounds: " + str(self.r_sum), 'VALUE')
                self.r_sum = 0
            else:
                self.r_sum += reward
            batch = np.random.permutation(self.state_buffer)[:self.buffer_size]
            self.log("training")
            for prev_s, next_s in batch:
                train_step(self.Q, reward, prev_s, next_s, self.gamma, self.learning_rate)

        # except Exception as e:
        #     self.log("Something Went Wrong...", 'EXCEPTION')
        #     self.log(e, 'EXCEPTION')

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if too_slow:
            self.log(f"Round #{round}: too slow, returning random choice.")
            return np.random.choice(bp.Policy.ACTIONS)
        if np.random.rand() < self.epsilon:
            self.log(f"Round #{round}: under epsilon, returning random choice")
            return np.random.choice(bp.Policy.ACTIONS)

        new_state_vector = vectorize_state(new_state)
        if prev_state is not None:
            prev_state_vector = vectorize_state(prev_state)

            if len(self.state_buffer) >= self.buffer_size:
                self.state_buffer = np.roll(self.state_buffer, -1)
                self.state_buffer[-1] = (prev_state_vector, new_state_vector)
            else:
                self.state_buffer.append((prev_state_vector, new_state_vector))
        if round < 100:
            self.log(f"Round #{round}: Low round, returning random choice.")
            return np.random.choice(bp.Policy.ACTIONS)
        # pred = self.Q.predict(new_state_vector)
        pred = np.matmul(self.Q, new_state_vector)
        action = bp.Policy.ACTIONS[np.argmax(pred[:, 0])]
        self.log(f"Round #{round}: predicting. Action taken: {action}")
        return action

def QEstimator(board_size, num_actions):
    # model = Sequential()
    # model.add(Dense(num_actions))
    # return model


    # class QEstimator(Model):
    #
    #     def __init__(self, num_actions):
    #         super(Linear.QEstimator, self).__init__()
    #         self.dense = Dense(num_actions)
    #
    #     def call(self, x):
    #         return self.dense(x)

    Q_table = np.random.uniform(size=(num_actions, board_size[0]*board_size[1] + 3))
    return Q_table
