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
BATCH_SIZE = 10
optimizer = tf.keras.optimizers.Adam()
loss_op = tf.keras.losses.MSE
directions_indices = {k: i for i, k in enumerate(bp.Policy.TURNS)}

def vectorize_state(state):
    board, head = state
    head_pos, direction = head
    flat_board = board.flatten()
    head_dir = [*head_pos, directions_indices[direction]]
    state_vector = np.concatenate([head_dir,
                                   flat_board]).astype(np.float)
    return state_vector[np.newaxis, ...]

@tf.function
def train_step(model, reward, prev_state, next_state, gamma):
    prev_Q = model.predict(prev_state)
    with tf.GradientTape() as tape:
        new_Q = model.predict(next_state)
        loss = loss_op(prev_Q, reward + gamma * new_Q)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(grads, model.trainable_variables)

class Linear(bp.Policy):
    """
    A policy which avoids collisions with obstacles and other snakes. It has an epsilon parameter which controls the
    percentag of actions which are randomly chosen.
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['gamma'] = policy_args.get('gamma', MAX_BUFFER_SIZE)
        policy_args['buffer_size'] = policy_args.get('buffer_size', MAX_BUFFER_SIZE)
        policy_args['batch_size'] = policy_args.get('batch_size', BATCH_SIZE)

        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.state_buffer = []
        self.Q = QEstimator(len(bp.Policy.ACTIONS))

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

            batch = np.random.choice(self.state_buffer, self.batch_size)
            self.log("training")
            train_step(self.Q, reward, batch[..., 0], batch[..., 1], self.gamma)

        except Exception as e:
            self.log("Something Went Wrong...", 'EXCEPTION')
            self.log(e, 'EXCEPTION')

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        if too_slow:
            self.log("too slow, returning random choice")
            return np.random.choice(bp.Policy.ACTIONS)
        if np.random.rand() < self.epsilon:
            self.log("returning random choice")
            return np.random.choice(bp.Policy.ACTIONS)

        self.log("vectorizing")
        new_state_vector = vectorize_state(new_state)
        if prev_state is not None:
            prev_state_vector = vectorize_state(prev_state)

            if len(self.state_buffer) >= self.buffer_size:
                self.state_buffer = np.roll(self.state_buffer, -1)
                self.state_buffer[-1] = (prev_state_vector, new_state_vector)
            else:
                self.state_buffer.append((prev_state_vector, new_state_vector))
        self.log("predicting")
        pred = self.Q.predict(new_state_vector)
        action = bp.Policy.ACTIONS[np.argmax(pred)]
        self.log(f"Action taken: {action}")
        return action

def QEstimator(num_actions):
    model = Sequential()
    model.add(Dense(num_actions, activation='softmax'))
    return model
    # class QEstimator(Model):
    #
    #     def __init__(self, num_actions):
    #         super(Linear.QEstimator, self).__init__()
    #         self.dense = Dense(num_actions)
    #
    #     def call(self, x):
    #         return self.dense(x)
