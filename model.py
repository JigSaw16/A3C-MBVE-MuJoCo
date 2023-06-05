import numpy as np
import tensorflow as tf


# Definicja modelu sieci neuronowej
class AC_Network(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(AC_Network, self).__init__()
        self.hidden = tf.Variable(tf.random.normal([state_size, 16]), name="hidden")
        self.output_policy = tf.Variable(tf.random.normal([16, action_size]), name="output_policy")
        self.output_value = tf.Variable(tf.random.normal([16, 1]), name="output_value")

    def call(self, inputs):
        hidden = tf.nn.relu(tf.matmul(inputs, self.hidden))
        policy = tf.nn.softmax(tf.matmul(hidden, self.output_policy))
        value = tf.matmul(hidden, self.output_value)
        return policy, value

    def get_action(self, state):
        policy, _ = self(state)
        return np.random.choice(np.arange(policy.shape[-1]), p=policy.numpy()[0])

    def update_network(self, network):
        self.local_network.set_weights(network.get_weights())
