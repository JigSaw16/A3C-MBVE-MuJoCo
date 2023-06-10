import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

# Definicja modelu sieci neuronowej
# class ActorCritic(tf.keras.Model):
#     def __init__(self, state_size, action_size):
#         super(ActorCritic, self).__init__()
#         self.hidden = tf.Variable(tf.random.normal([state_size, 16]), name="hidden")
#         self.output_policy = tf.Variable(tf.random.normal([16, action_size]), name="output_policy")
#         self.output_value = tf.Variable(tf.random.normal([16, 1]), name="output_value")
#
#     def call(self, inputs):
#         hidden = tf.nn.relu(tf.matmul(inputs, self.hidden))
#         policy = tf.nn.softmax(tf.matmul(hidden, self.output_policy))
#         value = tf.matmul(hidden, self.output_value)
#         return policy, value
#
#     def get_action(self, state):
#         policy, _ = self(state)
#         return np.random.choice(np.arange(policy.shape[-1]), p=policy.numpy()[0])
#
#     def update_network(self, network):
#         self.local_network.set_weights(network.get_weights())


# Globalna sieć neuronowa (actor-critic)
class AC_Network(nn.Module):
    def __init__(self, state_size, action_size):
        super(AC_Network, self).__init__()
        self.dense1 = nn.Linear(state_size, 256)
        self.policy_logits = nn.Linear(256, action_size)
        self.dense2 = nn.Linear(256, 256)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        policy = self.policy_logits(x)
        x = F.relu(self.dense2(x))
        value = self.value(x)
        return policy, value

# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(ActorCritic, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 64)
#         self.fc2 = nn.Linear(64, 64)
#
#         # Warstwa liniowa dla polityki (aktora)
#         self.policy = nn.Linear(64, action_dim)
#
#         # Warstwa liniowa dla wartości (krytyka)
#         self.value = nn.Linear(64, 1)
#
#     def forward(self, state):
#         x = torch.relu(self.fc1(state))
#         x = torch.relu(self.fc2(x))
#
#         # Polityka (aktor)
#         policy_output = self.policy(x)
#         action_probs = F.softmax(policy_output, dim=-1)
#
#         # Wartość (krytyk)
#         value = self.value(x)

        # return action_probs, value

# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(ActorCritic, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, action_dim)
#
#     def forward(self, state):
#         x = torch.relu(self.fc1(state))
#         x = torch.relu(self.fc2(x))
#         q_values = self.fc3(x)
#         return q_values

# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(ActorCritic, self).__init__()
#
#         # Warstwy sieci policy
#         self.fc_policy1 = nn.Linear(state_dim, 64)
#         self.fc_policy2 = nn.Linear(64, 64)
#         self.fc_policy3 = nn.Linear(64, action_dim)
#         self.softmax = nn.Softmax(dim=1)
#
#         # Warstwy sieci krytyka
#         self.fc_value1 = nn.Linear(state_dim, 64)
#         self.fc_value2 = nn.Linear(64, 64)
#         self.fc_value3 = nn.Linear(64, 1)
#
#     def forward(self, state):
#         # Obliczenia dla sieci policy
#         x_policy = torch.relu(self.fc_policy1(state))
#         x_policy = torch.relu(self.fc_policy2(x_policy))
#         logits = self.fc_policy3(x_policy)
#         action_probs = self.softmax(logits)
#
#         # Obliczenia dla sieci krytyka
#         x_value = torch.relu(self.fc_value1(state))
#         x_value = torch.relu(self.fc_value2(x_value))
#         state_value = self.fc_value3(x_value)
#
#         return action_probs, state_value
