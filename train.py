import os
import gym
import keras
import numpy as np
import tensorflow as tf
import threading
from model import AC_Network
import random
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.multiprocessing import Process


def epsilon_greedy_action(action_probs, eps):
    if random.uniform(0, 1) < eps:
        # Wybierz losową akcję
        action = torch.randint(0, len(action_probs), (1,))
    else:
        # Wybierz akcję o najwyższej wartości prawdopodobieństwa
        _, action = torch.max(action_probs, dim=1)
    return action.item()


def compute_q_values(state, q_network):
    with torch.no_grad():
        q_values = q_network(state)
    return q_values


# --------------------------- 1. Inicjalizacja ---------------------------
# Definicja globalnych zmiennych
NUM_THREADS = 1
glob_env = gym.make('InvertedPendulum-v4')

global_episodes = 250  # Inicjalizacja globalnego licznika kroków.
global_network = ActorCritic(glob_env.observation_space.shape[0],
                             glob_env.action_space.shape[0])  # Inicjalizacja globalnego modelu sieci neuronowej.
n_updates = 1  # * 100 = 1000 for each agent
T = 0
T_max = 1000000
Itarget = 20
IAsyncUpdate = 250

global_network_lock = threading.Lock()
epsilon = 0.9
local_networks = []
best_reward = None
best_index = None
best_local_network = None
# episode_rewards = [0] * NUM_THREADS
episode_rewards = []


# --------------------------- 2. Asynchroniczne środowiska ---------------------------
# Funkcja agenta
def agent(thread_id):
    global global_episodes
    global T
    global T_max
    global best_reward, best_index, best_local_network
    global local_networks
    global episode_rewards

    t = 0
    env = gym.make('InvertedPendulum-v4')  # , render_mode='human'

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    # Przypisanie każdemu agentowi lokalnego modelu sieci neuronowej, który jest klonem globalnego modelu.
    local_network = global_network

    # optimizer = tf.optimizers.Adam(learning_rate=0.001)
    optimizer = torch.optim.Adam(local_network.parameters(), lr=0.001)
    # --------------------------- 3. Asynchroniczne uczenie ---------------------------
    while T < T_max:
        # Pobierz stan początkowy ze środowiska
        s = env.reset()
        episode_reward = 0
        # episode_steps = 0
        done = False

        # Dopóki nie osiągnięto pewnego warunku końcowego
        while not done:
            if len(s) != state_size:
                s = s[0]

            # # Take action a with E-greedy policy based on Q(s, a; θ)
            state_np = np.array([s], dtype=np.float32)
            state = torch.from_numpy(state_np)

            action_probs, _ = local_network(state)  # obliczanie prawdopodobieństw akcji
            action = epsilon_greedy_action(action_probs, epsilon)  # wybieranie akcji
            s_prim, reward, done, _, info = env.step(np.array([action]))

            if done:
                reward = -5
                target_value = [reward]
            else:
                next_value = local_network(torch.tensor(np.array(s_prim), dtype=torch.float32))
                target_value = reward + epsilon * torch.max(next_value[0])

            # policy, value = local_network(torch.tensor([s], dtype=torch.float32))
            value = local_network(torch.tensor(np.array(s), dtype=torch.float32))

            if type(target_value) == list:
                target_value = torch.tensor(target_value, dtype=torch.float32)
            else:
                target_value = target_value.clone().detach()
            loss = torch.mean((target_value - value[0]) ** 2)  # Obliczanie błędu kwadratowego

            if t % IAsyncUpdate == 0 or done:
                local_network.zero_grad()  # Wyzerowanie gradientów sieci lokalnej
                loss.backward()  # Obliczenie gradientów
                optimizer.step()  # Aktualizacja wag sieci

            episode_reward += reward
            s = s_prim

        T += 1
        t += 1

        # if T % Itarget == 0:
        #     global_network.load_state_dict(local_network.state_dict())
        #
        # if t % IAsyncUpdate == 0 or done:
        #     local_network.zero_grad()  # Wyzerowanie gradientów sieci lokalnej
        #     loss.backward()  # Obliczenie gradientów
        #     optimizer.step()  # Aktualizacja wag sieci

        episode_rewards.append(episode_reward)

        if T % 100 == 0:
            print("Thread: {}, Episode: {}/{}  Reward: {}".format(thread_id, T, T_max, episode_reward))

    local_networks.append(local_network)


# Tworzenie i uruchamianie wątków agentów
threads = []

agent(1)

plt.plot(episode_rewards)
plt.title('learning')
plt.ylabel('episode_reward')
plt.xlabel('n_episode')
plt.show()

# for rep in range(n_updates):
#     for i in range(NUM_THREADS):
#         thread = threading.Thread(target=agent, args=(i,))
#         threads.append(thread)
#
#     for thread in threads:
#         thread.start()
#
#     for thread in threads:
#         thread.join()
#
#     # --------------------------- 4. Aktualizacja globalnego modelu ---------------------------
#     best_reward = max(episode_rewards)
#     best_index = episode_rewards.index(best_reward)
best_local_net = local_networks[0]
with global_network_lock:
    global_network.set_weights(best_local_net.get_weights())
#
#     print(best_reward, best_index)

# threads = []
# local_networks = []

best_local_net.save_weights('wyuczone_wagi_steps_1000000.h5')
