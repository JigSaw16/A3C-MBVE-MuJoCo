import os

import gym
import keras
import numpy as np
import tensorflow as tf
import threading
from model import AC_Network

# --------------------------- 1. Inicjalizacja ---------------------------
# Definicja globalnych zmiennych

glob_env = gym.make('InvertedPendulum-v4')
glob_state_size = glob_env.observation_space.shape[0]
glob_action_size = glob_env.action_space.shape[0]
global_network = AC_Network(glob_state_size, glob_action_size)

global_network_lock = threading.Lock()
global_episodes = 200  # 10000
gamma = 0.95
update_frequency = 10
# global_episodes_lock = threading.Lock()
local_networks = []
episode_rewards = []
best_reward = None
best_index = None
best_local_network = None


# Funkcja agenta
def agent(thread_id):
    # --------------------------- 2. Asynchroniczne środowiska ---------------------------
    global global_episodes
    global best_reward, best_index, best_local_network
    global local_networks
    global episode_rewards

    # Utworzenie kilku asynchronicznych agentów, każdy w swoim środowisku.
    n_episodes = global_episodes
    n_episodes_lock = threading.Lock()
    env = gym.make('InvertedPendulum-v4')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    # Przypisanie każdemu agentowi lokalnego modelu sieci neuronowej, który jest klonem globalnego modelu.
    local_network = global_network

    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    # --------------------------- 3. Asynchroniczne uczenie ---------------------------
    while n_episodes > 0:
        # Pobierz stan początkowy ze środowiska
        s = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        local_network = global_network
        # Dopóki nie osiągnięto pewnego warunku końcowego
        while not done:
            with tf.GradientTape() as tape:
                if len(s) != state_size:
                    s = s[0]

                policy, value = local_network(tf.convert_to_tensor([s], dtype=tf.float32))
                action = np.random.choice(np.arange(action_size), p=policy[0])
                s1, reward, done, _, info = env.step(np.array([action]))

                if done:
                    reward = -1
                episode_reward += reward

                # --------------------------- 4. Aktualizacja globalnego modelu ---------------------------
                if episode_steps % update_frequency == 0 or done:
                    if done:
                        target_value = [reward]
                    else:

                        _, next_value = local_network(tf.convert_to_tensor([s1], dtype=tf.float32))
                        target_value = reward + gamma * next_value[0]

                    loss_policy = -tf.math.log(policy[0][action]) * (target_value - value[0])
                    loss_value = tf.square(target_value - value[0])
                    total_loss = loss_policy + loss_value

                    # Backpropagation
                    gradients = tape.gradient(total_loss, local_network.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, local_network.trainable_variables))

                    # Aktualizuj globalną sieć na podstawie najlepszej lokalnej sieci
                    with global_network_lock:
                        global_network.set_weights(local_network.get_weights())

            s = s1
            episode_steps += 1

        episode_rewards.append(episode_reward)
        local_networks.append(local_network)

        with n_episodes_lock:
            n_episodes -= 1
        print("Thread:", thread_id, "Episode:", n_episodes, "Reward:", episode_reward)

        if n_episodes == 0:
            best_reward = max(episode_rewards)
            best_index = episode_rewards.index(best_reward)
            best_local_network = local_networks[best_index]


# Tworzenie i uruchamianie wątków agentów
NUM_THREADS = 16
threads = []

for i in range(NUM_THREADS):
    thread = threading.Thread(target=agent, args=(i,))
    threads.append(thread)

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

best_local_network.save_weights('wyuczone_wagi_steps_200.h5')


