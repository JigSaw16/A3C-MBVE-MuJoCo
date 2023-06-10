import gym
import numpy as np
import tensorflow as tf
from model import ActorCritic

print("---------------------- TEST ----------------------")
# Testowanie modelu
test_episodes = 1000
test_rewards = []
steps = []

env_test = gym.make('InvertedPendulum-v4')  # , render_mode='human'
for _ in range(test_episodes):
    s_ = env_test.reset()
    episode_reward = 0
    done = False
    step = 0
    state_size = env_test.observation_space.shape[0]
    action_size = env_test.action_space.shape[0]
    best_network = ActorCritic(state_size, action_size)

    best_network.build(input_shape=(None, state_size))
    best_network.load_weights('wyuczone_wagi_steps_100t10.h5')

    while not done:
        if len(s_) != state_size:
            s_ = s_[0]

        policy, _ = best_network(tf.convert_to_tensor([s_], dtype=tf.float32))
        action = best_network.get_action(tf.convert_to_tensor([s_], dtype=tf.float32))
        s_, reward, done, _, info = env_test.step(np.array([action]))

        episode_reward += reward
        step += 1
    steps.append(step)
    test_rewards.append(episode_reward)

average_steps = np.mean(steps)
average_reward = np.mean(test_rewards)
print("Średnia nagroda po testowaniu:", average_reward)
print("Średnia ilość kroków po testowaniu:", average_steps)
