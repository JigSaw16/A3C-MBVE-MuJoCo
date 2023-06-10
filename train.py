import gym
import torch
import numpy as np
from model import ActorCritic

import torch.optim as optim
import torch.nn.functional as nnfun
import torch.multiprocessing as mp

import matplotlib.pyplot as plt

# import threading


def select_action(state, policy_network):
    with torch.no_grad():
        # Przekształć stan na tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)

        # Oblicz wartości akcji przy użyciu sieci policy_network
        policy_probs = policy_network(state_tensor)

        # Wybierz akcję na podstawie rozkładu prawdopodobieństwa
        action = torch.multinomial(policy_probs, 1).item()

        return action


# Tworzenie środowiska Gym
env = gym.make('InvertedPendulum-v4')
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

# Hyperparametry
learning_rate = 0.0001
num_workers = 4  # torch.multiprocessing.cpu_count()
global_episodes = 10000
max_steps = 100
update_frequency = 20

actor_critic = ActorCritic(state_size, action_size)

global_theta = torch.zeros(1)  # Przykładowa inicjalizacja wartości początkowej
global_theta_v = torch.zeros(1)  # Przykładowa inicjalizacja wartości początkowej
# Tworzenie optymalizatorów dla parametrów θ i θv
optimizer_theta = optim.Adam([global_theta], lr=learning_rate)
optimizer_theta_v = optim.Adam([global_theta_v], lr=learning_rate)


# T_counter = 0
episode_rewards = []


# Funkcja pomocnicza dla agenta pracującego w środowisku
def worker_agent(agent_, env_, max_steps_, global_episodes_, idx):

    # global T_counter
    global episode_rewards

    for episode in range(global_episodes_):
        
        if episode % update_frequency == 0:
            # Reset gradients dθ and dθv
            optimizer_theta.zero_grad()  # Reset dθ
            optimizer_theta_v.zero_grad()  # Reset dθv

        # Synchronize thread-specific parameters θ0 = θ and θ0v = θv
        agent_.synchronize_parameters(global_theta, global_theta_v)

        # t_start = t_counter
        state = env_.reset()

        states, actions, rewards = [], [], []
        episode_reward = 0
        t_counter = 0
        done = False

        while not done:
            if len(state) != state_size:
                state = state[0]

            # with threading.Lock():
            state_tensor = torch.from_numpy(np.array([state], dtype=np.float32))
            action = agent_.sample_action(state_tensor)
            next_state, reward, done, _, info = env_.step(np.array([action]))

            episode_reward += reward
            t_counter += 1
            if t_counter >= max_steps_:
                break
            state = next_state

            with mp.Lock():
                actions.append(action)
                states.append(state)
                rewards.append(reward)

        if episode % 100 == 0 and idx == 0:
            episode_rewards.append(episode_reward)

        if done:
            R = 0  # cumulative reward
        else:
            # state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            state_tensor = torch.from_numpy(np.array(state, dtype=np.float32))
            _, value = agent_(state_tensor)
            R = value.item()

        for index in reversed(range(len(rewards))):
            R = rewards[index] + 0.9 * R  # Aktualizacja R

            # Obliczenie logarytmu prawdopodobieństwa akcji
            state_tensor = torch.from_numpy(np.array(states[index], dtype=np.float32))
            log_prob, value = agent_(state_tensor)
            advantage = R - value.item()

            # Obliczenie gradientów względem θ0
            policy_loss = -log_prob * advantage
            value_loss = nnfun.mse_loss(value, torch.tensor([R]))

            optimizer_theta.zero_grad()  # Reset gradientów θ0
            policy_loss.backward(retain_graph=True)  # Akumulacja gradientów θ0
            value_loss.backward(retain_graph=True)  # Akumulacja gradientów θ0v
            optimizer_theta.step()  # Aktualizacja parametrów θ0

            optimizer_theta_v.zero_grad()  # Reset gradientów θ0v
            value_loss.backward(retain_graph=True)  # Akumulacja gradientów θ0v
            optimizer_theta_v.step()  # Aktualizacja parametrów θ0v

        # Perform asynchronous update of θ using dθ and of θv using dθv.
        # Asynchroniczna aktualizacja θ
        optimizer_theta.zero_grad()
        optimizer_theta_v.zero_grad()

        # Aktualizacja θ
        for param in agent_.parameters():
            if param.grad is not None:
                param.grad.data *= -1  # Odwróć gradient
        optimizer_theta.step()

        # Aktualizacja θv
        for param in agent_.fc_value.parameters():
            if param.grad is not None:
                param.grad.data *= -1  # Odwróć gradient
        for param in agent_.fc_value_output.parameters():
            if param.grad is not None:
                param.grad.data *= -1  # Odwróć gradient
        optimizer_theta_v.step()

        if episode % 500 == 0:
            print(f"Thread {idx}, Episode {episode}: Total Reward = {episode_reward}")

        # if idx == 0 and episode % 10:
        #     episode_rewards.append(episode_reward)

        # # Wywołanie funkcji run_episode dla danego agenta i zwrócenie nagrody
        # episode_reward = run_episode(agent, env, max_steps)
        # print(f"Episode {episode}: Total Reward = {episode_reward}")

    plt.plot(episode_rewards)
    plt.title('learning')
    plt.ylabel('episode_reward')
    plt.xlabel('n_episode')
    plt.show()


if __name__ == '__main__':

    agents = []
    for _ in range(num_workers):
        agent = ActorCritic(state_size, action_size)
        agent.load_state_dict(actor_critic.state_dict())
        agents.append(agent)

    processes = []
    for i in range(num_workers):
        process = mp.Process(target=worker_agent, args=(agents[i], env, max_steps, global_episodes, i))
        processes.append(process)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    print("END")

    torch.save(actor_critic.state_dict(), 'wyuczone_wagi_steps_10000.pth')
