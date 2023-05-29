import threading
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# Definicja sieci neuronowej aktora (actor) i krytyka (critic)
class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, output_size)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value


# Funkcja wykonująca asynchroniczne zbieranie doświadczeń i aktualizację modelu
def train(rank_):
    env = gym.make('Hopper-v4')
    torch.manual_seed(rank_)

    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    max_episodes = 1000
    max_steps = 1000

    for episode in range(max_episodes):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []

        for steps in range(max_steps):
            print(state)
            state = torch.FloatTensor(state)
            logits, value = model(state)
            dist = Categorical(logits=logits)

            action = dist.sample()
            next_state, reward, done, _, info = env.step(action.item())

            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            state = next_state

            if done or steps == max_steps - 1:
                Qval = 0
                returns = []

                for r in reversed(rewards):
                    Qval = r + 0.99 * Qval
                    returns.insert(0, Qval)

                log_probs = torch.cat(log_probs)
                returns = torch.FloatTensor(returns)
                values = torch.cat(values)

                advantage = returns - values

                actor_loss = -(log_probs * advantage.detach()).mean()
                critic_loss = advantage.pow(2).mean()

                optimizer.zero_grad()
                loss = actor_loss + 0.5 * critic_loss
                loss.backward()
                optimizer.step()

                break


# Tworzenie i uruchomienie wątków zbierających doświadczenia
num_threads = 4
threads = []

for rank in range(num_threads):
    thread = threading.Thread(target=train, args=(rank,))
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()



# import gym
#
#
# def main():
#     env = gym.make('Hopper-v4', render_mode='human')  # utworzenie środowiska
#     env.reset()  # reset środowiska do stanu początkowego
#     for _ in range(1000):  # kolejne kroki symulacji
#         env.render()  # renderowanie obrazu
#         action = env.action_space.sample()  # wybór akcji (tutaj: losowa akcja)
#         observation, reward, done, _, info = env.step(action)  # wykonanie akcji
#         if done:
#             env.reset()
#     env.close()  # zamknięcie środowiska
#
#
# if __name__ == '__main__':
#     main()
