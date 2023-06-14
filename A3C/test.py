import gym
import torch as T
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super(ActorCritic, self).__init__()
        self.gamma = gamma
        self.pi1 = nn.Linear(*input_dims, 128)
        self.v1 = nn.Linear(*input_dims, 128)
        self.pi = nn.Linear(128, n_actions)
        self.v = nn.Linear(128, 1)
        self.actions = []
    

    def forward(self, state):
        pi1 = F.relu(self.pi1(state))
        v1 = F.relu(self.v1(state))
        pi = self.pi(pi1)
        v = self.v(v1)
        return pi, v


    def choose_action(self, observation):
        observation=np.array([observation])
        state = T.tensor(observation, dtype=T.float)
        pi, _ = self.forward(state)
        probs = T.softmax(pi,dim=1)
        dist = Categorical(probs)
        action = dist.sample().numpy()[0]

        return action


class Agent(mp.Process):
    def __init__(self, global_actor_critic:ActorCritic, input_dims, n_actions, 
                gamma, lr, name, global_ep_idx, env_id):
        super(Agent, self).__init__()
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
        self.global_actor_critic = global_actor_critic
        self.name = 'w%02i' % name
        self.episode_idx = global_ep_idx
        self.env = gym.make(env_id, render_mode='human')
        
    def run(self):
        self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())
        while self.episode_idx.value < N_GAMES:
            done = False

            observation = self.env.reset()
            observation=observation[0]
            score = 0
            while not done:
                action = self.local_actor_critic.choose_action(observation)
                observation_, reward, done, _, _ = self.env.step([(action-6)/2])
                score += reward
                observation = observation_
            print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % score)

if __name__ == '__main__':
    lr = 1e-4
    env_id = "InvertedPendulum-v4"
    n_actions = 13
    input_dims = [4]
    N_GAMES = 8000
    global_actor_critic = ActorCritic(input_dims, n_actions)
    global_actor_critic.load_state_dict(T.load('A3C/model_weights.pth'))
    global_actor_critic.share_memory()
    global_ep = mp.Value('i', 0)
    global_scores= mp.Array('d',N_GAMES+10)

    worker = Agent(global_actor_critic,
                        input_dims,
                        n_actions,
                        gamma=0.99,
                        lr=lr,
                        name=0,
                        global_ep_idx=global_ep,
                        env_id=env_id)
    worker.start()
    worker.join()
