import gym
import torch as T
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

scores=[]

class SharedAdam(T.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
            weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = T.zeros_like(p.data)
                state['exp_avg_sq'] = T.zeros_like(p.data)
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
        osd = self.state_dict()
        for _, bufs in osd["state"].items():
            if "step" in bufs.keys():
                bufs["step"] = T.tensor(bufs["step"])


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super(ActorCritic, self).__init__()
        self.gamma = gamma
        self.pi1 = nn.Linear(*input_dims, 128)
        self.v1 = nn.Linear(*input_dims, 128)
        self.pi = nn.Linear(128, n_actions)
        self.v = nn.Linear(128, 1)
        self.rewards = []
        self.actions = []
        self.states = []
    

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)


    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []


    def forward(self, state):
        pi1 = F.relu(self.pi1(state))
        v1 = F.relu(self.v1(state))
        pi = self.pi(pi1)
        v = self.v(v1)

        return pi, v


    def calc_R(self, done):
        buf=np.array(self.states)
        states = T.tensor(buf, dtype=T.float)
        _, v = self.forward(states)
        R = v[-1]*(1-int(done))
        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma*R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float)

        return batch_return


    def calc_loss(self, done):
        buf=np.array(self.states)
        states = T.tensor(buf, dtype=T.float)
        actions = T.tensor(self.actions, dtype=T.float)
        returns = self.calc_R(done)
        pi, values = self.forward(states)
        values = values.squeeze()
        critic_loss = (returns-values)**2
        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*(returns-values)
        total_loss = (critic_loss + actor_loss).mean()
    
        return total_loss


    def choose_action(self, observation):
        observation=np.array([observation])
        state = T.tensor(observation, dtype=T.float)
        pi, v = self.forward(state)
        probs = T.softmax(pi,dim=1)
        dist = Categorical(probs)
        action = dist.sample().numpy()[0]

        return action


class Agent(mp.Process):
    def __init__(self, global_actor_critic:ActorCritic, optimizer:SharedAdam, input_dims, n_actions, 
                gamma, lr, name, global_ep_idx, env_id):
        super(Agent, self).__init__()
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
        self.global_actor_critic = global_actor_critic
        self.name = 'w%02i' % name
        self.episode_idx = global_ep_idx
        self.env = gym.make(env_id)
        self.optimizer = optimizer
        
    def run(self):
        t_step = 1
        render_human = False
        while self.episode_idx.value < N_GAMES:
            done = False
            observation = self.env.reset()
            observation=observation[0]
            score = 0
            self.local_actor_critic.clear_memory()
            while not done:
                action = self.local_actor_critic.choose_action(observation)
                observation_, reward, done, _, _ = self.env.step([(action-6)/2])
                score += reward
                high=False
                if score >= 5000:
                    high=True
                self.local_actor_critic.remember(observation, action, reward)
                if t_step % T_MAX == 0 or done or high:
                    loss = self.local_actor_critic.calc_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for local_param, global_param in zip(
                            self.local_actor_critic.parameters(),
                            self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad
                    self.optimizer.step()
                    self.local_actor_critic.load_state_dict(
                            self.global_actor_critic.state_dict())
                t_step += 1
                observation = observation_
                if high:
                    break

            with self.episode_idx.get_lock():
                with global_scores.get_lock():
                    global_scores.get_obj()[self.episode_idx.value-1]=score
                self.episode_idx.value += 1

            if t_step%100==0:
                print(self.name, 'episode', self.episode_idx.value, 'reward %.1f' % score)


if __name__ == '__main__':
    lr = 1e-4
    env_id = "InvertedPendulum-v4"
    n_actions = 13
    input_dims = [4]
    N_GAMES = 8000
    T_MAX = 100
    global_actor_critic = ActorCritic(input_dims, n_actions)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr, 
                        betas=(0.92, 0.999))
    global_ep = mp.Value('i', 0)
    global_scores= mp.Array('d',N_GAMES+10)
    workers = [Agent(global_actor_critic,
                    optim,
                    input_dims,
                    n_actions,
                    gamma=0.99,
                    lr=lr,
                    name=i,
                    global_ep_idx=global_ep,
                    env_id=env_id) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    
    [w.join() for w in workers]
    scores=[]
    x=[]
    for i in range(global_scores.get_obj()._length_):
        if global_scores.get_obj()[i] >0:
            x.append(i)
            scores.append(global_scores.get_obj()[i])

    T.save(global_actor_critic.state_dict(), 'A3C/model_weights.pth')
    coefficients = np.polyfit(x, scores, 6)
    trendline = np.polyval(coefficients, x)
    plt.scatter(x,scores)
    plt.plot(x, trendline, color='red')
    plt.show()

