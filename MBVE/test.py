from ddpg_torch import Agent
import gym
import numpy as np
from utils import plotLearning

"InvertedPendulum-v4"
env = gym.make("InvertedPendulum-v4",render_mode="human")
model=gym.make("InvertedPendulum-v4")
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[4], tau=0.001, env=env,model=model,H=10,
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=1,mode="Exploit")

agent.load_models()
np.random.seed(0)
# test=InvertedPendulumEnv()
# test.state_vector
score_history = []
for i in range(1000):
    obs = env.reset()[0]
    done = False
    score = 0
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done,truncated ,info = env.step(3*act)
        score += reward
        obs = new_state
        env.render()
    score_history.append(score)


    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

filename = 'new.png'
plotLearning(score_history, filename, window=100)
