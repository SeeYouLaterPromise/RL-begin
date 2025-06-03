import gym

from stable_baselines3 import A2C,PPO,DQN

env = gym.make("CartPole-v1")

model = A2C("MlpPolicy", env, verbose=1,tensorboard_log='logs')
model.learn(total_timesteps=1000000)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()