import gym


class SkipFrameWrapper(gym.Wrapper):

    def __init__(self,env,skip):

        super().__init__(env)
        self.skip = skip


    def step(self,action):
        obs,reward_total,done,info = None,0,False,None
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            reward_total+=reward
            if done:
                break

        return obs,reward_total,done,info