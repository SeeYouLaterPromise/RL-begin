import gym


class SkipFrameWrapper(gym.Wrapper):#跳过游戏中的帧数
#初始化方法 __init__
    def __init__(self,env,skip):

        super().__init__(env)
        self.skip = skip #每次step调用时重复执行动作的次数


    def step(self,action):
        obs,reward_total,done,info = None,0,False,None
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            reward_total+=reward
            if done:
                break

        return obs,reward_total,done,info