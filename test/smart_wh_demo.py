# env
import gym
from gym import spaces
import numpy as np

print('''
set up the problem.
we have a home, with many rooms
each room has a pipe going from the heater to room.
we have pipes splited by a splitter near the pump
we have the pump at the heater.
the goal is to control the heater rate and pump flow rate and valuves, to provide desired temp for each room. 
with minimal time cost, and fuel/electricity cost.

state:
difference for each room from the desired temp.
current flow rates or valuves splitters
current heater rate

control:
heater rate
pump flow rate
valve splitter

state transition:
room
heat = flow rate * temp diff * room gain
heat loss = - room loss 
water heater
water heater outtemp = sum(heater rate / flow rate i + return temp i) for each room i 
pump and splitter
pipe i flow rate = splitter i * pump flow rate
''')


class Home(gym.Env):

    def __init__(self, nroom=1, outside=-20) -> None:
        super().__init__()
        self.outside=outside
        self.nroom = nroom
        ctrldim=nroom+2
        statedim=4*nroom+2
        low=-100
        high=100
        self.action_space = spaces.Box(low=low, high=high,shape=(ctrldim,))
        self.observation_space = spaces.Box(low=low, high=high,shape=(statedim,))
        self.reset()

        
        
    def reset(self,
        # roomgain=np.array([1,1,1.]),
        # roomloss=np.array([.1,.1,.1]),
        # roomtarget=np.array([25,25,22.]),
        # roominit=np.array([10,10,10.]),
        training=True
        ):
        
        if training:
            roomgain=np.random.random(self.nroom)
            roomloss=np.random.random(self.nroom)
            roomtarget=np.random.random(self.nroom)*5+22
            roominit=np.random.random(self.nroom)*10+5
        # state: tempdiff, spliters,roomgain,roomloss, pump, heater
        tardiff=roominit-roomtarget
        self.rooms=roominit
        spliters=np.ones(self.nroom)/self.nroom
        pump=np.zeros(1)
        heater=np.zeros(1)

        self.s=np.hstack([tardiff, spliters,roomgain,roomloss, pump, heater])

        self.timer=0
        self.rewardtimer=0
        self.totalreward=0
        
        return self.s


    def step(self, action, debug={}):
    
        self.timer+=1
        
        # vars
        tardiff=self.s[:self.nroom]
        # spliters=self.s[self.nroom:self.nroom*2]
        roomgain=self.s[2*self.nroom:self.nroom*3]
        roomloss=self.s[3*self.nroom:self.nroom*4]
        # pump=self.s[self.nroom*4]
        # heater=self.s[self.nroom*4+1]

        spliters=np.abs(action[:self.nroom])
        spliters=spliters/sum(spliters)
        pump=abs(action[self.nroom])
        heater=action[self.nroom+1]*100

        # heater temp
        heater=heater.clip(max(0,min(self.rooms)),100)
        # heaterreturns=heater-spliters*pump * tempdiff * roomgain

        # heat gain and loss for each room
        # gain = flow rate * temp diff * room gain
        tempdiff=heater-self.rooms
        gains=spliters*pump * tempdiff * roomgain
        loss=roomloss*(self.rooms-self.outside)
        tardiff+=gains
        tardiff-=loss
        self.rooms+=gains
        self.rooms-=loss

        self.s=np.hstack([tardiff, spliters,roomgain,roomloss, pump, heater/100])

        reward=self.reward()
        self.totalreward+=reward

        done=self.timer>100 or np.any(self.rooms>30) or np.any(self.rooms<self.outside+5)
        if done:
            print('end temp', self.rooms, 'at', self.timer, self.totalreward)
        return self.s.clip(0,100),reward,done,debug


    def reward(self):
        reward=np.zeros(1)
        
        # state cost
        # minimiz the target diff
        tardiff=self.s[:self.nroom]
        reward-=np.sum(np.power(tardiff,2))
        good=np.sum(np.power(tardiff,2))<self.nroom
        if good:
            self.rewardtimer+=1
        if self.rewardtimer>5:
            if good:
                reward+=20
            else:
                self.rewardtimer=0


        # minimize the time
        reward-=1

        # minimize the heater
        heater=self.s[self.nroom*4]
        reward-=(heater-np.mean(self.rooms))**2

        # minimize the pump 
        pump=self.s[self.nroom*4+1]
        reward-=pump**2


        return reward




# training for optimal control policy ----------------

from stable_baselines3 import PPO

# Parallel environments
env = Home()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)




obs = env.reset()
roomtemps=[]
done=False
while not done:
    roomtemps.append(env.rooms)
    action, _states = model.predict(obs)
    # action=abs(action)
    # action=(action*100).clip(-100,100)
    obs, rewards, done, info = env.step(action)

import matplotlib.pyplot as plt
roomtemps=np.array(roomtemps)
plt.plot(roomtemps)



env.reset()
# env.step(np.array([0.3,0.3,0.3, 3,33. ]))
# print(env.reward())
print(env.rooms)

self=env


