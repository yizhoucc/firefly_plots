

print('''
version 2, 
model the 2 noise firefly task with real units and real dynamics.
try my best not to use approximations, translate everything from the vr code

''')
import numpy as np
from matplotlib import pyplot as plt
from plot_ult import *

savedTau=1
meandist=25
meantime=2.5

def calculatevmax():
    maxspeed=(meandist/meantime)* (1.0 / (-1. + (2 * (savedTau / meantime)) * np.log((1 + np.exp(meantime / savedTau)) / 2.) ))
    return maxspeed


global MaxSpeed
# MaxSpeed=max(30, calculatevmax())
MaxSpeed=200


# world model ----------------------------

class Dynamic:
    def __init__(self, tau, dt=0.01):
        self.tau=tau
        self.kappa=self.getkappa(dt=dt)

    def getkappa(self,dt=0.01):
        # get alpha from tau. x'=x*a + (1-a)u
        kappa = np.exp(-dt/self.tau)
        return kappa
    
class ObsNoise(Dynamic):

    def __init__(self, tau, gain=8,dt=0.01):
        super().__init__(tau,dt)
        self.prevVelObsEps=0
        self.prevVelObsZet=0
        self.gain=gain

    def __call__(self,joysticknorm):
        lamda = 1 - self.kappa
        epsilon = self.kappa * self.prevVelObsEps + lamda * self.gain * np.random.normal()
        zeta = self.kappa * self.prevVelObsZet + lamda * epsilon
        ObsNoiseMagnitude = joysticknorm
        result_speed = zeta * ObsNoiseMagnitude * MaxSpeed
        self.prevVelObsEps = epsilon
        self.prevVelObsZet = zeta
        return result_speed

print('''
check, the obs noise of 10s, tau=1s, given max joystick input
''')
on=ObsNoise(1)
plt.xlabel('time, s')
plt.ylabel('obs noise, cm/s')
quickspine(plt.gca())
plt.plot(np.linspace(0,10,1000),[on(1) for _ in range(1000)]); plt.show()



class ProcessNoise(Dynamic):

    def __init__(self, tau, gain=5, dt=0.01):
        super().__init__(tau,dt)
        self.prevVelKsi=0
        self.prevVelEta=0
        self.prevCleanVel=0
        self.velProcessNoiseGain=gain

    def __call__(self,cleanVel):
        gamma = self.kappa
        delta = (1.0 - gamma)
        velKsi = gamma * self.prevVelKsi + delta * self.velProcessNoiseGain * np.random.normal()
        velEta = gamma * self.prevVelEta + delta * velKsi
        self.prevCleanVel = cleanVel
        self.prevVelKsi = velKsi
        self.prevVelEta = velEta
        ProcessNoiseMagnitude = cleanVel / MaxSpeed
        if np.abs(velEta * ProcessNoiseMagnitude) > MaxSpeed:
            currentSpeed = cleanVel + np.sign(velEta) * cleanVel
        else:
            # print(cleanVel, velEta ,ProcessNoiseMagnitude , MaxSpeed)
            currentSpeed = cleanVel + velEta * ProcessNoiseMagnitude * MaxSpeed
        return currentSpeed


print('''
check, the process noise of 10s, tau=1s given max clean velocity (200cm/s)''')
pn=ProcessNoise(1)
quickspine(plt.gca())
plt.xlabel('time, s')
plt.ylabel('process noise, cm/s')
plt.plot(np.linspace(0,10,1000),[pn(200) for _ in range(1000)]); plt.show()



class System(Dynamic):

    def __init__(self, tau, dt=0.01):
        super().__init__(tau,dt)
        self.prevCleanVel=0
    
    def __call__(self,moveX):
        cleanVel = self.kappa * self.prevCleanVel + MaxSpeed * (1-self.kappa) * moveX
        self.prevCleanVel = cleanVel
        return cleanVel


print('''
check, system clean velocity of 10s, tau=1s given max joytick input''')
quickspine(plt.gca())
plt.xlabel('time, s')
plt.ylabel('clean velocity, cm/s')
sys=System(1)
plt.plot(np.linspace(0,10,1000),[sys(1) for _ in range(1000)]); plt.show()

from collections import defaultdict

class Task:
    # def __init__(self, ctrlgain, obsgain, processgain, ctrltau, obstau, processtau) -> None:
    def __init__(self,**kwargs) -> None:
        self._kwargs=kwargs
        self.reset()

    def reset(self,):
        # reset log
        self.log=defaultdict(list)
        # for k,v in self._kwargs:
        #     if k.startswith('log_') and v:
        #         self.log[k]=[]

        # reset system and noises
        self.sys=System(self._kwargs['ctrltau'],self._kwargs['dt'])
        self.pn=ProcessNoise(self._kwargs['processtau'],self._kwargs['processgain'],self._kwargs['dt'])
        self.on=ObsNoise(self._kwargs['obstau'],self._kwargs['obsgain'],self._kwargs['dt'])

    def __call__(self, joystickctrl):
        cleanvel=self.sys(joystickctrl)
        pn=self.pn(cleanvel)
        on=self.on(joystickctrl)

        self.log['cleanvel'].append(cleanvel)
        self.log['pn'].append(on)
        self.log['on'].append(pn)
        self.log['ctrl'].append(joystickctrl)
        self.log['trueopticflow'].append(pn)
        self.log['distractor'].append(pn+on)
        self.log['distance'].append(sum(self.log['trueopticflow'])*self._kwargs['dt'])

print('''
check, use max ctrl to play with the task, and see the prediction and 2 optic flows''')

kwargs={
    'ctrlgain':1, 
    'obsgain':8, 
    'processgain':5, 
    'ctrltau':1, 
    'obstau':1, 
    'processtau':1,
    'dt':0.01
}
task=Task(**kwargs)
for _ in range(1000):
    task(1) 

# task=Task(**kwargs)
# for _ in range(700):
#     task(1) 
# for _ in range(300):
#     task(-1) 

vars=['cleanvel', 'trueopticflow', 'distractor']
for k in vars:
    plt.plot(np.linspace(0,10,1000),task.log[k], label=k); 
quickleg(plt.gca(), bbox_to_anchor=[1,1])
quickspine(plt.gca())
plt.xlabel('time, s')
plt.ylabel('cm/s')
plt.show()

vars=['distance']
for k in vars:
    plt.plot(np.linspace(0,10,1000),task.log[k], label=k); 
quickleg(plt.gca(), bbox_to_anchor=[1,1])
quickspine(plt.gca())
plt.xlabel('time, s')
plt.ylabel('cm')
plt.show()





# belief (internal) model ---------------------------

class Agent:

    def __init__(self,**kwargs) -> None:
        self._kwargs=kwargs
        self.reset()

    def reset(self,):
        self.log=defaultdict(list)

        # reset system and noises
        self.sys=System(self._kwargs['ctrltau'],self._kwargs['dt'])
        self.pn=ProcessNoise(self._kwargs['processtau'],self._kwargs['processgain'],self._kwargs['dt'])
        self.on=ObsNoise(self._kwargs['obstau'],self._kwargs['obsgain'],self._kwargs['dt'])

        self._on_obs=[]
        self._on_belief=[]
        self.on_obs(self._kwargs['obs_fn'])
        self.on_belief(self._kwargs['belief_fn'])

    def __call__(self, joystickctrl):
        cleanvel=self.sys(joystickctrl)
        pn=self.pn(cleanvel)
        on=self.on(joystickctrl)

        self.log['cleanvel'].append(cleanvel*self._kwargs['ctrlgain'])
        self.log['pn'].append(on)
        self.log['on'].append(pn)
        self.log['ctrl'].append(joystickctrl)
        self.log['trueopticflow'].append(pn)
        self.log['distractor'].append(pn+on)
        self.log['distance'].append(sum(self.log['trueopticflow'])*self._kwargs['dt'])

        # visual velocity preceptions
        if len(self._on_obs)==1:
            mu,var=self._on_obs[0](pn, pn+on, self._kwargs['trueoptvar'], self._kwargs['distoptvar'])
            self.log['obsvelpreceptionmu'].append(mu)
            self.log['obsvelpreceptionvar'].append(var)

        # belief velocity preceptions
        if len(self._on_belief)==1:
            mu,var=self._on_obs[0](cleanvel*self._kwargs['ctrlgain'], mu, self._kwargs['predictionvar'], var)
            self.log['bvelpreceptionmu'].append(mu)
            self.log['bvelpreceptionvar'].append(var)

        self.log['distance_hatmu'].append(sum(self.log['bvelpreceptionmu'])*self._kwargs['dt'])
        self.log['distance_hatvar'].append(sum(np.array(self.log['bvelpreceptionvar'])*(self._kwargs['dt'])))
            
    def on_obs(self, callback):
        self._on_obs.append(callback)

    def on_belief(self, callback):
        self._on_belief.append(callback)


def mix(mu1, mu2, var1, var2):
    p=var2/(var1+var2)
    mu=(mu1*p+mu2*(1-p))
    var=(var1*p+var2*(1-p)) + (mu1-mu2)**2
    return mu, var

def integration(mu1, mu2, var1, var2):
    mu=(mu1*var2+mu2*var1)/(var1+var2)
    var=(var1+var2)/(var1*var2)
    return mu, var


assumptiom_kwargs={
    'ctrlgain':1, 
    'obsgain':8, 
    'processgain':5, 
    'ctrltau':1, 
    'obstau':1, 
    'processtau':1,
    'dt':0.01,
    # uncertainty
    'trueoptvar':66,
    'distoptvar':66,
    'predictionvar':666,
    # obs and belief update strategy
    'obs_fn': mix,
    'belief_fn': integration
}
agent=Agent(**assumptiom_kwargs)
for _ in range(1000):
    agent(1) 

vars=['cleanvel', 'trueopticflow', 'distractor','obsvelpreceptionmu','bvelpreceptionmu',]
vars=['trueopticflow', 'distractor','obsvelpreceptionmu',]
vars=['cleanvel','obsvelpreceptionmu','bvelpreceptionmu',]
for k in vars:
    if k.endswith('mu') and k[:-2]+'var' in agent.log:
        stds=np.array(agent.log[k[:-2]+'var'])**0.5
        upper=np.array(agent.log[k])+stds
        lower=np.array(agent.log[k])-stds
        plt.fill_between(np.linspace(0,10,1000), lower, upper, alpha=0.4, label=k)
    else:
        plt.plot(np.linspace(0,10,1000),agent.log[k], label=k); 
quickleg(plt.gca(), bbox_to_anchor=[1,1])
quickspine(plt.gca())
plt.xlabel('time, s')
plt.ylabel('cm/s')
plt.show()

vars=['distance','distance_hatmu' ]
for k in vars:
    if k.endswith('mu') and k[:-2]+'var' in agent.log:
        stds=np.array(agent.log[k[:-2]+'var'])**0.5
        upper=np.array(agent.log[k])+stds
        lower=np.array(agent.log[k])-stds
        plt.fill_between(np.linspace(0,10,1000), lower, upper, alpha=0.4, label=k)
    else:
        plt.plot(np.linspace(0,10,1000),agent.log[k], label=k); quickleg(plt.gca(), bbox_to_anchor=[1,1])
quickspine(plt.gca())
plt.xlabel('time, s')
plt.ylabel('cm')
plt.show()



# ---------------------------
# if agnet has a different assumption than ground truth

kwargs={
    'ctrlgain':1, 
    'obsgain':8, 
    'processgain':5, 
    'ctrltau':1, 
    'obstau':1, 
    'processtau':1,
    'dt':0.01
}
task=Task(**kwargs)
for _ in range(1000):
    task(1) 

# vars=['cleanvel', 'trueopticflow', 'distractor']
# for k in vars:
#     plt.plot(np.linspace(0,10,1000),task.log[k], label=k); 
# quickleg(plt.gca(), bbox_to_anchor=[1,1])
# quickspine(plt.gca())
# plt.xlabel('time, s')
# plt.ylabel('cm/s')
# plt.show()


assumptiom_kwargs={
    'ctrlgain':0.5, 
    'obsgain':8, 
    'processgain':5, 
    'ctrltau':1, 
    'obstau':1, 
    'processtau':1,
    'dt':0.01,
    # uncertainty
    'trueoptvar':33,
    'distoptvar':33,
    'predictionvar':33,
    # obs and belief update strategy
    'obs_fn': mix,
    'belief_fn': integration
}
agent=Agent(**assumptiom_kwargs)
for _ in range(1000):
    agent(1) 


vars=['cleanvel']
for k in vars:
    plt.plot(np.linspace(0,10,1000),task.log[k], label=k, color='k')
# vars=['cleanvel', 'trueopticflow', 'distractor','obsvelpreceptionmu','bvelpreceptionmu',]
# vars=['trueopticflow', 'distractor','obsvelpreceptionmu',]
vars=['cleanvel','obsvelpreceptionmu','bvelpreceptionmu',]
for k in vars:
    if k.endswith('mu') and k[:-2]+'var' in agent.log:
        stds=np.array(agent.log[k[:-2]+'var'])**0.5
        upper=np.array(agent.log[k])+stds
        lower=np.array(agent.log[k])-stds
        plt.fill_between(np.linspace(0,10,1000), lower, upper, alpha=0.4, label=k)
    else:
        plt.plot(np.linspace(0,10,1000),agent.log[k], label=k); 
quickleg(plt.gca(), bbox_to_anchor=[1,1])
quickspine(plt.gca())
plt.xlabel('time, s')
plt.ylabel('cm/s')
plt.show()


vars=['distance']
for k in vars:
    plt.plot(np.linspace(0,10,1000),task.log[k], label=k); 
vars=['distance_hatmu' ]
for k in vars:
    if k.endswith('mu') and k[:-2]+'var' in agent.log:
        stds=np.array(agent.log[k[:-2]+'var'])**0.5
        upper=np.array(agent.log[k])+stds
        lower=np.array(agent.log[k])-stds
        plt.fill_between(np.linspace(0,10,1000), lower, upper, alpha=0.4, label=k)
    else:
        plt.plot(np.linspace(0,10,1000),agent.log[k], label=k); quickleg(plt.gca(), bbox_to_anchor=[1,1])
quickspine(plt.gca())
plt.xlabel('time, s')
plt.ylabel('cm')
plt.show()

# 0130 notes
1, prediction change. 
prediction is based on previous belief of velocity, not purelly clean velocity
2, the variance of the obs
the variance may not be the calculation from instentious var calculation, but based on history