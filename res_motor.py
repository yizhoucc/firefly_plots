# behavior stats results
from locale import normalize
from operator import inv
import numpy as np
from plot_ult import * 
from scipy import stats 
from sklearn import svm
import matplotlib
from playsound import playsound
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import os
import pandas as pd
import numpy as np
from cmaes import CMA
import copy
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import torch
import heapq
from torch.distributions.multivariate_normal import MultivariateNormal
from matplotlib import pyplot as plt
import time
from stable_baselines3 import TD3
torch.manual_seed(0)
from numpy import linspace, pi
from InverseFuncs import *
from monkey_functions import *
from firefly_task import ffacc_real
from env_config import Config
# from cma_mpi_helper import run
import ray
from pathlib import Path
arg = Config()
import os
from timeit import default_timer as timer
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

env=ffacc_real.FireFlyPaper(arg)
env.debug=True
env.noise_scale=1
env.terminal_vel=0.2
agent_=TD3.load('trained_agent/paper.zip')
agent=agent_.actor.mu.cpu()

    
def relu(arr):
    arr[arr<0]=0
    return arr

thetask=[3,1]
ntrial=5
phi=torch.tensor([[1],   
                [pi/2],   
                [0.0],   
                [0.0],   
                [0.0],   
                [0.0],   
                [0.13],   
                [0.5],   
                [0.5],   
                [0.5],   
                [0.5]])


# --------------------------------
# low cost
theta=torch.tensor([[1],   
                        [pi/2],   
                        [0.2],   
                        [0.2],   
                        [0.0],   
                        [0.0],   
                        [0.13],   
                        [0.1],   
                        [0.1],   
                        [0.5],   
                        [0.5]])

given_obs=torch.vstack([torch.ones(size=(9,2))*0.9,0.1*torch.ones(size=(95,2))])
# given_obs[:,1]=0
states,actions, beliefs, covs=run_trials(agent=agent, env=env, phi=phi, theta=theta,task=thetask,ntrials=ntrial,given_obs=given_obs,return_belief=True)
mintime=min([len(s) for s in states])+5
states=[s[:mintime] for s in states]
actions=[s[:mintime] for s in actions]
quickoverhead_state(states, np.array([thetask]*2), color='b')
ax=quickoverhead_state(beliefs, np.array([thetask]*2), color='r')
# quicksave('low cost adjustment overhead', fig=ax.get_figure())
ax,_=plotctrlasd(actions)
# quicksave('low cost adjustment control',fig=ax.get_figure())
costs=[np.sum( (np.power(relu(np.diff(np.array(a), axis=0)),2)), axis=1) for a in actions]
minlen=min([len(a) for a in costs])
costsmu=[np.mean([a[t] for a in costs]) for t in range(minlen)]
costserr=[np.std([a[t] for a in costs]) for t in range(minlen)]
with initiate_plot(2,2,300) as fig:
    ax=fig.add_subplot(111)
    ax.errorbar(np.arange(len(costsmu)), costsmu, yerr=costserr)
    quickspine(ax)
    ax.set_xlabel('time, dt')
    ax.set_ylabel('costs')
    # quicksave('low cost vs time')


# high cost
theta=torch.tensor([[1],   
                        [pi/2],   
                        [0.2],   
                        [0.2],   
                        [0.0],   
                        [0.0],   
                        [0.13],   
                        [0.9],   
                        [0.9],   
                        [0.5],   
                        [0.5]])
given_obs=torch.vstack([torch.ones(size=(9,2))*0.9,0.1*torch.ones(size=(95,2))])
# given_obs[:,1]=0
states,actions, beliefs, covs=run_trials(agent=agent, env=env, phi=phi, theta=theta,task=thetask,ntrials=ntrial,given_obs=given_obs,return_belief=True)
mintime=min([len(s) for s in states])+5
states=[s[:mintime] for s in states]
actions=[s[:mintime] for s in actions]
quickoverhead_state(states, np.array([thetask]*2), color='b')
ax=quickoverhead_state(beliefs, np.array([thetask]*2), color='r')
# quicksave('high cost adjustment overhead', fig=ax.get_figure())
ax,_=plotctrlasd(actions)
# quicksave('high cost adjustment control',fig=ax.get_figure())
# ax,_=plotctrlasd([given_obs[:29]])
# quicksave('biased observation',fig=ax.get_figure())
costs=[np.sum( (np.power(relu(np.diff(np.array(a), axis=0)),2)), axis=1) for a in actions]
minlen=min([len(a) for a in costs])
costsmu=[np.mean([a[t] for a in costs]) for t in range(minlen)]
costserr=[np.std([a[t] for a in costs]) for t in range(minlen)]
with initiate_plot(2,2,300) as fig:
    ax=fig.add_subplot(111)
    ax.errorbar(np.arange(len(costsmu)), costsmu, yerr=costserr)
    quickspine(ax)
    ax.set_xlabel('time, dt')
    ax.set_ylabel('costs')
    # quicksave('high cost vs time')



# no adjustment
ntrial=5
theta=torch.tensor([[1],   
                        [pi/2],   
                        [0.001],   
                        [0.001],   
                        [0.9],   
                        [0.9],   
                        [0.13],   
                        [0.9],   
                        [0.9],   
                        [0.5],   
                        [0.5]])
states,actions, beliefs, covs=run_trials(agent=agent, env=env, phi=phi, theta=theta,task=thetask,ntrials=ntrial,pert=None,given_obs=None,return_belief=True)
mintime=min([len(s) for s in states])+5
states=[s[:mintime] for s in states]
actions=[s[:mintime] for s in actions]
ax=quickoverhead_state(states, np.array([thetask]*2), color='r')
# quicksave('no adjust adjustment overhead', fig=ax.get_figure())

costs=[np.sum( (np.power(relu(np.diff(np.array(a), axis=0)),2)), axis=1) for a in actions]
minlen=min([len(a) for a in costs])
costsmu=[np.mean([a[t] for a in costs]) for t in range(minlen)]
costserr=[np.std([a[t] for a in costs]) for t in range(minlen)]
with initiate_plot(2,2,300) as fig:
    ax=fig.add_subplot(111)
    ax.errorbar(np.arange(len(costsmu)), costsmu, yerr=costserr)
    quickspine(ax)
    ax.set_xlabel('time, dt')
    ax.set_ylabel('costs')
    # quicksave('no adjust cost vs time')

# quicksave('no adjust adjustment overhead', fig=ax.get_figure())
pert=torch.vstack([torch.ones(size=(9,2))*0.9,0.1*torch.ones(size=(95,2))])
states,actions, beliefs, covs=run_trials(agent=agent, env=env, phi=phi, theta=theta,task=thetask,ntrials=ntrial,pert=pert,given_obs=None,return_belief=True,given_action=actions[0])
ax=quickoverhead_state(beliefs, np.array([thetask]*2), color='r',ax=ax)
ax.get_figure()
# quicksave('no adjust adjustment overhead', fig=ax.get_figure())

# quicksave('no adjust control', fig=ax.get_figure())





costs=[np.sum( (np.power(relu(np.diff(np.array(a), axis=0)),2)), axis=1) for a in actions]
minlen=min([len(a) for a in costs])
costsmu=[np.mean([a[t] for a in costs]) for t in range(minlen)]
costserr=[np.std([a[t] for a in costs]) for t in range(minlen)]
with initiate_plot(2,2,300) as fig:
    ax=fig.add_subplot(111)
    ax.errorbar(np.arange(len(costsmu)), costsmu, yerr=costserr)
    quickspine(ax)
    ax.set_xlabel('time, dt')
    ax.set_ylabel('costs')
    # ax.fill_between(np.arange(len(costsmu)), np.array(costsmu)-np.array(costserr),np.array(costsmu)+np.array(costserr))






 
np.sum(np.sum( (np.power( relu(np.diff(np.array(a), axis=0)),2)), axis=1))


given_obs=torch.vstack([torch.ones(size=(0,2)),torch.zeros(size=(95,2))])
given_obs[:,1]=0
states,actions=run_trials(agent=agent, env=env, phi=phi, theta=theta,task=thetask,ntrials=ntrial,given_obs=given_obs)
mintime=min([len(s) for s in states])+5
states=[s[:mintime] for s in states]
actions=[s[:mintime] for s in actions]
quickoverhead_state(states, np.array([thetask]*2))
plotctrlasd(actions)

given_obs=torch.vstack([torch.ones(size=(10,2)),torch.zeros(size=(95,2))])
given_obs[:,1]=0
states,actions=run_trials(agent=agent, env=env, phi=phi, theta=theta,task=thetask,ntrials=ntrial,given_obs=given_obs)
mintime=min([len(s) for s in states])+9
states=[s[:mintime] for s in states]
actions=[s[:mintime] for s in actions]
quickoverhead_state(states, np.array([thetask]*2))
plotctrlasd(actions)


given_obs=torch.vstack([torch.ones(size=(20,2)),torch.zeros(size=(95,2))])
given_obs[:,1]=0
states,actions=run_trials(agent=agent, env=env, phi=phi, theta=theta,task=thetask,ntrials=ntrial,given_obs=given_obs)
mintime=min([len(s) for s in states])+5
states=[s[:mintime] for s in states]
actions=[s[:mintime] for s in actions]
quickoverhead_state(states, np.array([thetask]*2))
plotctrlasd(actions)

given_obs=torch.vstack([torch.ones(size=(25,2))*2,torch.zeros(size=(95,2))])
given_obs[:,1]=0
states,actions=run_trials(agent=agent, env=env, phi=phi, theta=theta,task=thetask,ntrials=ntrial,given_obs=given_obs)
mintime=min([len(s) for s in states])+5
states=[s[:mintime] for s in states]
actions=[s[:mintime] for s in actions]
quickoverhead_state(states, np.array([thetask]*2))
plotctrlasd(actions)




theta=torch.tensor([[1-0.2],   
                    [pi/2-0.3],   
                    [0.9],   
                    [0.9],   
                    [0.1],   
                    [0.1],   
                    [0.13],   
                    [1.5],   
                    [1.5],   
                    [0.5],   
                    [0.5]])

given_obs=torch.vstack([torch.ones(size=(10,2))*2,torch.zeros(size=(95,2))])
given_obs[:,1]=0
states,actions=run_trials(agent=agent, env=env, phi=phi, theta=theta,task=thetask,ntrials=ntrial,given_obs=given_obs)
mintime=min([len(s) for s in states])+9
states=[s[:mintime] for s in states]
actions=[s[:mintime] for s in actions]
quickoverhead_state(states, np.array([thetask]*2))
plotctrlasd(actions)


theta=torch.tensor([[1],   
                    [pi/2],   
                    [0.1],   
                    [0.1],   
                    [0.9],   
                    [0.9],   
                    [0.13],   
                    [0.1],   
                    [0.1],   
                    [0.5],   
                    [0.5]])

given_obs=torch.vstack([torch.ones(size=(6,2))*2,torch.zeros(size=(95,2))])
given_obs[:,1]=0
states,actions=run_trials(agent=agent, env=env, phi=phi, theta=theta,task=thetask,ntrials=ntrial,given_obs=given_obs,action_noise=0.)
mintime=min([len(s) for s in states])+9
states=[s[:mintime] for s in states]
actions=[s[:mintime] for s in actions]
quickoverhead_state(states, np.array([thetask]*2))
plotctrlasd(actions)



thetask=[0.7,0.5]
ntrial=20

theta=torch.tensor([[1],   
                    [pi/2],   
                    [0.9],   
                    [0.9],   
                    [0.1],   
                    [0.1],   
                    [0.13],   
                    [0.1],   
                    [0.1],   
                    [0.1],   
                    [0.1]])

given_obs=torch.vstack([torch.ones(size=(10,2))*2,torch.zeros(size=(95,2))])
given_obs[:,1]=0
pert=torch.zeros(size=(200,2))
pert[5:7,:]=0.3
states,actions=run_trials(agent=agent, env=env, phi=phi, theta=theta,task=thetask,ntrials=ntrial,given_obs=given_obs, pert=pert)
mintime=min([len(s) for s in states])+9
states=[s[:mintime] for s in states]
actions=[s[:mintime] for s in actions]
quickoverhead_state(states, np.array([thetask]*2))
plotctrlasd(actions)

env.reset(phi=phi, theta=theta, goal_position=thetask, pro_traj=given_obs,vctrl=0.,wctrl=0., obs_traj=given_obs)
epactions,_,_,epstates=run_trial(agent,env,given_action=None, given_state=None, action_noise=0,stimdur=None, pert=pert)
quickoverhead_state([torch.stack(epstates)[:,:,0]], np.array([thetask]))
env.pro_noisev
env.state_step(torch.ones(size=(1,2)),env.s)



