import os
import pandas as pd
from numpy.lib.npyio import save
import numpy as np
from cmaes import CMA
import copy
import matplotlib.pyplot as plt
# from numpy.core.defchararray import array
# from FireflyEnv.env_utils import is_pos_def
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
torch.manual_seed(42)
from numpy import pi
from InverseFuncs import *
from monkey_functions import *
from firefly_task import ffacc_real
from env_config import Config
from notification import notify
# from cma_mpi_helper import run
import ray
from pathlib import Path
arg = Config()
import os
import requests
import configparser
from plot_ult import *
config = configparser.ConfigParser()
config.read_file(open('privateconfig'))
token=config['Notification']['token']


print('loading data')
datapath=Path("/data/neuraltest/1208pack")
with open(datapath,'rb') as f:
    states, actions, tasks = pickle.load(f)

# try use the longest trial (larger belief mismatch)
sortind=np.argsort([len(s) for s in states])



maskind=[]
# keep error trial
err=np.array([torch.norm(resp[-1,:2]-tar) for resp, tar in zip(states, tasks)])
maskind+=list(np.where(err>0.4)[0])
# keep long trials
# ts=np.array([len(s) for s in states])
# maskind+=list(np.where(ts<6)[0])

mask = np.zeros(len(tasks), dtype=bool)
mask[maskind]=True 

states=[states[i] for i in range(len(mask)) if mask[i]]
actions=[actions[i] for i in range(len(mask)) if mask[i]]
tasks=tasks[mask]

env=ffacc_real.FireFlyPaper2(arg)
env.debug=1
phi=torch.tensor([[0.4],
            [pi/2],
            [0.001],
            [0.001],
            [0.001],
            [0.001],
            [0.13],
            [0.001],
            [0.001],
            [0.001],
            [0.001],
    ])
agent_=TD3.load('trained_agent/paper.zip')
agent=agent_.actor.mu.cpu()


invfile=Path('/data/neuraltest/inv_schroall_constrain_nopert_part2')
finaltheta, finalcov, err=process_inv(invfile,removegr=False, usingbest=False)


print('done process data')


# plot code -------------------
# run the agent
ind=np.random.randint(low=0, high=len(tasks))

i=len(sortind)+1
i-=1
ind=sortind[i]

thetask=tasks[ind]
ntrial=1
theta=finaltheta
ep_states, _, beliefs, covs=run_trials(agent=agent, env=env, phi=phi, theta=theta,task=thetask,ntrials=ntrial,pert=None,given_obs=None,return_belief=True, given_action=actions[ind], given_state=states[ind])

# plot
fontsize=11
alpha=1
every=5 # plot every x belief ellipse
with initiate_plot(3,3, 300) as fig:
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
    # ax.set_xlim([-235, 235]); ax.set_ylim([-2, 430])
    xrange=np.cos(pi/180*(90-42.5))*300
    x_temp = np.linspace(-xrange, xrange)
    ax.plot(x_temp, np.sqrt(300**2 - x_temp**2), c='k', ls=':')
    ax.text(-10, 310, s=r'$85\degree$', fontsize=fontsize)

    ax.plot(np.linspace(-300, -200), np.linspace(-100, -100), c='k') # 100 is cm
    ax.plot(np.linspace(-300, -300), np.linspace(-100, 0), c='k')
    ax.text(-280, -160, s=r'$1 m$', fontsize=fontsize)
    quickallspine(ax)
    ax.axis('equal')

    # path
    for s in ep_states:
        ax.plot(s[:,1]*200,s[:,0]*200, c='grey',alpha=alpha, linewidth=0.5)
    # ax.scatter(subtasks[:5][:,1]*200,subtasks[:5][:,0]*200)

    s = states[ind]
    ax.plot(s[:,1]*200,s[:,0]*200, c='blue',alpha=alpha, linewidth=0.5, label='actual path')
    # ax.scatter(subtasks[:5][:,1]*200,subtasks[:5][:,0]*200)

    # # belief path
    # for s in beliefs:
    #     b=s[:,:,0]
    #     ax.plot(b[:,1]*200,b[:,0]*200, c='grey',alpha=alpha, linewidth=0.5)

    # belief path
    for mus, cs in (zip(beliefs,covs)):
        for t, (mu, cov) in enumerate(zip(mus, cs)):
            if t%every==0 or t==len(mus)-1:
                mu_=torch.tensor([[0,1],[1.,0]])@mu[:2]*200
                cov_=torch.tensor([[0,200],[200.,0]])@cov[:2,:2]@torch.tensor([[0,200],[200.,0]])
                plot_cov_ellipse(cov_, mu_, alpha=1,nstd=2,ax=ax, edgecolor=color_settings['b'], label='belief')

    # start
    ax.scatter(0,0, marker='*', color='black', s=55) 

    # goal
    plot_circle(np.eye(2)*65,[thetask[1]*200,thetask[0]*200],ax=ax,edgecolor='k')

    quickleg(ax)
    # quicksave('example schro inferred belief path 3')







