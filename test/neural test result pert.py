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
from plot_ult import *

import configparser
config = configparser.ConfigParser()
config.read_file(open('privateconfig'))
datafolder=config['Datafolder']['data']

print('loading data')
datapath=Path(datafolder)/"victor_pert/packed"
with open(datapath,'rb') as f:
    df_ = pickle.load(f)

df=df_[(~np.isnan(df_.perturb_start_time)) & (df_.perturb_start_time+1<df_.trial_dur)]

states, actions, tasks=monkey_data_downsampled(df,factor=0.0025)
tasks=np.array(tasks)

perts, pertmeta=df_downsamplepert(df,factor=0.0025)



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
perts= [perts[i] for i in range(len(mask)) if mask[i]]

err=np.array([torch.norm(resp[-1,:2]-tar) for resp, tar in zip(states, tasks)])
sortind=np.argsort(err*-1)



env=ffacc_real.FireFlyPaper(arg)
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


invfile=Path(datapath/'victor_pert/newinvvictor_pert_ds')
finaltheta_, finalcov_, err=process_inv(invfile,removegr=False, usingbest=False)
finaltheta=finaltheta_[:-1] # remove the time cost
finalcov=finalcov_[:-1,:-1]

print('done process data')


# plot code -------------------
# run the agent
ind=np.random.randint(low=0, high=len(tasks))

i=len(sortind)
i-=1
ind=sortind[i]
print(ind)
thetask=tasks[ind]
ntrial=1
theta=finaltheta
ep_states, _, beliefs, covs=run_trials(agent=agent, env=env, phi=phi, theta=theta,task=thetask,ntrials=ntrial,pert=None,given_obs=None,return_belief=True, given_action=actions[ind], given_state=states[ind])

ep_states_unpert,_,_,_=run_trials(agent=agent, env=env, phi=phi, theta=theta,task=thetask,ntrials=ntrial,pert=None,given_obs=None,return_belief=True, given_action=actions[ind], given_state=None)

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
    # goal
    plot_circle(np.eye(2)*65,[thetask[1]*200,thetask[0]*200],ax=ax,edgecolor='k')

    # actual data path (path after pert)
    s = states[ind]
    ax.plot(s[:,1]*200,s[:,0]*200, c='blue',alpha=alpha, linewidth=0.5, label='actual path with perturbation')

    # path before pert
    s = ep_states_unpert[0]
    ax.plot(s[:,1]*200,s[:,0]*200, c='grey',alpha=alpha, linewidth=0.5, label='if only relied on prediction')

   
    pert=perts[ind]
    pertmask=np.argwhere(abs(pert[:,0])>0.01)
    for point,delta in zip(states[ind][pertmask,:], pert[pertmask]):
        plt.plot([point[0,1]*200, point[0,1]*200 + delta[0,1]*200], [point[0,0]*200, point[0,0]*200+delta[0,0]*200], color='red', label='velocity perturbation', linewidth=0.2)

    # connection pre and post pert
    # plt.quiver(
    #     states[ind][pertmask,1]*200, 
    #     states[ind][pertmask,0]*200, 
    #     np.sin(pert[pertmask,1]+np.array(states[ind][pertmask,2]))*pert[pertmask,0]*200,
    #     np.cos(pert[pertmask,1]+np.array(states[ind][pertmask,2]))*pert[pertmask,0]*200,
    #     scale_units='xy', angles='xy', scale=1)
    
    # # belief path
    # for s in beliefs:
    #     b=s[:,:,0]
    #     ax.plot(b[:,1]*200,b[:,0]*200, c='orange',alpha=alpha, linewidth=0.5)

    # belief path
    beliefs=[beliefs[0]-beliefs[0][0]]
    for mus, cs in (zip(beliefs,covs)):
        for t, (mu, cov) in enumerate(zip(mus, cs)):
            if t%every==0 or t==len(mus)-1:
                mu_=torch.tensor([[0,1],[1.,0]])@mu[:2]*200
                cov_=torch.tensor([[0,200],[200.,0]])@cov[:2,:2]@torch.tensor([[0,200],[200.,0]])
                plot_cov_ellipse(cov_, mu_, alpha=1,nstd=2,ax=ax, edgecolor=color_settings['b'], label='belief')

    # start
    ax.scatter(0,0, marker='*', color='black', s=55) 


    plt.plot([-0,-0],[-0,-0],color=color_settings['b'], label='belief')


quickleg(ax, bbox_to_anchor=[0,0])
quicksave('example victor pert belief {}'.format(ind), fig=ax.get_figure())







