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
from matplotlib import cm
import scipy.stats as stats

env=ffacc_real.FireFlyPaper(arg)
env.debug=True
env.terminal_vel=0.05
agent_=TD3.load('trained_agent/paper.zip')
agent=agent_.actor.mu.cpu()

# load data ----------------------------
asd_data_set={}
numhsub,numasub=25,14
fulltrainfolder='persub1cont'
parttrainfolder='persub3of5dp'
for invtag in ['h','a']:
    for isub in range(numhsub):
        thesub="{}sub{}".format(invtag,str(isub))
        evalname=Path("/data/human/{}/evaltrain_inv{}sub{}".format(parttrainfolder,invtag,str(isub)))
        fullinverseres=Path("/data/human/{}".format(fulltrainfolder))/"inv{}sub{}".format(invtag,str(isub))
        partinverseres=Path("/data/human/{}".format(parttrainfolder))/"inv{}sub{}".format(invtag,str(isub))
        # load inv res
        if fullinverseres.is_file():
            asd_data_set['res'+thesub]=process_inv(fullinverseres, usingbest=True, removegr=False)
        # load data
        if Path('/data/human/{}'.format(thesub)).is_file():
            with open('/data/human/{}'.format(thesub), 'rb') as f:
                states, actions, tasks = pickle.load(f)
            print(len(states))
            asd_data_set['data'+thesub]=states, actions, tasks
        

# frequency of action changes ------------------------------
with open('/data/human/woagroup','rb') as f:
    _,aactions,_=pickle.load(f)
with open('/data/human/wohgroup','rb') as f:
    _,hactions,_=pickle.load(f)

atmp=[ np.diff( np.linalg.norm(np.array(a), axis=1) ) for a in aactions]
achanges=[]
for trial in atmp:
    for d in trial:
        # if d>0:
            achanges.append(d)

htmp=[ np.diff( np.linalg.norm(np.array(a), axis=1) ) for a in hactions]
hchanges=[]
for trial in htmp:
    for d in trial:
        # if d>0:
            hchanges.append(d)

with initiate_plot(3,2, 300) as f:
    ax=f.add_subplot(111)
    ax.hist(achanges, density=True, bins=999,alpha=0.5, color='r', label='ASD')
    ax.hist(hchanges, density=True, bins=999, alpha=0.5, color='b', label='NT')
    ax.set_title('ctrl change')
    ax.set_xlabel('ctrl changes per dt')
    ax.set_ylabel('probability')
    plt.xlim(-0.1,0.2)
    quickspine(ax)
    quickleg(ax)
    # quicksave('dev action cost compared')

# frequency of action changes, angular only ------------------------------
with open('/data/human/woagroup','rb') as f:
    _,aactions,_=pickle.load(f)
with open('/data/human/wohgroup','rb') as f:
    _,hactions,_=pickle.load(f)

atmp=[ (np.power(relu(np.diff(np.array(a)[:,1], axis=0)),2)) for a in aactions]
achanges=[]
for trial in atmp:
    for d in trial:
        if d>1e-4:
            achanges.append(d)

htmp=[ (np.power(relu(np.diff(np.array(a)[:,1], axis=0)),2)) for a in hactions]
hchanges=[]
for trial in htmp:
    for d in trial:
        if d>1e-4:
            hchanges.append(d)

achanges,hchanges = np.array(achanges),np.array(hchanges)
with initiate_plot(3,2, 300) as f:
    xrange=[0,0.005]
    ax=f.add_subplot(111)
    ax.hist(achanges[achanges<xrange[1]], density=True, bins=99,alpha=0.5, color='r', label='ASD')
    ax.hist(hchanges[hchanges<xrange[1]], density=True, bins=99, alpha=0.5, color='b', label='NT')
    ax.set_title('ctrl change')
    ax.set_xlabel('ctrl changes per dt')
    ax.set_ylabel('probability')
    plt.xlim(xrange)
    quickspine(ax)
    quickleg(ax)
    # quicksave('dev action cost compared')

# focuos on larger changes -------------------------------
thresholds=np.linspace(0.1,0.5,33)
acounts=[]
for th in thresholds:
    acounts.append(sum([ len(np.where( (np.power(relu(np.diff(np.linalg.norm(np.array(a), axis=1), axis=0)),2)) >th)[0]) for a in aactions]))
hcounts=[]
for th in thresholds:
    hcounts.append(sum([ len(np.where( (np.power(relu(np.diff(np.linalg.norm(np.array(a), axis=1), axis=0)),2)) >th)[0]) for a in hactions]))


with initiate_plot(3,2, 300) as f:
    ax=f.add_subplot(111)
    ax.plot(thresholds,np.array(acounts)/sum([len(a) for a in aactions]),color='r', label='ASD')
    ax.plot(thresholds,np.array(hcounts)/sum([len(a) for a in hactions]), color='b', label='NT')
    ax.set_title('ctrl change')
    ax.set_xlabel('ctrl changes amplitude')
    ax.set_ylabel('probability')
    quickspine(ax)
    quickleg(ax)
    # quicksave('dev action cost compared')

# cumulative probability plot
with initiate_plot(3,2, 300) as f:
    ax=f.add_subplot(111)
    ax.plot(thresholds,np.cumsum(np.array(acounts)/sum([len(a) for a in aactions])),color='r', label='ASD')
    ax.plot(thresholds,np.cumsum(np.array(hcounts)/sum([len(a) for a in hactions])), color='b', label='NT')
    ax.set_title('ctrl change > 0.1')
    ax.set_xlabel('ctrl changes amplitude')
    ax.set_ylabel('cumulative probability')
    quickspine(ax)
    quickleg(ax)
    # quicksave('cumulative prob for control change > 0.1')


# focuos on larger changes, angular only ----------------
thresholds=np.linspace(0.01,0.5,33)
acounts=[]
for th in thresholds:
    acounts.append(sum([ len(np.where( (np.power(relu(np.diff(np.array(a)[:,1], axis=0)),2)) >th)[0]) for a in aactions]))
hcounts=[]
for th in thresholds:
    hcounts.append(sum([ len(np.where( (np.power(relu(np.diff(np.array(a)[:,1], axis=0)),2)) >th)[0]) for a in hactions]))

with initiate_plot(3,2, 300) as f:
    ax=f.add_subplot(111)
    ax.plot(thresholds,np.array(acounts)/sum([len(a) for a in aactions]),color='r', label='ASD')
    ax.plot(thresholds,np.array(hcounts)/sum([len(a) for a in hactions]), color='b', label='NT')
    ax.set_title('ctrl change')
    ax.set_xlabel('ctrl changes amplitude')
    ax.set_ylabel('probability')
    quickspine(ax)
    quickleg(ax)
    # quicksave('dev action cost compared')
max(np.array(acounts)/sum([len(a) for a in aactions])-np.array(hcounts)/sum([len(a) for a in hactions]))



# focuos on larger changes at begining-------------------------------
with initiate_plot(3,2, 300) as f:
    ax=f.add_subplot(111)
    for t in range(1,20,3):
        # thresholds=np.linspace(0.001,0.5,33)
        xs=np.linspace(np.log(0.01),np.log(1.5),33)
        thresholds=np.exp(xs)
        acounts=[]
        for th in thresholds:
            acounts.append(sum([ len(np.where( (np.power(relu(np.diff(np.linalg.norm(np.array(a)[:t], axis=1), axis=0)),2)) >th)[0]) for a in aactions]))
        hcounts=[]
        for th in thresholds:
            hcounts.append(sum([ len(np.where( (np.power(relu(np.diff(np.linalg.norm(np.array(a)[:t], axis=1), axis=0)),2)) >th)[0]) for a in hactions]))

        ax.plot(thresholds,np.array(acounts)/sum([len(a) for a in aactions]),color='r', label='ASD')
        ax.plot(thresholds,np.array(hcounts)/sum([len(a) for a in hactions]), color='b', label='NT')
    ax.set_title('ctrl change')
    ax.set_xlabel('ctrl changes amplitude')
    ax.set_ylabel('probability')
    quickspine(ax)
    quickleg(ax)
    # quicksave('dev action cost compared')

# magnitude of actions ------------------------------
atmp=[ ( np.linalg.norm(np.array(a), axis=1) ) for a in aactions]
achanges=[]
for trial in atmp:
    for d in trial:
        if d>0:
            achanges.append(d)

htmp=[ ( np.linalg.norm(np.array(a), axis=1) ) for a in hactions]
hchanges=[]
for trial in htmp:
    for d in trial:
        if d>0:
            hchanges.append(d)
with initiate_plot(3,2, 300) as f:
    ax=f.add_subplot(111)
    ax.hist(achanges, density=True, bins=99,alpha=0.5, color='r', label='ASD')
    ax.hist(hchanges, density=True, bins=99, alpha=0.5, color='b', label='NT')
    ax.set_title('ctrl magnitude')
    ax.set_xlabel('ctrl magnitude')
    ax.set_ylabel('probability')
    quickspine(ax)
    quickleg(ax)
    # quicksave('mag action cost compared')


# magnitude of actions, angular ------------------------------
atmp=[ (np.array(a)[:,1] ) for a in aactions]
achanges=[]
for trial in atmp:
    for d in trial:
        if d>0:
            achanges.append(d)

htmp=[ ( np.array(a)[:,1]) for a in hactions]
hchanges=[]
for trial in htmp:
    for d in trial:
        if d>0:
            hchanges.append(d)
with initiate_plot(3,2, 300) as f:
    ax=f.add_subplot(111)
    ax.hist(achanges, density=True, bins=99,alpha=0.5, color='r', label='ASD')
    ax.hist(hchanges, density=True, bins=99, alpha=0.5, color='b', label='NT')
    ax.set_title('ctrl magnitude')
    ax.set_xlabel('ctrl magnitude')
    ax.set_ylabel('probability')
    quickspine(ax)
    quickleg(ax)
    # quicksave('mag action cost compared')

# magnitude of actions, forward ------------------------------
atmp=[ (np.array(a)[:,0] ) for a in aactions]
achanges=[]
for trial in atmp:
    for d in trial:
        if d>0:
            achanges.append(d)

htmp=[ ( np.array(a)[:,0]) for a in hactions]
hchanges=[]
for trial in htmp:
    for d in trial:
        if d>0:
            hchanges.append(d)
with initiate_plot(3,2, 300) as f:
    ax=f.add_subplot(111)
    ax.hist(achanges, density=True, bins=99,alpha=0.5, color='r', label='ASD')
    ax.hist(hchanges, density=True, bins=99, alpha=0.5, color='b', label='NT')
    ax.set_title('ctrl magnitude')
    ax.set_xlabel('ctrl magnitude')
    ax.set_ylabel('probability')
    quickspine(ax)
    quickleg(ax)
    # quicksave('mag action cost compared')




# indivividual endpoint stats to svm ----------------------------
asd_data_endpoint={}
asd_data_endpoint_polar={}
for invtag in ['h','a']:
    for isub in range(numhsub):
        thesub="{}sub{}".format(invtag,str(isub))
        if 'data'+thesub in asd_data_set:
            states,_,tasks=asd_data_set['data'+thesub]
            endpoint=np.array([s[-1,:2].tolist() for s in states])
            r,a=xy2pol(endpoint.T, rotation=False)
            endpointpolar=np.stack([a,r]).T
            r,a=xy2pol(tasks.T, rotation=False)
            taskspolar=np.stack([a,r]).T
            asd_data_endpoint[thesub]=(endpoint, tasks)
            asd_data_endpoint_polar[thesub]=(endpointpolar, taskspolar)

# xy coord
for thesub in asd_data_endpoint.keys():
    f=plt.figure()
    f.add_subplot(111)
    (endpoint, tasks)=asd_data_endpoint[thesub]
    plt.scatter(endpoint[:,0],endpoint[:,1])
    plt.scatter(tasks[:,0],tasks[:,1])
    plt.axis('equal')
    plt.show()

# polar
for thesub in asd_data_endpoint_polar.keys():
    f=plt.figure()
    f.add_subplot(111, polar=True)
    (endpoint, tasks)=asd_data_endpoint_polar[thesub]
    plt.scatter(endpoint[:,0],endpoint[:,1])
    plt.scatter(tasks[:,0],tasks[:,1])
    plt.show()


# as a group, linear regression of the angular err ---------------------------------------
xa, ya, xh, yh =[],[],[],[]
for thesub in asd_data_endpoint_polar.keys():
    (endpoint, tasks)=asd_data_endpoint_polar[thesub]
    if thesub[0]=='a': 
        xa.append(tasks[:,0])
        ya.append(endpoint[:,0])
    else: 
        xh.append(tasks[:,0])
        yh.append(endpoint[:,0])

xa, ya, xh, yh =np.hstack(xa),np.hstack(ya),np.hstack(xh),np.hstack(yh)
stats.linregress(xa,ya)
stats.linregress(xh, yh)



# as a group, linear regression of the radial err ---------------------------------------
xa, ya, xh, yh =[],[],[],[]
for thesub in asd_data_endpoint_polar.keys():
    (endpoint, tasks)=asd_data_endpoint_polar[thesub]
    if thesub[0]=='a': 
        xa.append(tasks[:,1])
        ya.append(endpoint[:,1])
    else: 
        xh.append(tasks[:,1])
        yh.append(endpoint[:,1])

xa, ya, xh, yh =np.hstack(xa),np.hstack(ya),np.hstack(xh),np.hstack(yh)
stats.linregress(xa,ya)
stats.linregress(xh, yh)



# per sub, linear regression of the angular err ---------------------------------------
xa, ya, xh, yh =[],[],[],[]
ares, hres=[],[]
for thesub in asd_data_endpoint_polar.keys():
    (endpoint, tasks)=asd_data_endpoint_polar[thesub]
    if thesub[0]=='a': 
        ares.append(stats.linregress(tasks[:,0],endpoint[:,0]))
    else: 
        hres.append(stats.linregress(tasks[:,0],endpoint[:,0]))

npsummary([a.slope for a in ares])
npsummary([a.slope for a in hres])

stats.ttest_ind([a.slope for a in ares],[a.slope for a in hres])

npsummary([a.rvalue**2 for a in ares])
npsummary([a.rvalue**2 for a in hres])

npsummary([a.pvalue for a in ares])
npsummary([a.pvalue for a in hres])


# per sub, linear regression of the radial err ---------------------------------------
xa, ya, xh, yh =[],[],[],[]
ares, hres=[],[]
for thesub in asd_data_endpoint_polar.keys():
    (endpoint, tasks)=asd_data_endpoint_polar[thesub]
    if thesub[0]=='a': 
        ares.append(stats.linregress(tasks[:,1],endpoint[:,1]))
    else: 
        hres.append(stats.linregress(tasks[:,1],endpoint[:,1]))

npsummary([a.slope for a in ares])
npsummary([a.slope for a in hres])

stats.ttest_ind([a.slope for a in ares],[a.slope for a in hres])


# per sub, error of first half vs later half ---------------------------------------
xa, ya, xh, yh =[],[],[],[]
ares, hres=[],[]
ares2, hres2=[],[]
for thesub in asd_data_endpoint_polar.keys():
    (endpoint, tasks)=asd_data_endpoint_polar[thesub]
    
    if thesub[0]=='a': 
        ares.append(stats.linregress(tasks[:75,0],endpoint[:75,0]))
        ares2.append(stats.linregress(tasks[75:,0],endpoint[75:,0]))
    else: 
        hres.append(stats.linregress(tasks[:75,0],endpoint[:75,0]))
        hres2.append(stats.linregress(tasks[75:,0],endpoint[75:,0]))

stats.ttest_ind([a.slope for a in ares],[a.slope for a in hres]) # angular

xa, ya, xh, yh =[],[],[],[]
ares, hres=[],[]
ares2, hres2=[],[]
for thesub in asd_data_endpoint_polar.keys():
    (endpoint, tasks)=asd_data_endpoint_polar[thesub]
    
    if thesub[0]=='a': 
        ares.append(stats.linregress(tasks[:75,1],endpoint[:75,1]))
        ares2.append(stats.linregress(tasks[75:,1],endpoint[75:,1]))
    else: 
        hres.append(stats.linregress(tasks[:75,1],endpoint[:75,1]))
        hres2.append(stats.linregress(tasks[75:,1],endpoint[75:,1]))

stats.ttest_ind([a.slope for a in ares],[a.slope for a in hres]) # radial




# angular err
with initiate_plot(3,3,300) as f:
    ax=f.add_subplot(111)
    for thesub in asd_data_endpoint_polar.keys():
        if thesub[0]=='a': color='r'
        else: color='b'
        (endpoint, tasks)=asd_data_endpoint_polar[thesub]
        ax.scatter(tasks[:,0],endpoint[:,0],color=color, s=0.5)
    ax.plot([-0.7,.7],[-0.7,.7],'k')
    ax.set_xlim(-0.7,0.7)
    ax.set_ylim(-1,1)
    quickspine(ax)
    ax.set_xlabel('target angle')
    ax.set_ylabel('response angle')
    # quicksave('asd angular err')

with initiate_plot(6,3,300) as f:
    ax1=f.add_subplot(121)
    ax2=f.add_subplot(122, sharex=ax1, sharey=ax1)
    for thesub in asd_data_endpoint_polar.keys():
        if thesub[0]=='a': 
            color='r'
            (endpoint, tasks)=asd_data_endpoint_polar[thesub]
            ax1.scatter(tasks[:,0],endpoint[:,0],color=color, s=0.5)
        else: 
            color='b'
            (endpoint, tasks)=asd_data_endpoint_polar[thesub]
            ax2.scatter(tasks[:,0],endpoint[:,0],color=color, s=0.5)
    ax1.set_xlim(-0.7,0.7)
    ax1.set_ylim(-1,1)
    quickspine(ax1)
    quickspine(ax2)
    ax1.set_xlabel('target angle')
    ax2.set_xlabel('target angle')
    ax1.set_ylabel('response angle')
    ax1.plot([-0.7,.7],[-0.7,.7],'k')
    ax2.plot([-0.7,.7],[-0.7,.7],'w')
    # quicksave('asd angular err sep')

# radial err
with initiate_plot(3,3,300) as f:
    ax=f.add_subplot(111)
    for thesub in asd_data_endpoint_polar.keys():
        if thesub[0]=='a': color='r'
        else: color='b'
        (endpoint, tasks)=asd_data_endpoint_polar[thesub]
        ax.scatter(tasks[:,1],endpoint[:,1],color=color, s=0.5)
    ax.set_xlim(0.5,None)
    ax.set_ylim(0,None)
    quickspine(ax)
    ax.set_xlabel('target distance')
    ax.set_ylabel('response distance')
    ax.plot([-0.7,3],[-0.7,3],'k')
    # quicksave('asd radiual err')


with initiate_plot(6,3,300) as f:
    ax1=f.add_subplot(121)
    ax2=f.add_subplot(122, sharex=ax1, sharey=ax1)
    for thesub in asd_data_endpoint_polar.keys():
        if thesub[0]=='a': 
            color='r'
            (endpoint, tasks)=asd_data_endpoint_polar[thesub]
            ax1.scatter(tasks[:,1],endpoint[:,1],color=color, s=0.5)
        else: 
            color='b'
            (endpoint, tasks)=asd_data_endpoint_polar[thesub]
            ax2.scatter(tasks[:,1],endpoint[:,1],color=color, s=0.5)
    ax1.set_xlim(0.5,None)
    ax1.set_ylim(0,None)
    quickspine(ax1)
    quickspine(ax2)
    ax1.set_xlabel('target distance')
    ax2.set_xlabel('target distance')
    ax1.set_ylabel('response distance')
    ax1.plot([-0.7,3],[-0.7,3],'k')
    ax2.plot([-0.7,3],[-0.7,3],'w')
    # quicksave('asd radiual err sep')


# error of first half vs later half (no obvious learning happends) ------------------------
with initiate_plot(5,5,300) as f:
    ax1=f.add_subplot(221)
    ax2=f.add_subplot(222, sharex=ax1, sharey=ax1)
    for thesub in asd_data_endpoint_polar.keys():
        ntrial=len(asd_data_endpoint_polar[thesub][0])//2
        if thesub[0]=='a': 
            color='r'
            (endpoint, tasks)=asd_data_endpoint_polar[thesub]
            ax1.scatter(tasks[:ntrial,1],endpoint[:ntrial,1],color=color, s=0.5)
        else: 
            color='b'
            (endpoint, tasks)=asd_data_endpoint_polar[thesub]
            ax2.scatter(tasks[:ntrial,1],endpoint[:ntrial,1],color=color, s=0.5)
    ax1.set_xlim(0.5,None)
    ax1.set_ylim(0,None)
    quickspine(ax1)
    quickspine(ax2)
    ax1.set_xlabel('target distance')
    ax2.set_xlabel('target distance')
    ax1.set_ylabel('response distance')
    ax1.plot([-0.7,3],[-0.7,3],'k')
    ax2.plot([-0.7,3],[-0.7,3],'w')

    ax1=f.add_subplot(223)
    ax2=f.add_subplot(224, sharex=ax1, sharey=ax1)
    for thesub in asd_data_endpoint_polar.keys():
        ntrial=len(asd_data_endpoint_polar[thesub][0])//2
        if thesub[0]=='a': 
            color='r'
            (endpoint, tasks)=asd_data_endpoint_polar[thesub]
            ax1.scatter(tasks[ntrial:,1],endpoint[ntrial:,1],color=color, s=0.5)
        else: 
            color='b'
            (endpoint, tasks)=asd_data_endpoint_polar[thesub]
            ax2.scatter(tasks[ntrial:,1],endpoint[ntrial:,1],color=color, s=0.5)
    ax1.set_xlim(0.5,None)
    ax1.set_ylim(0,None)
    quickspine(ax1)
    quickspine(ax2)
    ax1.set_xlabel('target distance')
    ax2.set_xlabel('target distance')
    ax1.set_ylabel('response distance')
    ax1.plot([-0.7,3],[-0.7,3],'k')
    ax2.plot([-0.7,3],[-0.7,3],'w')
    quicksave('first vs last half radial err')

with initiate_plot(5,5,300) as f:
    ax1=f.add_subplot(221)
    ax2=f.add_subplot(222, sharex=ax1, sharey=ax1)
    for thesub in asd_data_endpoint_polar.keys():
        ntrial=len(asd_data_endpoint_polar[thesub][0])//2
        if thesub[0]=='a': 
            color='r'
            (endpoint, tasks)=asd_data_endpoint_polar[thesub]
            ax1.scatter(tasks[:ntrial,0],endpoint[:ntrial,0],color=color, s=0.5)
        else: 
            color='b'
            (endpoint, tasks)=asd_data_endpoint_polar[thesub]
            ax2.scatter(tasks[:ntrial,0],endpoint[:ntrial,0],color=color, s=0.5)
    ax1.set_xlim(-0.7,0.7)
    ax1.set_ylim(-1,1)
    quickspine(ax1)
    quickspine(ax2)
    ax1.set_xlabel('target angle')
    ax2.set_xlabel('target angle')
    ax1.set_ylabel('response angle')
    ax1.plot([-0.7,.7],[-0.7,.7],'k')
    ax2.plot([-0.7,.7],[-0.7,.7],'w')

    ax1=f.add_subplot(223)
    ax2=f.add_subplot(224, sharex=ax1, sharey=ax1)
    for thesub in asd_data_endpoint_polar.keys():
        ntrial=len(asd_data_endpoint_polar[thesub][0])//2
        if thesub[0]=='a': 
            color='r'
            (endpoint, tasks)=asd_data_endpoint_polar[thesub]
            ax1.scatter(tasks[ntrial:,0],endpoint[ntrial:,0],color=color, s=0.5)
        else: 
            color='b'
            (endpoint, tasks)=asd_data_endpoint_polar[thesub]
            ax2.scatter(tasks[ntrial:,0],endpoint[ntrial:,0],color=color, s=0.5)
    ax1.set_xlim(-0.7,0.7)
    ax1.set_ylim(-1,1)
    quickspine(ax1)
    quickspine(ax2)
    ax1.set_xlabel('target angle')
    ax2.set_xlabel('target angle')
    ax1.set_ylabel('response angle')
    ax1.plot([-0.7,.7],[-0.7,.7],'k')
    ax2.plot([-0.7,.7],[-0.7,.7],'w')
    quicksave('first vs last half angular err')

# radial and angular err
polar_err={}
for thesub in asd_data_endpoint.keys():
    (endpoint, tasks)=asd_data_endpoint_polar[thesub]
    err=(endpoint - tasks) # a, r
    polar_err[thesub]=err

# subjects accuracy
accs=[ (len(err[:,0][err[:,0]<0.13])/len(err[:,0])) for _,err in polar_err.items()]
npsummary(accs[:numhsub])
npsummary(accs[numhsub:])

# ttest of angular err---------------------------------------
xa, ya, xh, yh =[],[],[],[]
ares, hres=[],[]
ares2, hres2=[],[]
for thesub in asd_data_endpoint_polar.keys():
    (endpoint, tasks)=asd_data_endpoint_polar[thesub]
    session=(endpoint[:,0]/(tasks[:,0])).tolist()    
    if thesub[0]=='a': 
        ares+=session[:len(session)//2]
        ares2+=session[len(session)//2:]
    else: 
        hres+=session[:len(session)//2]
        hres2+=session[len(session)//2:]

stats.ttest_ind(ares,ares2)
stats.ttest_ind(hres,hres2)



# t test of the radial err ---------------------------------------
xa, ya, xh, yh =[],[],[],[]
ares, hres=[],[]
ares2, hres2=[],[]
for thesub in asd_data_endpoint_polar.keys():
    (endpoint, tasks)=asd_data_endpoint_polar[thesub]
    session=(endpoint[:,1]/(tasks[:,1])).tolist()    
    if thesub[0]=='a': 
        ares+=session[:len(session)//2]
        ares2+=session[len(session)//2:]
    else: 
        hres+=session[:len(session)//2]
        hres2+=session[len(session)//2:]

stats.ttest_ind(ares,ares2)
stats.ttest_ind(hres,hres2)


# svm
alltag=[]
allsamples=[]
for invtag in ['h','a']:
    for isub in range(numhsub):
        thesub="{}sub{}".format(invtag,str(isub))
        if thesub in asd_data_endpoint_polar:
            (endpoint, tasks)=asd_data_endpoint_polar[thesub]
            err=polar_err[thesub]
            allsamples.append(np.hstack([endpoint, err]))
            if invtag=='a':
                alltag+=[1]*len(endpoint)
            else:
                alltag+=[0]*len(endpoint)
allsamples=np.abs(np.vstack(allsamples))
alltag=np.array(alltag).astype('int')

X, Y=allsamples,alltag
X = X[np.logical_or(Y==0,Y==1)][:,:8]
Y = Y[np.logical_or(Y==0,Y==1)]
model = svm.SVC(kernel='linear')
clf = model.fit(X, Y)
def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.xlabel('parameter coef')
    ax=plt.gca()
    quickspine(ax)
f_importances(np.abs(clf.coef_[0]),['rad resp', 'ang resp', 'rad err', 'ang err'])

print('''
project the individual thetas on to the normal vector.
''')
w=clf.coef_[0]
ticks=X[:,:8].dot(w)
fig = plt.figure()
ax  = fig.add_subplot(111)
ax.hist(ticks[Y==0],density=True,color='b',bins=55,label='health control',alpha=0.6)
ax.hist(ticks[Y==1],density=True,color='r',bins=55,label='ASD',alpha=0.6)
ax.set_xlabel('param value')
ax.set_ylabel('probability')
quickspine(ax)
# quicksave('svm on end response')

# t test cannot seperate
stats.ttest_ind(ticks[Y==0],ticks[Y==1])
stats.ks_2samp(ticks[Y==0],ticks[Y==1])


# # hit rates and false alarm rates
# hitP = 23/30
# faP  =  4/30
# # z-scores
# hitZ = stats.norm.ppf(hitP)
# faZ  = stats.norm.ppf(faP)
# # d-prime
# dPrime = hitZ-faZ
# print(dPrime)


print('''
conclusion, with svm and subjects end response + err, we cannot tell them apart
''')



# individusla behavior trajectory to svm ----------------------------

def pad_to_dense(M, maxlen=0):
    """Appends the minimal required amount of zeroes at the end of each 
     array in the jagged array `M`, such that `M` looses its jagedness."""
    if maxlen==0:
        maxlen = max(len(r) for r in M)

    Z = np.zeros((len(M), maxlen))
    for enu, row in enumerate(M):
        Z[enu, :len(row)] += row 
    return Z

alltag=[]
allsamples=[]
maxlen=0
cumsum=[0]
running=0
for invtag in ['h','a']:
    astartind=running
    for isub in range(numhsub):
        thesub="{}sub{}".format(invtag,str(isub))
        if 'data'+thesub in asd_data_set:
            _,actions,tasks=asd_data_set['data'+thesub]
            curmax=max([len(a) for a in actions])
            maxlen=max(maxlen, curmax)
            running+=len(actions)
            cumsum.append(running)
for invtag in ['h','a']:
    for isub in range(numhsub):
        thesub="{}sub{}".format(invtag,str(isub))
        if 'data'+thesub in asd_data_set:
            _,actions,tasks=asd_data_set['data'+thesub]
            paddedv=pad_to_dense([np.array(a[:,0]) for a in actions],maxlen=maxlen)
            paddedw=pad_to_dense([np.array(a[:,1]) for a in actions],maxlen=maxlen)
            allsamples.append(np.hstack([tasks,paddedv,paddedw]))
            if invtag=='a':
                alltag+=[1]*len(actions)
            else:
                alltag+=[0]*len(actions)


allsamples=np.abs(np.vstack(allsamples))
alltag=np.array(alltag).astype('int')

X, Y=allsamples,alltag
X = X[np.logical_or(Y==0,Y==1)][:,:]
Y = Y[np.logical_or(Y==0,Y==1)]
model = svm.SVC(kernel='linear')
clf = model.fit(X, Y)
f_importances(np.abs(clf.coef_[0][2:]),list(range(9)))
# quicksave('svm trajectory weights')

with initiate_plot(5,2,300) as f:
    vwcoef=np.abs(clf.coef_[0][2:])
    maxcoef=max(vwcoef)
    normvwcoef=vwcoef/maxcoef
    ax=f.add_subplot(121)
    ax.plot((normvwcoef[:maxlen]))
    quickspine(ax)
    ax.set_xlabel('time dt')
    ax.set_ylabel('v control coef')
    ax.set_ylim(0,1)
    ax.set_xlim(0,)
    ax=f.add_subplot(122)
    ax.plot((normvwcoef[maxlen:]))
    quickspine(ax)
    ax.set_xlabel('time dt')
    ax.set_ylabel('w control coef')
    ax.set_ylim(0,1)
    ax.set_xlim(0,)
    # quicksave('v and w coef of trajectory svm')

with initiate_plot(5,2,300) as f:
    smoothconvwindow=22
    vwcoef=np.abs(clf.coef_[0][2:])
    vcoef=smooth(vwcoef[:maxlen], smoothconvwindow)
    wcoef=smooth(vwcoef[maxlen:], smoothconvwindow)
    maxcoef=max(max(vcoef), max(wcoef))
    vcoef, wcoef=vcoef/maxcoef, wcoef/maxcoef
    ax=f.add_subplot(121)
    ax.plot(vcoef)
    quickspine(ax)
    ax.set_xlabel('time dt')
    ax.set_ylabel('v control coef')
    ax.set_ylim(0,1)
    ax.set_xlim(0,)
    ax=f.add_subplot(122)
    ax.plot(wcoef)
    quickspine(ax)
    ax.set_xlabel('time dt')
    ax.set_ylabel('w control coef')
    ax.set_ylim(0,1)
    ax.set_xlim(0,)
    # quicksave('v and w coef of trajectory svm smooth kernalsize{}'.format(smoothconvwindow))


print('''
project the individual trajectory on to the normal vector.
''')
# svm and curve together
w=clf.coef_[0]
ticks=X[:,:].dot(w)
fig = plt.figure()
ax  = fig.add_subplot(111)
ax.hist(ticks[Y==0],density=True,color='b',bins=99,label='health control',alpha=0.6)
ax.hist(ticks[Y==1],density=True,color='r',bins=99,label='ASD',alpha=0.6)
ax.set_xlabel('param value')
ax.set_ylabel('probability')
quickspine(ax)
# quicksave('svm on trajectory')

# t test can seperate
stats.ttest_ind(ticks[Y==0],ticks[Y==1])
stats.ks_2samp(ticks[Y==0],ticks[Y==1])


# raw curve
y_values, bin_edges = np.histogram(ticks, bins=20,density=True)
bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
plt.plot(bin_centers, y_values, '-')
plt.hist(ticks, density=True)
plt.show()

# smoothed curve
density = stats.gaussian_kde(ticks)
n, x = np.histogram(ticks, bins=20,density=True)
plt.plot(x, density(x))
# plt.hist(ticks, density=True,bins=20,)
plt.show()

# per subject curve
print('there are some diff. asd seems to be a bi modal distribution. plot individuals to check')
w=clf.coef_[0]
fig = plt.figure(figsize=(6,3))
ax1  = fig.add_subplot(121)
ax2  = fig.add_subplot(122, sharey=ax1,sharex=ax1)
for i, (s,e) in enumerate(zip(cumsum[:-1], cumsum[1:])):
    ticks=X[s:e,:].dot(w)
    if s>=astartind: # ASD
        cm_subsection = linspace(0,1, 25) 
        colors = [ cm.gist_heat(x) for x in cm_subsection ]
        # ax2.hist(ticks,density=True,bins=33,alpha=0.6)
        density = stats.gaussian_kde(ticks)
        n, x = np.histogram(ticks, bins=33,density=True)
        ax2.plot(x, density(x),linewidth=1, color=colors[i-numhsub])
    else: # NT
        cm_subsection = linspace(0, 0.8, 25) 
        colors = [ cm.winter(x) for x in cm_subsection ]
        # ax1.hist(ticks,density=True,bins=33,alpha=0.6)
        density = stats.gaussian_kde(ticks)
        n, x = np.histogram(ticks, bins=33,density=True)
        ax1.plot(x, density(x), linewidth=1, color=colors[i])

    ax1.set_xlabel('param value')
    ax1.set_ylabel('probability density')
    quickspine(ax1)
    ax2.set_xlabel('param value')
    quickspine(ax2)
# quicksave('asd behavior trajectory svm')
print('the other peak suggest asd behavior. still they are mixed')




# individual action cost and control onset -----------------------------------

# version 1, cost vs time
ind=np.random.randint(low=0, high=len(tasks))
thetask=tasks[ind]

ax=plt.subplot()
invtag='h'
ax1,ax2=None, None
for isub in range(numhsub):
    thesub="{}sub{}".format(invtag,str(isub))
    k='res'+thesub
    if k in asd_data_set:
        theta=asd_data_set[k][0]
        costs=theta[-4:-2]
        k='data'+thesub
        _,actions,tasks=asd_data_set[k]
        indls=similar_trials2this(tasks, thetask, ntrial=2)
        subactions=[actions[i] for i in indls]
        subtasks=tasks[indls]
        costs=[np.linalg.norm(np.diff(a,axis=0),axis=1)  for a in subactions]
        for c in costs:
            ax.plot(c,'b',linewidth=0.5)
#         ax1,ax2=plotctrlasd(subactions,ax1=ax1,ax2=ax2)
# ax1.get_figure()

ax=plt.subplot()
invtag='a'
ax1,ax2=None, None
for isub in range(numhsub):
    thesub="{}sub{}".format(invtag,str(isub))
    k='res'+thesub
    if k in asd_data_set:
        theta=asd_data_set[k][0]
        costs=theta[-4:-2]
        k='data'+thesub
        _,actions,tasks=asd_data_set[k]
        indls=similar_trials2this(tasks, thetask, ntrial=2)
        subactions=[actions[i] for i in indls]
        subtasks=tasks[indls]
        costs=[np.linalg.norm(np.diff(a,axis=0),axis=1)  for a in subactions]
        for c in costs:
            x=np.arange(0,-len(c),-1)
            ax.plot(x,c,'r',linewidth=0.5)
#         ax1,ax2=plotctrlasd(subactions,ax1=ax1,ax2=ax2)
# ax1.get_figure()

# version2, cumsum, two side
ind=np.random.randint(low=0, high=len(tasks))
thetask=tasks[ind]
# subjective cost, cost*param
with initiate_plot(3,3,300) as f:
    ntrial=3
    dtlim=20
    ax=f.add_subplot(111)
    cm_subsection = linspace(0,1, 25) 
    colors = [ cm.gist_heat(x) for x in cm_subsection ]
    invtag='a'
    for isub in range(numhsub):
        thesub="{}sub{}".format(invtag,str(isub))
        k='res'+thesub
        if k in asd_data_set:
            theta=asd_data_set[k][0]
            costparams=theta[-4:-2]
            k='data'+thesub
            _,actions,tasks=asd_data_set[k]
            indls=similar_trials2this(tasks, thetask, ntrial=ntrial)
            subactions=[actions[i] for i in indls]
            # subtasks=tasks[indls]
            costs=[np.linalg.norm(np.diff(a,axis=0)*np.array(costparams).reshape(-1),axis=1)  for a in subactions]
            minlen=min([len(a) for a in costs])
            costsmu=np.mean(np.stack([a[:minlen] for a in costs]), axis=0)
            x=np.arange(0,-len(costsmu[:dtlim]),-1)
            ax.plot(x,np.cumsum(costsmu[:dtlim]),color=colors[isub],linewidth=0.5)
    invtag='h'
    cm_subsection = linspace(0, 0.5, 25) 
    colors = [ cm.winter(x) for x in cm_subsection ]
    for isub in range(numhsub):
        thesub="{}sub{}".format(invtag,str(isub))
        k='res'+thesub
        if k in asd_data_set:
            theta=asd_data_set[k][0]
            costparams=theta[-4:-2]
            k='data'+thesub
            _,actions,tasks=asd_data_set[k]
            indls=similar_trials2this(tasks, thetask, ntrial=ntrial)
            subactions=[actions[i] for i in indls]
            # subtasks=tasks[indls]
            costs=[np.linalg.norm(np.diff(a,axis=0)*np.array(costparams).reshape(-1),axis=1)  for a in subactions]
            minlen=min([len(a) for a in costs])
            costsmu=np.mean(np.stack([a[:minlen] for a in costs]), axis=0)
            ax.plot(np.cumsum(costsmu[:dtlim]),color=colors[isub],linewidth=0.5)
    quickleg(ax)
    quickspine(ax)
    ax.set_xlabel('time, dt')
    ax.set_ylabel('cumulative subjective cost')
    # quicksave('cumsum subjective cost asd vs nt for particular target ind={}'.format(ind))

# objective cost, no cost param
with initiate_plot(3,3,300) as f:
    ntrial=3
    dtlim=20
    ax=f.add_subplot(111)
    invtag='h'
    cm_subsection = linspace(0, 0.5, 25) 
    colors = [ cm.winter(x) for x in cm_subsection ]
    for isub in range(numhsub):
        thesub="{}sub{}".format(invtag,str(isub))
        k='res'+thesub
        if k in asd_data_set:
            k='data'+thesub
            _,actions,tasks=asd_data_set[k]
            indls=similar_trials2this(tasks, thetask, ntrial=ntrial)
            subactions=[actions[i] for i in indls]
            costs=[np.linalg.norm(np.diff(a,axis=0),axis=1)  for a in subactions]
            minlen=min([len(a) for a in costs])
            costsmu=np.mean(np.stack([a[:minlen] for a in costs]), axis=0)
            x=np.arange(0,len(costsmu[:dtlim]),1)
            ax.plot(x,np.cumsum(costsmu[:dtlim]),color=colors[isub],linewidth=0.5)

    cm_subsection = linspace(0,1, 25) 
    colors = [ cm.gist_heat(x) for x in cm_subsection ]
    invtag='a'
    for isub in range(numhsub):
        thesub="{}sub{}".format(invtag,str(isub))
        k='res'+thesub
        if k in asd_data_set:
            k='data'+thesub
            _,actions,tasks=asd_data_set[k]
            indls=similar_trials2this(tasks, thetask, ntrial=ntrial)
            subactions=[actions[i] for i in indls]
            costs=[np.linalg.norm(np.diff(a,axis=0),axis=1)  for a in subactions]
            minlen=min([len(a) for a in costs])
            costsmu=np.mean(np.stack([a[:minlen] for a in costs]), axis=0)
            x=np.arange(0,-len(costsmu[:dtlim]),-1)
            ax.plot(x,np.cumsum(costsmu[:dtlim]),color=colors[isub],linewidth=0.5)
    quickleg(ax)
    quickspine(ax)
    ax.set_xlabel('time, dt')
    ax.set_ylabel('cumulative cost')
    # quicksave('cumsum cost asd vs nt for particular target ind={}'.format(ind))

# version 3, asd-nt cost overhead
thetasks=tasks
costdiffs=[]
for thetask in thetasks:
    thiscostdiff=0
    invtag='a'
    for isub in range(numasub):
        thesub="{}sub{}".format(invtag,str(isub))
        k='res'+thesub
        if k in asd_data_set:
            theta=asd_data_set[k][0]
            costs=theta[-4:-2]
            k='data'+thesub
            _,actions,tasks=asd_data_set[k]
            indls=similar_trials2this(tasks, thetask, ntrial=2)
            subactions=[actions[i] for i in indls]
            subtasks=tasks[indls]
            costs=[np.linalg.norm(np.diff(a,axis=0),axis=1)  for a in subactions]
            thiscostdiff+=np.mean([np.mean(c) for c in costs])/numasub
    invtag='h'
    for isub in range(numhsub):
        thesub="{}sub{}".format(invtag,str(isub))
        k='res'+thesub
        if k in asd_data_set:
            theta=asd_data_set[k][0]
            costs=theta[-4:-2]
            k='data'+thesub
            _,actions,tasks=asd_data_set[k]
            indls=similar_trials2this(tasks, thetask, ntrial=2)
            subactions=[actions[i] for i in indls]
            subtasks=tasks[indls]
            costs=[np.linalg.norm(np.diff(a,axis=0),axis=1)  for a in subactions]
            thiscostdiff-=np.mean([np.mean(c) for c in costs])/numhsub
    costdiffs.append(thiscostdiff)

normedcostdiffs=normalizematrix(costdiffs)
# normedcostdiffs=np.linspace(0,10,len(thetasks))

with initiate_plot(3,3,300) as f:
    ax=f.add_subplot(111)
    im=ax.scatter(thetasks[:,0], thetasks[:,1], c=normedcostdiffs, cmap='bwr', vmin=-1, vmax=1)
    f.colorbar(im,ax=ax, label='ASD - NT cost')
    quickspine(ax)
    ax.axis('equal')
    ax.set_xlabel('world x [2m]')
    ax.set_ylabel('world y [2m]')
    # quicksave('asd-nt cost overhead')

# version4, cumsum same side, solid mean and fade individual
ind=np.random.randint(low=0, high=len(tasks))
thetask=tasks[ind]
# subjective cost, cost*param
with initiate_plot(3,3,300) as f:
    ntrial=3
    dtlim=20
    ax=f.add_subplot(111)
    cm_subsection = linspace(0,1, 25) 
    colors = [ cm.gist_heat(x) for x in cm_subsection ]
    invtag='a'
    allsubdata=[]
    for isub in range(numhsub):
        thesub="{}sub{}".format(invtag,str(isub))
        k='res'+thesub
        if k in asd_data_set:
            theta=asd_data_set[k][0]
            costparams=theta[-4:-2]
            k='data'+thesub
            _,actions,tasks=asd_data_set[k]
            indls=similar_trials2this(tasks, thetask, ntrial=ntrial)
            subactions=[actions[i] for i in indls]
            # subtasks=tasks[indls]
            costs=[np.linalg.norm(np.diff(a,axis=0)*np.array(costparams).reshape(-1),axis=1)  for a in subactions]
            minlen=min([len(a) for a in costs])
            costsmu=np.mean(np.stack([a[:minlen] for a in costs]), axis=0)
            x=np.arange(len(costsmu[:dtlim]))
            allsubdata.append(np.cumsum(costsmu[:dtlim]))
            ax.plot(x,np.cumsum(costsmu[:dtlim]),color='pink',linewidth=0.5)
    minlen=min([len(a) for a in allsubdata])
    allsubdata=[a[:minlen] for a in allsubdata]
    allsubdata=np.array(allsubdata)
    yerr=np.std(allsubdata,axis=0)
    for ii in range(minlen):
        if ii%2==0:
            yerr[ii]=0
    ax.errorbar(np.arange(minlen),np.mean(allsubdata,axis=0),yerr=yerr,color='r',linewidth=3)

    invtag='h'
    allsubdata=[]
    cm_subsection = linspace(0, 0.5, 25) 
    colors = [ cm.winter(x) for x in cm_subsection ]
    for isub in range(numhsub):
        thesub="{}sub{}".format(invtag,str(isub))
        k='res'+thesub
        if k in asd_data_set:
            theta=asd_data_set[k][0]
            costparams=theta[-4:-2]
            k='data'+thesub
            _,actions,tasks=asd_data_set[k]
            indls=similar_trials2this(tasks, thetask, ntrial=ntrial)
            subactions=[actions[i] for i in indls]
            # subtasks=tasks[indls]
            costs=[np.linalg.norm(np.diff(a,axis=0)*np.array(costparams).reshape(-1),axis=1)  for a in subactions]
            minlen=min([len(a) for a in costs])
            costsmu=np.mean(np.stack([a[:minlen] for a in costs]), axis=0)
            allsubdata.append(np.cumsum(costsmu[:dtlim]))
            ax.plot(np.cumsum(costsmu[:dtlim]),color='tab:blue',linewidth=0.5)
    minlen=min([len(a) for a in allsubdata])
    allsubdata=[a[:minlen] for a in allsubdata]
    allsubdata=np.array(allsubdata)
    yerr=np.std(allsubdata,axis=0)
    for ii in range(minlen):
        if ii%2!=0:
            yerr[ii]=0
    ax.errorbar(np.arange(minlen)+0.4,np.mean(allsubdata,axis=0),yerr=yerr,color='b',linewidth=3)

    quickleg(ax)
    quickspine(ax)
    ax.set_xlabel('time, dt')
    ax.set_ylabel('cumulative subjective cost')
    # quicksave('cumsum subjective cost asd vs nt for particular target ind={}'.format(ind))

# objective cost, no cost param
with initiate_plot(3,3,300) as f:
    ntrial=3
    dtlim=20
    ax=f.add_subplot(111)
    cm_subsection = linspace(0,1, 25) 
    colors = [ cm.gist_heat(x) for x in cm_subsection ]
    invtag='a'
    allsubdata=[]
    for isub in range(numhsub):
        thesub="{}sub{}".format(invtag,str(isub))
        k='res'+thesub
        if k in asd_data_set:
            theta=asd_data_set[k][0]
            costparams=theta[-4:-2]
            k='data'+thesub
            _,actions,tasks=asd_data_set[k]
            indls=similar_trials2this(tasks, thetask, ntrial=ntrial)
            subactions=[actions[i] for i in indls]
            # subtasks=tasks[indls]
            costs=[np.linalg.norm(np.diff(a,axis=0),axis=1)  for a in subactions]
            minlen=min([len(a) for a in costs])
            costsmu=np.mean(np.stack([a[:minlen] for a in costs]), axis=0)
            x=np.arange(len(costsmu[:dtlim]))
            allsubdata.append(np.cumsum(costsmu[:dtlim]))
            ax.plot(x,np.cumsum(costsmu[:dtlim]),color='pink',linewidth=0.5)
    minlen=min([len(a) for a in allsubdata])
    allsubdata=[a[:minlen] for a in allsubdata]
    allsubdata=np.array(allsubdata)
    yerr=np.std(allsubdata,axis=0)
    for ii in range(minlen):
        if ii%2==0:
            yerr[ii]=0
    ax.errorbar(np.arange(minlen),np.mean(allsubdata,axis=0),yerr=yerr,color='r',linewidth=3)

    invtag='h'
    allsubdata=[]
    cm_subsection = linspace(0, 0.5, 25) 
    colors = [ cm.winter(x) for x in cm_subsection ]
    for isub in range(numhsub):
        thesub="{}sub{}".format(invtag,str(isub))
        k='res'+thesub
        if k in asd_data_set:
            theta=asd_data_set[k][0]
            costparams=theta[-4:-2]
            k='data'+thesub
            _,actions,tasks=asd_data_set[k]
            indls=similar_trials2this(tasks, thetask, ntrial=ntrial)
            subactions=[actions[i] for i in indls]
            # subtasks=tasks[indls]
            costs=[np.linalg.norm(np.diff(a,axis=0),axis=1)  for a in subactions]
            minlen=min([len(a) for a in costs])
            costsmu=np.mean(np.stack([a[:minlen] for a in costs]), axis=0)
            allsubdata.append(np.cumsum(costsmu[:dtlim]))
            ax.plot(np.cumsum(costsmu[:dtlim]),color='tab:blue',linewidth=0.5)
    minlen=min([len(a) for a in allsubdata])
    allsubdata=[a[:minlen] for a in allsubdata]
    allsubdata=np.array(allsubdata)
    yerr=np.std(allsubdata,axis=0)
    for ii in range(minlen):
        if ii%2!=0:
            yerr[ii]=0
    ax.errorbar(np.arange(minlen)+0.4,np.mean(allsubdata,axis=0),yerr=yerr,color='b',linewidth=3)

    quickleg(ax)
    quickspine(ax)
    ax.set_xlabel('time, dt')
    ax.set_ylabel('cumulative subjective cost')
    # quicksave('cumsum cost asd vs nt for particular target ind={}'.format(ind))


# control onset ----------------------------
ind=np.random.randint(low=0, high=len(tasks))
thetask=tasks[ind]

ax=plt.subplot()
invtag='h'
ax1,ax2=None, None
hgroupa=[]
for isub in range(numhsub):
    thesub="{}sub{}".format(invtag,str(isub))
    k='res'+thesub
    if k in asd_data_set:
        theta=asd_data_set[k][0]
        costs=theta[-4:-2]
        k='data'+thesub
        _,actions,tasks=asd_data_set[k]
        indls=similar_trials2this(tasks, thetask, ntrial=2)
        subactions=[actions[i] for i in indls]
        subtasks=tasks[indls]
        # meana=np.mean(np.array([np.diff(a[:7,0]) for a in subactions]), axis=0)
        # meana=np.mean(np.array([np.diff(a[:7,1]) for a in subactions]), axis=0)
        meana=np.mean(np.array([np.linalg.norm(np.diff(a[:7]),axis=1) for a in subactions]), axis=0)
        ax.plot(meana,'b',linewidth=0.5)
        hgroupa.append(meana)
invtag='a'
ax1,ax2=None, None
agroupa=[]
for isub in range(numhsub):
    thesub="{}sub{}".format(invtag,str(isub))
    k='res'+thesub
    if k in asd_data_set:
        theta=asd_data_set[k][0]
        costs=theta[-4:-2]
        k='data'+thesub
        _,actions,tasks=asd_data_set[k]
        indls=similar_trials2this(tasks, thetask, ntrial=2)
        subactions=[actions[i] for i in indls]
        subtasks=tasks[indls]
        # meana=np.mean(np.array([np.diff(a[:7,0]) for a in subactions]), axis=0)
        # meana=np.mean(np.array([np.diff(a[:7,1]) for a in subactions]), axis=0)
        meana=np.mean(np.array([np.linalg.norm(np.diff(a[:7]),axis=1) for a in subactions]), axis=0)
        ax.plot(meana,'r',linewidth=0.5)
        agroupa.append(meana)
quickspine(ax)
# quickleg(ax)
ax.set_xlabel('time, dt')
ax.set_ylabel('control acceration')


with initiate_plot(2,2,300) as fig:
    ax=fig.add_subplot(111)
    ax.errorbar(np.arange(len(hgroupa[0])),np.mean(np.array(hgroupa), axis=0),yerr=np.std(np.array(hgroupa), axis=0), color='b', label='NT mean')
    ax.errorbar(np.arange(len(agroupa[0]))+0.2,np.mean(np.array(agroupa), axis=0),yerr=np.std(np.array(agroupa), axis=0), color='r', label='ASD mean')
    quickspine(ax)
    ax.set_xlabel('time, dt')
    ax.set_ylabel('control acceration')
    invtag='h'
    ax1,ax2=None, None
    hgroupa=[]
    for isub in range(numhsub):
        thesub="{}sub{}".format(invtag,str(isub))
        k='res'+thesub
        if k in asd_data_set:
            theta=asd_data_set[k][0]
            costs=theta[-4:-2]
            k='data'+thesub
            _,actions,tasks=asd_data_set[k]
            indls=similar_trials2this(tasks, thetask, ntrial=2)
            subactions=[actions[i] for i in indls]
            subtasks=tasks[indls]
            # meana=np.mean(np.array([np.diff(a[:7,0]) for a in subactions]), axis=0)
            # meana=np.mean(np.array([np.diff(a[:7,1]) for a in subactions]), axis=0)
            meana=np.mean(np.array([np.linalg.norm(np.diff(a[:7]),axis=1) for a in subactions]), axis=0)
            ax.plot(meana,'tab:blue',linewidth=0.5, label='NT subject')
            hgroupa.append(meana)
    invtag='a'
    ax1,ax2=None, None
    agroupa=[]
    for isub in range(numhsub):
        thesub="{}sub{}".format(invtag,str(isub))
        k='res'+thesub
        if k in asd_data_set:
            theta=asd_data_set[k][0]
            costs=theta[-4:-2]
            k='data'+thesub
            _,actions,tasks=asd_data_set[k]
            indls=similar_trials2this(tasks, thetask, ntrial=2)
            subactions=[actions[i] for i in indls]
            subtasks=tasks[indls]
            # meana=np.mean(np.array([np.diff(a[:7,0]) for a in subactions]), axis=0)
            # meana=np.mean(np.array([np.diff(a[:7,1]) for a in subactions]), axis=0)
            meana=np.mean(np.array([np.linalg.norm(np.diff(a[:7]),axis=1) for a in subactions]), axis=0)
            ax.plot(meana,'pink',linewidth=0.5, label='ASD subject')
            agroupa.append(meana)
    quickleg(ax)
    # quicksave('asd accerate control faster v2')



a,n,x=[-4,3,-5,9,-1,0,2],7,3
for i in range(n):
    a[i] = abs(a[i])

# Sort the array
a = sorted(a)
print('sorted',a)
# Assign K = N - K
x = n - x

# Count number of zeros
z = a.count(0)
print('n zeros',z)

# If number of zeros if greater
if (x > n - z):
    print("-1")
    
for i in range(0, n, 2):
    if x <= 0:
        break

    # Using 2nd operation convert
    # it into one negative
    a[i] = -a[i]
    x -= 1
for i in range(n - 1, -1, -1):
    if x <= 0:
        break

    # Using 2nd operation convert
    # it into one negative
    if (a[i] > 0):
        a[i] = -a[i]
        x -= 1

# Print array
for i in range(n):
    print(a[i], end = " ")




list(range(10,0, -1))



