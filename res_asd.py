
# for res_asd figures
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



# ASD theta bar (not in use) --------------------------------------------
logls=['/data/human/fixragroup','/data/human/clusterpaperhgroup']
monkeynames=['ASD', 'Ctrl' ]
mus,covs,errs=[],[],[]
for inv in logls:
    finaltheta,finalcov, err=process_inv(inv,ind=60)
    mus.append(finaltheta)
    covs.append(finalcov)
    errs.append(err)

ax=multimonkeytheta(monkeynames, mus, covs, errs, )
ax.set_yticks([0,1,2])
ax.plot(np.linspace(-1,9),[2]*50)
ax.get_figure()




# % behavioral small gain prior, and we confirmed it ------------------------

# load human without feedback data
datapath=Path("/data/human/wohgroup")
with open(datapath, 'rb') as f:
    hstates, hactions, htasks = pickle.load(f)

datapath=Path("/data/human/woagroup")
with open(datapath, 'rb') as f:
    astates, aactions, atasks = pickle.load(f)
# get the side tasks (stright trials do not have curvature)
res=[]
for task in htasks:
    d,a=xy2pol(task, rotation=False)
    # if  env.min_angle/2<=a<env.max_angle/2:
    if a<=-pi/5*0.7 or a>=pi/5*0.7:
        res.append(task)
sidetasks=np.array(res)

ares=np.array([s[-1].tolist() for s in astates])
# radial and angular distance response
ardist=np.linalg.norm(ares[:,:2],axis=1)
aadist=np.arctan2(ares[:,1],ares[:,0])
# radial and angular distance target
atasksda=np.array([xy2pol(t,rotation=False) for t in atasks])
artar=atasksda[:,0]
aatar=atasksda[:,1] # hatar=np.arctan2(htasks[:,1],htasks[:,0])
artarind=np.argsort(artar)
aatarind=sorted(aatar)


hres=np.array([s[-1].tolist() for s in hstates])
# radial and angular distance response
hrdist=np.linalg.norm(hres[:,:2],axis=1)
hadist=np.arctan2(hres[:,1],hres[:,0])
# radial and angular distance target
htasksda=np.array([xy2pol(t,rotation=False) for t in htasks])
hrtar=htasksda[:,0]
hatar=htasksda[:,1] # hatar=np.arctan2(htasks[:,1],htasks[:,0])
hrtarind=np.argsort(hrtar)
hatarind=sorted(hatar)


# plot the radial error and angular error

with initiate_plot(4,2,300) as f:
    ax=f.add_subplot(121)
    ax.scatter(hatar,hadist,s=1,alpha=0.5,color='b')
    ax.scatter(aatar,aadist,s=1,alpha=0.2,color='r')
    ax.set_xlim(-0.7,0.7)
    ax.set_ylim(-2,2)
    ax.plot([-1,1],[-1,1],'k',alpha=0.5)
    ax.set_xlabel('target angle')
    ax.set_ylabel('response angle')
    # ax.axis('equal')
    quickspine(ax)

    ax=f.add_subplot(122)
    ax.scatter(hrtar,hrdist,s=1,alpha=0.5,color='b')
    ax.scatter(artar,ardist,s=1,alpha=0.3,color='r')
    # ax.plot([.5,3],[.5,3],'k',alpha=0.5)
    ax.set_xlim(.5,3)
    ax.set_ylim(0.5,5)
    ax.plot([0,3],[0,3],'k',alpha=0.5)
    ax.set_xlabel('target distance')
    ax.set_ylabel('response distance')
    quickspine(ax)
    # ax.axis('equal')
    plt.tight_layout()

# plot the inferred gains (biases?)

# or plot the model reproduced radial error and angular error





# per subject behavior ------------------------
datapath=Path("/data/human/hgroup")
with open(datapath, 'rb') as f:
    hstates, hactions, htasks = pickle.load(f)

datapath=Path("/data/human/agroup")
with open(datapath, 'rb') as f:
    astates, aactions, atasks = pickle.load(f)

filename='/data/human/fbsimple.mat'
data=loadmat(filename)

# seperate into two groups
hdata,adata=[],[]
for d in data:
    if d['name'][0]=='A':
        adata.append(d)
    else:
        hdata.append(d)
print('we have these number of health and autiusm subjects',len(hdata),len(adata))

hsublen=[len(eachsub['targ']['r']) for eachsub in hdata]
asublen=[len(eachsub['targ']['r']) for eachsub in adata]
hcumsum=np.cumsum(hsublen)
acumsum=np.cumsum(asublen)


# load inv data
numhsub,numasub=25,14
foldername='persub1cont'
logs={'a':'/data/human/fixragroup','h':'/data/human/clusterpaperhgroup'}

invres={'a':[],'h':[]}
for isub in range(numhsub):
    dataname="hsub{}".format(str(isub))
    savename=Path("/data/human/{}".format(foldername))/"invhsub{}".format(str(isub))
    if savename.is_file():
        invres['h'].append(process_inv(savename,ind=31, usingbest=True))
for isub in range(numasub):
    dataname="asub{}".format(str(isub))
    savename=Path("/data/human/{}".format(foldername))/"invasub{}".format(str(isub))
    if savename.is_file():
        invres['a'].append(process_inv(savename,ind=31, usingbest=True))


# plot overhead all trials
for isub in range(len(hdata)):
    s=0 if isub==0 else hcumsum[isub-1]
    e=hcumsum[isub]
    substates=hstates[s:e]
    subtasks=htasks[s:e]
    ax=quickoverhead_state(substates,subtasks)
    print('infered theta: [\n', invres['h'][isub][0])


for isub in range(len(adata)):
    s=0 if isub==0 else acumsum[isub-1]
    e=acumsum[isub]
    substates=astates[s:e]
    subtasks=atasks[s:e]
    ax=quickoverhead_state(substates,subtasks)
    print('infered theta: [\n', invres['a'][isub][0])


# test, plot particular trial instead of all, test
isub=0
s=0 if isub==0 else acumsum[isub-1]
e=acumsum[isub]
substates=astates[s:e]
subtasks=atasks[s:e]
# selections
selctinds=similar_trials2this(subtasks, [1.8,1.3],ntrial=2)
a=subtasks[selctinds]
b=[substates[i] for i in selctinds]
# plt.scatter(a[:,0]*200,a[:,1]*200)
# plt.axis('equal')
fig, ax = plt.subplots()
ax=quickoverhead_state(b,a,ax=ax,goalcircle=True)
# ax.add_patch(plt.Circle((a[0,0],a[0,1]), 0.13, color='y', alpha=0.5))
ax.set_title(invres['a'][isub][0][1].item())
ax.set_xticklabels(np.array(ax.get_xticks()).astype('int')*200)
ax.set_yticklabels(np.array(ax.get_xticks()).astype('int')*200)
isub+=1
# quicksave('asd subject {} example path'.format(isub),fig=fig)





isub=0
s=0 if isub==0 else hcumsum[isub-1]
e=hcumsum[isub]
substates=hstates[s:e]
subtasks=htasks[s:e]
# selections
selctinds=similar_trials2this(subtasks, [1.8,1.3],ntrial=2)
a=subtasks[selctinds]
b=[substates[i] for i in selctinds]
# plt.scatter(a[:,0]*200,a[:,1]*200)
# plt.axis('equal')
fig, ax = plt.subplots()
ax=quickoverhead_state(b,a,ax=ax,goalcircle=True)
# ax.add_patch(plt.Circle((a[0,0],a[0,1]), 0.13, color='y', alpha=0.5))
ax.set_title(invres['h'][isub][0][1].item())
ax.set_xticklabels(np.array(ax.get_xticks()).astype('int')*200)
ax.set_yticklabels(np.array(ax.get_xticks()).astype('int')*200)
isub+=1
# quicksave('control subject {} example path'.format(isub),fig=fig)





# plot the inv res scatteers bar
subshift=0.015
with initiate_plot(7,2,300) as fig:
    ax=fig.add_subplot(111)
    quickspine(ax)
    ax.set_xticks(list(range(10)))
    ax.set_xticklabels(theta_names, rotation=45, ha='right')
    ax.set_ylabel('inferred param value')
    colory=np.linspace(0,2,50) # the verticle range of most parameters
    colory=torch.linspace(0,2,50) # the verticle range of most parameters
    for ithsub, log in enumerate(invres['h']): # each control subject
        for i, mu in enumerate(log[0]): # each parameter
            std=(torch.diag(log[1])**0.5)[i]*0.1
            prob=lambda y: torch.exp(-0.5*(y-mu)**2/std**2)
            proby=prob(colory)
            for j in range(len(colory)-1):
                c=proby[j].item()
                plt.plot([i-(ithsub+5)*subshift,i-(ithsub+5)*subshift],[colory[j],colory[j+1]],c=[0,0,1], alpha=c)
    for ithsub, log in enumerate(invres['a']): # each asd subject
        for i, mu in enumerate(log[0]): # each parameter
            std=(torch.diag(log[1])**0.5)[i]*0.1
            prob=lambda y: torch.exp(-0.5*(y-mu)**2/std**2)
            proby=prob(colory)
            for j in range(len(colory)-1):
                c=proby[j].item()
                plt.plot([i+(ithsub+5)*subshift,i+(ithsub+5)*subshift],[colory[j],colory[j+1]],c=[1,0,0], alpha=c)
    for ithsub, logfile in enumerate(logls): # each mk subject
        log=process_inv(logfile,ind=60)
        for i, mu in enumerate(log[0]): # each parameter
            std=(torch.diag(log[1])**0.5)[i]*0.1
            prob=lambda y: torch.exp(-0.5*(y-mu)**2/std**2)
            proby=prob(colory)
            for j in range(len(colory)-1):
                c=proby[j].item()
                plt.plot([i+(ithsub+25)*subshift,i+(ithsub+25)*subshift],[colory[j],colory[j+1]],c=[0.,0.7,0], alpha=c)
    quicksave('human per sub pixel line for poster with mk')
    # quicksave('human per sub pixel line new')


x=np.array(list(range(10)))
hy=np.array([np.array(log[0].view(-1)) for log in invres['h']])
ay=np.array([np.array(log[0].view(-1)) for log in invres['a']])
plt.errorbar(x, np.mean(hy,axis=0), np.std(hy,axis=0))
plt.errorbar(x+0.3, np.mean(ay,axis=0), np.std(ay,axis=0))


# biased degree v
''' bias*(p/p+o) '''
biash=(pi/2-hy[:,0])/(pi/2) * (hy[:,2]/(hy[:,2]+hy[:,4]))
biasa=(pi/2-ay[:,0])/(pi/2) * (ay[:,2]/(ay[:,2]+ay[:,4]))
plt.figure(figsize=(1,3))
plt.scatter([0]*len(biash), biash,alpha=0.1)
plt.scatter([1]*len(biasa), biasa,alpha=0.1)


# biased degree w
biash=(pi/2-hy[:,1])/(pi/2) * (hy[:,3]/(hy[:,3]+hy[:,5]))
biasa=(pi/2-ay[:,1])/(pi/2) * (ay[:,3]/(ay[:,3]+ay[:,5]))
plt.figure(figsize=(1,3))
plt.scatter([0]*len(biash), biash,alpha=0.1)
plt.scatter([1]*len(biasa), biasa,alpha=0.1)


# cost w
plt.figure(figsize=(1,3))
plt.scatter([0]*len(biash), hy[:,7],alpha=0.3)
plt.scatter([1]*len(biasa), ay[:,7],alpha=0.3)


stats.ttest_ind(biash,biasa)
stats.ttest_ind(hy[:,1],ay[:,1])
stats.ttest_ind(hy[:,7],ay[:,7])

npsummary(biash)
npsummary(biasa)




print('''
sliding along parameter axis and show this affects path curvature. 
show the likelihood of each location.
''')

# load agent and task
env=ffacc_real.FireFlyPaper(arg)
env.episode_len=50
env.debug=1
env.terminal_vel=0.2
phi=torch.tensor([[1],
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

# define the midpoint between ASD and healthy 
logls=['/data/human/fixragroup','/data/human/clusterpaperhgroup']
monkeynames=['ASD', 'Ctrl' ]

mus,covs,errs=[],[],[]
thetas=[]
for inv in logls:
    finaltheta,finalcov, err=process_inv(inv,ind=60,removegr=False)
    mus.append(np.array(finaltheta).reshape(-1))
    covs.append(finalcov)
    errs.append(err)
    thetas.append(finaltheta)
thetas=torch.tensor(mus)
theta_init=thetas[0]
theta_final=thetas[1]

# load theta distributions and compute svm
alltag=[]
alltheta=[]
loglls=[]
with open(logls[0],'rb') as f:
    log=pickle.load(f)
    res=[l[2] for l in log[19:99]]
    for r in res:
        for point in r:
            alltheta.append(point[0]) # theta
            loglls.append(point[1])
            alltag.append(0)
with open(logls[1],'rb') as f:
    log=pickle.load(f)
    res=[l[2] for l in log[19:99]]
    for r in res:
        for point in r:
            alltheta.append(point[0]) # theta
            loglls.append(point[1])
            alltag.append(1)

alltheta=np.array(alltheta)
alltag=np.array(alltag)

clf = svm.SVC(kernel="linear", C=1000)
clf.fit(alltheta, alltag)
w = clf.coef_[0] # the normal vector
midpoint=(mus[0]+mus[1])/2
lb=np.array([0,0,0,0,0,0,0.129,0,0,0,0])
hb=np.array([1,2,1,1,1,1,0.131,2,2,1,1])


# vary from asd to health (along the svm normal vector)
theta_init=np.min(midpoint-lb)/w[np.argmin(hb-midpoint)]*-w*10+midpoint
theta_final=np.min(midpoint-lb)/w[np.argmin(midpoint-lb)]*w*10+midpoint
theta_init,theta_final=torch.tensor(theta_init).view(-1,1).float(),torch.tensor(theta_final).view(-1,1).float()
vary_theta_new(agent, env, phi, theta_init, theta_final,5,etask=[1,1], ntrials=20,savename='vary asd axis')

# vary obs noise (x axis in delta plot)
theta_init=midpoint.copy();theta_final=midpoint.copy()
theta_init[1]=1
theta_final[1]=1
theta_init[5]=2
theta_final[5]=0
theta_init,theta_final=torch.tensor(theta_init).view(-1,1).float(),torch.tensor(theta_final).view(-1,1).float()
vary_theta_new(agent, env, phi, theta_init, theta_final,5,etask=[1,1], ntrials=20,savename='vary trust obs axis')

# vary small gain to correct
theta_init=midpoint.copy();theta_final=midpoint.copy()
theta_init[1]=0.5
theta_final[1]=2
theta_init,theta_final=torch.tensor(theta_init).view(-1,1).float(),torch.tensor(theta_final).view(-1,1).float()
vary_theta_new(agent, env, phi, theta_init, theta_final,5,etask=[1,1], ntrials=20,savename='vary assumed gain')
# quicksave('0 wgain to 2')

# vary bias (y axis in delta plot)
theta_init=midpoint.copy();theta_final=midpoint.copy()
theta_init[1]=0
theta_final[1]=2
theta_init,theta_final=torch.tensor(theta_init).view(-1,1).float(),torch.tensor(theta_final).view(-1,1).float()
vary_theta_new(agent, env, phi, theta_init, theta_final,5,etask=[1,1], ntrials=20)


# the delta logll plot
with open('distinguishparamZtwonoisessmaller2finer19', 'rb') as f:
    paramls,Z= pickle.load(f)
formatedZ=np.array(Z).reshape(int(len(Z)**0.5),int(len(Z)**0.5)).T
truedelta=mus[1]-mus[0]

with initiate_plot(3,3,300) as f:
    ax=f.add_subplot(111)
    # im=ax.imshow(formatedZ,origin='lower', extent=(X[0],X[-1],Y[0],Y[-1]),aspect='auto',vmin=-88, vmax=-73)
    # im=ax.imshow(formatedZ[:-1,:],origin='lower', extent=(X[0],X[-2],Y[0],Y[-1]),aspect='auto',vmin=-103, vmax=-73)
    formatedZ=np.clip(formatedZ[::-1,::-1],-99,0)
    im=ax.contourf(formatedZ,origin='lower', extent=(X[0],X[-1],Y[0],Y[-1]),vmin=-88, vmax=-73,cmap='binary')
    ax.set_aspect('equal')
    plt.colorbar(im,label='joint log likelihood') 
    ax.scatter(0,0,label='intermediate',color='k') # midpoint, 0,0
    ax.scatter(max(X)-0.05,0,label='unreliable obs (ASD)',color='r',marker='>') 
    ax.scatter(min(X)+0.05,0,label='reliable obs (ASD)',color='orange',marker='<') 
    ax.scatter(0, 0.2,label='smaller gain prior (ASD)',color='g',marker='^') 
    ax.scatter(0, -0.2,label='larger gain prior (ASD)',color='b',marker='v') 
    dx,dy=0.05*w[1],0.05*w[5]
    ax.plot([-dx,dx],[-dy,dy],label='ASD - control',color='k',) 
    ax.set_xlabel('delta observation noise')
    ax.set_ylabel('delta prediction gain')
    # ax.scatter(truedelta[5]/2,truedelta[3]/2,label='inferred delta') # inferred delta
    quickleg(ax)
    quickspine(ax)

    quicksave('gray logll obs prediction vs obs with label')



# added 9/12, short long validation --------------------
# load theta infered for short trials

numhsub,numasub=25,14
logs={'a':'/data/human/fixragroup','h':'/data/human/clusterpaperhgroup'}

# full inverse
foldername='persub1cont'
invres={'a':[],'h':[]}
for isub in range(numhsub):
    dataname="hsub{}".format(str(isub))
    savename=Path("/data/human/{}".format(foldername))/"invhsub{}".format(str(isub))
    if savename.is_file():
        invres['h'].append(process_inv(savename,ind=31, usingbest=True))
for isub in range(numasub):
    dataname="asub{}".format(str(isub))
    savename=Path("/data/human/{}".format(foldername))/"invasub{}".format(str(isub))
    if savename.is_file():
        invres['a'].append(process_inv(savename,ind=31, usingbest=True))

# short inverse
foldername='persubshort3of4'
invres['ashort']=[]
invres['hshort']=[]
for isub in range(numhsub):
    dataname="hsub{}".format(str(isub))
    savename=Path("/data/human/{}".format(foldername))/"invhsub{}".format(str(isub))
    if savename.is_file():
        invres['hshort'].append(process_inv(savename,ind=31, usingbest=True))
for isub in range(numasub):
    dataname="asub{}".format(str(isub))
    savename=Path("/data/human/{}".format(foldername))/"invasub{}".format(str(isub))
    if savename.is_file():
        invres['ashort'].append(process_inv(savename,ind=31, usingbest=True))

fullinvh=[log[0] for log in invres['h']]
shortinvh=[log[0] for log in invres['hshort']]

plt.plot([theta[0] for theta in fullinvh])
plt.plot([theta[0] for theta in shortinvh])


# overhead of example trial -----------------------------------------------------------\

env=ffacc_real.FireFlyPaper(arg)
env.debug=True
env.terminal_vel=0.05
agent_=TD3.load('trained_agent/paper.zip')
agent=agent_.actor.mu.cpu()


# load data 
asd_data_set={}
fulltrainfolder='persub1cont'
parttrainfolder='persubshort3of4'
for invtag in ['h','a']:
    for isub in range(numhsub):
        thesub="{}sub{}".format(invtag,str(isub))
        trainname=Path("/data/human/{}/inv{}sub{}".format(foldername,invtag,str(isub)))
        evalname=Path("/data/human/{}/eval_inv{}sub{}".format(foldername,invtag,str(isub)))
        fullinverseres=Path("/data/human/{}".format(fulltrainfolder))/"invhsub{}".format(str(isub))
        partinverseres=Path("/data/human/{}".format(fulltrainfolder))/"invhsub{}".format(str(isub))
        # load inv res
        if partinverseres.is_file():
            asd_data_set['partres'+thesub]=process_inv(partinverseres, usingbest=True, removegr=False)
        # if fullinverseres.is_file():
            asd_data_set['res'+thesub]=process_inv(fullinverseres, usingbest=True, removegr=False)
        # load data
        if Path('/data/human/{}'.format(thesub)).is_file():
            with open('/data/human/{}'.format(thesub), 'rb') as f:
                states, actions, tasks = pickle.load(f)
            print(len(states))
            asd_data_set['data'+thesub]=states, actions, tasks
        # load test logll
        if evalname.is_file():
            with open(evalname, 'rb') as f:
                a = pickle.load(f)
                asd_data_set['trainlogll'+thesub] = a[-1][0]
                asd_data_set['testlogll'+thesub] = a[-1][1]

thesub='hsub9'
states, actions, tasks = asd_data_set['data'+thesub]

# select long trials (test set)
taskdist=np.array([np.linalg.norm(x) for x in tasks])
distsortind=np.argsort(taskdist)
testind=distsortind[int(len(distsortind)*3/4):]
states, actions, tasks = [states[t] for t in testind], [actions[t] for t in testind], tasks[testind]

ind=np.random.randint(low=0, high=len(tasks))
thetask=tasks[ind]
indls=similar_trials2this(tasks, thetask, ntrial=3)

substates=[states[i] for i in indls]
subactions=[actions[i] for i in indls]
subtasks=np.array(tasks)[indls]

# run trial with model (fully trained) ---------------------------
ax=plotoverhead_simple(substates,thetask,color='b',label=thesub,ax=None)
modelstates,modelactions=run_trials_multitask(agent, env, phi, asd_data_set['res'+thesub][0], subtasks, ntrials=1)
T=max([len(s) for s in substates])
modelstates=[m[:T] for m in modelstates]

# plot overhead
ax=plotoverhead_simple(modelstates,thetask,color='r',label='model',ax=ax)
ax.get_figure()
# quicksave('{} model vs data overhead'.format(thesub),fig=ax.get_figure())

# plot control curve 
fig=plotctrl_vs(subactions, modelactions, color1='b', color2='r', label1=thesub, label2='model', alpha=1)
# quicksave('{} model vs data control curve '.format(thesub),fig=fig)


# run trial with model (part trained for test) ---------------------------
ax=plotoverhead_simple(substates,thetask,color='b',label=thesub,ax=None)
# run trial with model 
modelstates,modelactions=run_trials_multitask(agent, env, phi, asd_data_set['partres'+thesub][0], subtasks, ntrials=1)
T=max([len(s) for s in substates])
modelstates=[m[:T] for m in modelstates]

# plot overhead
ax=plotoverhead_simple(modelstates,thetask,color='r',label='model',ax=ax)
ax.get_figure()
# quicksave('{} model vs data overhead testset'.format(thesub),fig=ax.get_figure())

# plot control curve 
fig=plotctrl_vs(subactions, modelactions, color1='b', color2='r', label1=thesub, label2='model', alpha=1)
# quicksave('{} model vs data control curve testset'.format(thesub),fig=fig)




# compare validation logll and test logll -----------------------------------------------
trainloglls=[]
testloglls=[]
subnames=[]
foldername='persubshort3of4'
for invtag in ['h','a']:
    for isub in range(numhsub):
        thesub="{}sub{}".format(invtag,str(isub))
        if 'trainlogll'+thesub in asd_data_set:
            trainloglls.append(asd_data_set['trainlogll'+thesub] )
            testloglls.append(asd_data_set['testlogll'+thesub] )
            subnames.append(thesub)

for trainlogll, testlogll, thesub in zip(trainloglls,testloglls,subnames):
    with initiate_plot(2,2,300) as fig:
        ax=fig.add_subplot(111)
        ax.hist(trainlogll, color='b', label='training data', bins=30, density=True)
        ax.hist(testlogll,  color='r', label='testing data', bins=30, density=True)
        quickspine(ax)
        ax.set_xlabel('– log likelihood')
        ax.set_ylabel('probability')
        ax.set_title(thesub)
        quickleg(ax)

        # quicksave('eval logll hist {}'.format(thesub))

# style 1, all together
with initiate_plot(2,2,300) as fig:
    ax=fig.add_subplot(111)
    for trainlogll, testlogll, thesub in zip(trainloglls,testloglls,subnames):
        ax.scatter(np.zeros_like(trainlogll),trainlogll, color='b', label='each subject log likelihood')
        ax.scatter(np.ones_like(testlogll),testlogll, color='r', label='each subject log likelihood ')
        lines=np.vstack([sorted(trainlogll),sorted(testlogll)])
        ax.plot(lines, color='yellow', alpha=0.2)
    quickspine(ax)
    ax.set_ylabel('– log likelihood')
    ax.set_xticks([0,1])
    ax.set_xticklabels(['traning', 'testing'])
    quickleg(ax)
    # quicksave('all subjects logll train vs test')


# style 2, sub by sub
with initiate_plot(4,2,300) as fig:
    i=0
    increment=0.3
    ax=fig.add_subplot(111)
    for trainlogll, testlogll, thesub in zip(trainloglls,testloglls,subnames):
        ax.scatter(np.zeros_like(trainlogll)+i,trainlogll, color='b', label='each subject training log likelihood',s=1)
        ax.scatter(np.zeros_like(testlogll)+i+increment,testlogll, color='r', label='each subject testing log likelihood ',s=1)
        
        lines=np.vstack([sorted(trainlogll),sorted(testlogll)]).T
        for y in lines:
            ax.plot([i, i+increment],y, color='yellow', alpha=0.2)
        i+=1
    quickspine(ax)
    ax.set_ylabel('– log likelihood')
    ax.set_xlabel('subjects')
    # ax.set_xticks([0,1])
    # ax.set_xticklabels(['traning', 'testing'])
    # quickleg(ax)
    # quicksave('each subjects logll train vs test')




# style 1, all together
with initiate_plot(2,2,300) as fig:
    ax=fig.add_subplot(111)
    for trainlogll, testlogll, thesub in zip(trainloglls,testloglls,subnames):
        ax.scatter(trainlogll,testlogll,s=1)
    ax.plot([0,20],[0,20])


