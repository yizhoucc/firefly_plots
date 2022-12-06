
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


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

env=ffacc_real.FireFlyPaper(arg)
env.debug=True
env.terminal_vel=0.05
agent_=TD3.load('trained_agent/paper.zip')
agent=agent_.actor.mu.cpu()



# # ASD theta bar (not in use) --------------------------------------------
# logls=['/data/human/fixragroup','/data/human/clusterpaperhgroup']
# monkeynames=['ASD', 'Ctrl' ]
# mus,covs,errs=[],[],[]
# for inv in logls:
#     finaltheta,finalcov, err=process_inv(inv,ind=60)
#     mus.append(finaltheta)
#     covs.append(finalcov)
#     errs.append(err)

# ax=multimonkeytheta(monkeynames, mus, covs, errs, )
# ax.set_yticks([0,1,2])
# ax.plot(np.linspace(-1,9),[2]*50)
# ax.get_figure()




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

datapath=Path("/data/human/wohgroup")
with open(datapath, 'rb') as f:
    wohstates, wohactions, wohtasks = pickle.load(f)

datapath=Path("/data/human/woagroup")
with open(datapath, 'rb') as f:
    woastates, woaactions, woatasks = pickle.load(f)


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


# test, plot particular trial instead of all
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




# feedback vs no feedback overhead -------------------------------


fig, ax = plt.subplots()
selctinds=similar_trials2this(htasks, [1.8,1.3],ntrial=2)
a=htasks[selctinds] # tasks
b=[hstates[i] for i in selctinds] # states
ax=quickoverhead_state(b,a,ax=ax,goalcircle=True)
ax.get_figure()

fig, ax = plt.subplots()
selctinds=similar_trials2this(wohtasks, [1.8,1.3],ntrial=2)
a=wohtasks[selctinds] # tasks
b=[wohstates[i] for i in selctinds] # states
ax=quickoverhead_state(b,a,ax=ax,goalcircle=True)
ax.get_figure()


ax.set_title(invres['a'][isub][0][1].item())
ax.set_xticklabels(np.array(ax.get_xticks()).astype('int')*200)
ax.set_yticklabels(np.array(ax.get_xticks()).astype('int')*200)



quickoverhead_state(hstates[100:200],htasks[100:200])
quickoverhead_state(wohstates[100:200],wohtasks[100:200])

quickoverhead_state(astates[100:200],atasks[100:200])
quickoverhead_state(woastates[100:200],woatasks[100:200])



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
    # quicksave('human per sub pixel line for poster with mk')
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


# t test each papram
for i in range(len(theta_names)):
    print(theta_names[i],'\n',stats.ttest_ind(hy[:,i],ay[:,i]))
    print('summary, asd \n', npsummary(ay[:,i]))
    print('summary, asd \n', npsummary(hy[:,i]))

stats.ttest_ind(biash,biasa)
npsummary(biash)
npsummary(biasa)



# vary params -----------------------------------------------------
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

# vary x(forward) uncertainty (y axis in delta plot)
theta_init=midpoint.copy();theta_final=midpoint.copy()
theta_init[-2]=0
theta_final[-2]=2
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
env.terminal_vel=0.1
agent_=TD3.load('trained_agent/paper.zip')
agent=agent_.actor.mu.cpu()
phi[0]=1

# load data 
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
                asd_data_set['testlogll'+thesub] = a[-1][1]
        
        # load train logll
        if partinverseres.is_file():
            with open(partinverseres, 'rb') as f:
                a = pickle.load(f)
                asd_data_set['trainlogll'+thesub] = a[-1][0]
                

thesub='hsub0'
states, actions, tasks = asd_data_set['data'+thesub]

# select long trials (test set)
taskdist=np.array([np.linalg.norm(x) for x in tasks])
distsortind=np.argsort(taskdist)
testind=distsortind[int(len(distsortind)*3/4):]
states, actions, tasks = [states[t] for t in testind], [actions[t] for t in testind], tasks[testind]

ind=np.random.randint(low=0, high=len(tasks))
thetask=tasks[ind]
indls=similar_trials2this(tasks, thetask, ntrial=3)
print(ind)

substates=[states[i] for i in indls]
subactions=[actions[i] for i in indls]
subtasks=np.array(tasks)[indls]

# run trial with model (fully trained) ---------------------------
ax=plotoverhead_simple(substates,thetask,color='b',label=thesub,ax=None)
modelstates,modelactions=run_trials_multitask(agent, env, phi, asd_data_set['res'+thesub][0], subtasks, ntrials=1, action_noise=0.05)
T=max([len(s) for s in substates])
modelstates=[m[:T] for m in modelstates]

# plot overhead
ax=plotoverhead_simple(modelstates,thetask,color='r',label='model',ax=ax,plotgoal=True)
ax.get_figure()
# quicksave('{} model vs data overhead ind={}'.format(thesub, ind),fig=ax.get_figure())

# plot control curve 
fig=plotctrl_vs(subactions, modelactions, color1='b', color2='r', label1=thesub, label2='model', alpha=1)
# quicksave('{} model vs data control curve ind={}'.format(thesub, ind),fig=fig)


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
for invtag in ['h','a']:
    for isub in range(25):
        thesub="{}sub{}".format(invtag,str(isub))
        subnames.append(thesub)
        if 'trainlogll'+thesub in asd_data_set:
            trainloglls.append(asd_data_set['trainlogll'+thesub] )
            testloglls.append(asd_data_set['testlogll'+thesub] )
     


# for trainlogll, testlogll, thesub in zip(trainloglls,testloglls,subnames):
#     with initiate_plot(2,2,300) as fig:
#         ax=fig.add_subplot(111)
#         ax.hist(trainlogll, color='b', label='training data', bins=30, density=True)
#         ax.hist(testlogll,  color='r', label='testing data', bins=30, density=True)
#         quickspine(ax)
#         ax.set_xlabel('– log likelihood')
#         ax.set_ylabel('probability')
#         ax.set_title(thesub)
#         quickleg(ax)

        # quicksave('eval logll hist {}'.format(thesub))

# # style 1, all together
# with initiate_plot(2,2,300) as fig:
#     ax=fig.add_subplot(111)
#     for trainlogll, testlogll, thesub in zip(trainloglls,testloglls,subnames):
#         ax.scatter(np.zeros_like(trainlogll),trainlogll, color='b', label='each subject log likelihood')
#         ax.scatter(np.ones_like(testlogll),testlogll, color='r', label='each subject log likelihood ')
#         lines=np.vstack([sorted(trainlogll),sorted(testlogll)])
#         ax.plot(lines, color='yellow', alpha=0.2)
#     quickspine(ax)
#     ax.set_ylabel('– log likelihood')
#     ax.set_xticks([0,1])
#     ax.set_xticklabels(['traning', 'testing'])
#     quickleg(ax)


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

# style 4, train vs test together
with initiate_plot(2,2,300) as fig:
    ax=fig.add_subplot(111)
    for trainlogll, testlogll, thesub in zip(trainloglls,testloglls,subnames):
        ax.scatter(trainlogll,testlogll,s=1, alpha=1)
    ax.plot([0,20],[0,20], color='k', linewidth=1)
    quickspine(ax)
    ax.set_xlabel('training')
    ax.set_ylabel('testing')


# style 4, train vs test per sub
for trainlogll, testlogll, thesub in zip(trainloglls,testloglls,subnames):
    with initiate_plot(2,2,300) as fig:
        ax=fig.add_subplot(111)
        # for trainlogll, testlogll, thesub in zip(trainloglls,testloglls,subnames):
        ax.scatter(trainlogll,testlogll,s=1, alpha=1)
        ax.plot([0,20],[0,20], color='k', linewidth=1)
        quickspine(ax)
        ax.set_xlabel('training')
        ax.set_ylabel('testing')
        ax.set_title(thesub)





# model performance vs actual data ---------------------------
env=ffacc_real.FireFlyPaper(arg)
env.episode_len=50
env.debug=1
env.terminal_vel=0.05
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

datapath=Path("/data/human/agroup")
with open(datapath, 'rb') as f:
    astate,_, tasks = pickle.load(f)

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
theta_asd=thetas[0]
theta_nt=thetas[1]

response_asd,_=run_trials_multitask(agent, env, phi, theta_asd, tasks,ntrials=1, stimdur=None)
response_nt,_=run_trials_multitask(agent, env, phi, theta_asd, tasks,ntrials=1, stimdur=None)

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

states=response_asd
endpoint=np.array([s[-1,:2].tolist() for s in states])
r,a=xy2pol(endpoint.T, rotation=False)
endpointpolar=np.stack([a,r]).T
r,a=xy2pol(tasks.T, rotation=False)
taskspolar=np.stack([a,r]).T
asd_model_endpoint=(endpoint, tasks)
asd_model_endpoint_polar=(endpointpolar, taskspolar)
states=response_nt
endpoint=np.array([s[-1,:2].tolist() for s in states])
r,a=xy2pol(endpoint.T, rotation=False)
endpointpolar=np.stack([a,r]).T
r,a=xy2pol(tasks.T, rotation=False)
taskspolar=np.stack([a,r]).T
nt_model_endpoint=(endpoint, tasks)
nt_model_endpoint_polar=(endpointpolar, taskspolar)


# angular err of data
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

quickoverhead_state(response_asd,tasks)

quickoverhead_state(astate,tasks)



# angular err of model compared to data
with initiate_plot(6,3,300) as f:
    ax1=f.add_subplot(121)
    ax2=f.add_subplot(122, sharex=ax1, sharey=ax1)
    color='r'
    (endpoint, _)=asd_model_endpoint_polar
    ax1.scatter(taskspolar[:,0],endpoint[:,0],color=color, s=0.5, label='ASD model')
    color='b'
    (endpoint, _)=nt_model_endpoint_polar
    ax2.scatter(taskspolar[:,0],endpoint[:,0],color=color, s=0.5, label='NT model')

    for thesub in asd_data_endpoint_polar.keys():
        if thesub[0]=='a': 
            color='pink'
            (endpoint, tasks)=asd_data_endpoint_polar[thesub]
            ax1.scatter(tasks[:,0],endpoint[:,0],color=color, s=0.5, alpha=0.5, label='ASD data')
        else: 
            color='tab:blue'
            (endpoint, tasks)=asd_data_endpoint_polar[thesub]
            ax2.scatter(tasks[:,0],endpoint[:,0],color=color, s=0.5, alpha=0.5, label='NT data')
    ax1.set_xlim(-0.7,0.7)
    ax1.set_ylim(-1,1)
    quickspine(ax1)
    quickspine(ax2)
    ax1.set_xlabel('target angle')
    ax2.set_xlabel('target angle')
    ax1.set_ylabel('response angle')
    ax1.plot([-0.7,.7],[-0.7,.7],'k')
    ax2.plot([-0.7,.7],[-0.7,.7],'w')
    quickleg(ax1); quickleg(ax2)
    # quicksave('model vs data asd angular err sep')

with initiate_plot(6,3,300) as f:
    ax1=f.add_subplot(121)
    ax2=f.add_subplot(122, sharex=ax1, sharey=ax1)
    color='r'
    (endpoint, _)=asd_model_endpoint_polar
    ax1.scatter(taskspolar[:,1],endpoint[:,1],color=color, s=0.5,alpha=0.5, label='ASD model')
    color='b'
    (endpoint, _)=nt_model_endpoint_polar
    ax2.scatter(taskspolar[:,1],endpoint[:,1],color=color, s=0.5,alpha=0.5, label='NT model')
    for thesub in asd_data_endpoint_polar.keys():
        if thesub[0]=='a': 
            color='pink'
            (endpoint, tasks)=asd_data_endpoint_polar[thesub]
            ax1.scatter(tasks[:,1],endpoint[:,1],color=color, s=0.5, alpha=0.5, label='ASD data')
        else: 
            color='tab:blue'
            (endpoint, tasks)=asd_data_endpoint_polar[thesub]
            ax2.scatter(tasks[:,1],endpoint[:,1],color=color, s=0.5, alpha=0.5, label='NT data')
    ax1.set_xlim(0.5,None)
    ax1.set_ylim(0,None)
    quickspine(ax1)
    quickspine(ax2)
    ax1.set_xlabel('target distance')
    ax2.set_xlabel('target distance')
    ax1.set_ylabel('response distance')
    ax1.plot([-0.7,3],[-0.7,3],'k')
    ax2.plot([-0.7,3],[-0.7,3],'w')
    quickleg(ax1); quickleg(ax2)
    # quicksave('model vs data asd radiual err sep')




import notification
notification.notify()



# ASD questionaire data (SCQ)----------------------------------------------
# fit to the most seperatble axis from svm

scqdf=pd.read_csv('/data/human/Demosgraphics.csv')

asdscqdata={}

for invtag in ['H','A']:
    nsub=14 if invtag=='A' else 25
    for i in range(nsub):
        sub="{}{}".format(invtag,str(i+1))
        asdscqdata["{}sub{}".format(invtag.lower(),str(i))]=int(scqdf[scqdf.Acronym==sub].SCQ)

with initiate_plot(2,2,300) as fig:
    ax=fig.add_subplot(111)
    for invtag in ['a','h']:
        c='r' if invtag=='a' else 'b'
        nsub=14 if invtag=='a' else 25
        for i in range(nsub):
            sub="{}sub{}".format(invtag,str(i))
            if 'res'+sub in asd_data_set:
                ax.scatter(asd_data_set['res'+sub][0][0],asdscqdata[sub],color=c)



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

numsamples=100
adjustratio=len(invres['h'])/len(invres['a'])
alltag=[]
allsamples=[]
for theta,cov,_ in invres['a']:
    distribution=MultivariateNormal(theta.view(-1),cov*0.01)
    samples=[]
    while len(samples)<int(numsamples*adjustratio):
        a=distribution.sample()
        if torch.all(a.clamp(0,2)==a):
            samples.append(a)
    allsamples.append(torch.stack(samples))
    alltag+=[1]*int(numsamples*adjustratio)
for theta,cov,_ in invres['h']:
    distribution=MultivariateNormal(theta.view(-1),cov*0.01)
    samples=[]
    while len(samples)<numsamples:
        a=distribution.sample()
        if torch.all(a.clamp(0,2)==a):
            samples.append(a)
    allsamples.append(torch.stack(samples))
    alltag+=[0]*numsamples

allsamples=np.array(torch.cat(allsamples,axis=0))
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
f_importances(np.abs(clf.coef_[0]),theta_names)

print('''
project the individual thetas on to the normal vector.
''')
w=clf.coef_[0]
ticks=X[:,:8].dot(w)
fig = plt.figure()
ax  = fig.add_subplot(111)
ax.hist(ticks[Y==0],density=True,color='b',bins=22,label='health control',alpha=0.6)
ax.hist(ticks[Y==1],density=True,color='r',bins=22,label='ASD',alpha=0.6)
ax.set_xlabel('param value')
ax.set_ylabel('probability')
quickspine(ax)


# scq vs param projection on svm axis
scq=[]
for i in range(14):
    sub='asub{}'.format(i)
    scq.append(asdscqdata[sub])
for i in range(23):
    sub='hsub{}'.format(i)
    scq.append(asdscqdata[sub])
projectedparam=np.hstack([np.array(torch.stack([log[0] for log in invres['a']])[:,:8,0]).dot(w),np.array(torch.stack([log[0] for log in invres['h']])[:,:8,0]).dot(w)])

datarange=max(scq)-min(scq)
paramrange=max(projectedparam)-min(projectedparam)
scaler=datarange/paramrange
projectedparam=projectedparam*scaler
projectedparam=projectedparam-np.mean(projectedparam)

regr = linear_model.LinearRegression()
regr.fit(np.array(scq).reshape(-1,1), projectedparam.reshape(-1,1))

with initiate_plot(3,3,300) as fig:
    ax=fig.add_subplot(111)
    ax.scatter(scq[:14], projectedparam[:14],color='r',s=2)
    ax.scatter(scq[14:], projectedparam[14:],color='b',s=2)
    datamin, datamax= min(min(scq),min(projectedparam)),max(max(scq),max(projectedparam))
    datarange=datamax-datamin
    datamin, datamax=datamin-datarange*0.1, datamax+datarange*0.1
    ax.set_xlim(datamin, datamax)
    ax.set_ylim(datamin, datamax)
    ax.plot([datamin, datamax], np.array([datamin, datamax])*regr.coef_[0,0]+regr.intercept_[0],'k')
    # ax.axis('equal')
    quickspine(ax)
    ax.set_xlabel('SCQ')
    ax.set_ylabel('normalized projected param')
    ax.set_title("{} \n corr={:.2f}".format(paramname, regr.coef_[0,0]))


# scq vs param axis
with initiate_plot(8,6,300) as fig:
    for i, paramname in enumerate(theta_names):
        
        projectedparam=np.hstack([
            np.array([log[0][i].item() for log in invres['a']]),
            np.array([log[0][i].item() for log in invres['h']])
        ])

        datarange=max(scq)-min(scq)
        paramrange=max(projectedparam)-min(projectedparam)
        scaler=datarange/paramrange
        projectedparam=projectedparam*scaler
        projectedparam=projectedparam-np.mean(projectedparam)

        regr = linear_model.LinearRegression()
        regr.fit(np.array(scq).reshape(-1,1), projectedparam.reshape(-1,1))

        ax=fig.add_subplot(3,4,i+1)
        ax.scatter(scq[:14], projectedparam[:14],color='r',s=2)
        ax.scatter(scq[14:], projectedparam[14:],color='b',s=2)
        datamin, datamax= min(min(scq),min(projectedparam)),max(max(scq),max(projectedparam))
        datarange=datamax-datamin
        datamin, datamax=datamin-datarange*0.1, datamax+datarange*0.1
        ax.set_xlim(datamin, datamax)
        ax.set_ylim(datamin, datamax)
        ax.plot([datamin, datamax], np.array([datamin, datamax])*regr.coef_[0,0]+regr.intercept_[0],'k')
        ax.axis('equal')
        quickspine(ax)
        ax.set_xlabel('SCQ')
        ax.set_ylabel(paramname)
        ax.set_title("{} \n corr={:.2f}".format(paramname, regr.coef_[0,0]))

    plt.subplots_adjust(top=0.85)
    fig.tight_layout(pad=.6)
    fig
    # plt.tight_layout()


# show the eigvector heatmap ----------------------


# load data 
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
            asd_data_set['res'+thesub]=process_inv(fullinverseres, usingbest=True, removegr=True)
        # load data
        if Path('/data/human/{}'.format(thesub)).is_file():
            with open('/data/human/{}'.format(thesub), 'rb') as f:
                states, actions, tasks = pickle.load(f)
            print(len(states))
            asd_data_set['data'+thesub]=states, actions, tasks
        
for invtag in ['h','a']:
    for isub in range(numhsub):
        thesub="{}sub{}".format(invtag,str(isub))
        if 'res'+thesub in asd_data_set:
            theta,cov,err=asd_data_set['res'+thesub]

            ev, evector=torch.eig(torch.tensor(cov),eigenvectors=True)
            ev=ev[:,0]
            ev,esortinds=ev.sort(descending=False)
            evector=evector[:,esortinds]
            inds=[1, 3, 5, 7, 0, 2, 4,6, 8, 9]
            with initiate_plot(5,5,300) as fig:
                ax=fig.add_subplot(1,1,1)
                img=ax.imshow(evector[inds],cmap=plt.get_cmap('bwr'),
                        vmin=-torch.max(evector),vmax=torch.max(evector),origin='upper')
                c=plt.colorbar(img,fraction=0.046, pad=0.04)
                c.set_label('parameter weight')
                ax.set_title('eigen vectors of covariance matrix')
                x_pos = np.arange(len(theta_names))
                plt.yticks(x_pos, [theta_names[i] for i in inds],ha='right')
                ax.set_xticks([])
                # quicksave('eigvector heatmap bruno pert')

            with initiate_plot(5,1,300) as fig:
                ax=fig.add_subplot(1,1,1)
                x_pos = np.arange(len(theta_names))
                # Create bars and choose color
                ax.bar(x_pos, torch.sqrt(ev), color = color_settings['hidden'])
                for i, v in enumerate(torch.sqrt(ev)):
                    number='{0:.2f}'.format(v.item())[1:] if v.item()<1 else '{0:.2f}'.format(v.item())
                    ax.text( i-0.4,v+0.2 , number, color=color_settings['hidden'])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.set_xticks([])
                ax.set_xlabel('sqrt of eigen values')
                ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
                



        
for invtag in ['h','a']:
    for isub in range(numhsub):
        thesub="{}sub{}".format(invtag,str(isub))
        if 'res'+thesub in asd_data_set:
            theta,cov,err=asd_data_set['res'+thesub]
            correlation=correlation_from_covariance(cov)
            inds=[1, 3, 5, 7, 0, 2, 4,6, 8, 9]



            with initiate_plot(4,4,300) as fig:
                ax=fig.add_subplot(1,1,1)
                im=ax.imshow(correlation[inds].t()[inds].t(),cmap=plt.get_cmap('bwr'),
                    vmin=-torch.max(correlation),vmax=torch.max(correlation))
                ax.set_title('correlation matrix', fontsize=16)
                c=plt.colorbar(im,fraction=0.046, pad=0.04,ticks=[-1, 0, 1])
                c.set_label('correlation')
                x_pos = np.arange(len(theta_names))
                plt.yticks(x_pos, [theta_names[i] for i in inds],ha='right')
                plt.xticks(x_pos, [theta_names[i] for i in inds],rotation=45,ha='right')

invtag='h'
isub=0
for invtag in ['h','a']:
    for isub in range(numhsub):
        thesub="{}sub{}".format(invtag,str(isub))
        if 'res'+thesub in asd_data_set:
            theta,cov,err=asd_data_set['res'+thesub]
            correlation=correlation_from_covariance(cov)
            inds=[1, 3, 5, 7, 0, 2, 4,6, 8, 9]

            with initiate_plot(4,4,300) as fig:
                ax=fig.add_subplot(1,1,1)
                cov=torch.tensor(cov)
                im=plt.imshow(cov[inds].t()[inds].t(),cmap=plt.get_cmap('bwr'),vmin=-torch.max(cov),vmax=torch.max(cov))
                ax.set_title('covariance matrix', fontsize=20)
                c=plt.colorbar(im,fraction=0.046, pad=0.04)
                c.set_label('covariance')
                x_pos = np.arange(len(theta_names))
                plt.yticks(x_pos, [theta_names[i] for i in inds],ha='right')
                plt.xticks(x_pos, [theta_names[i] for i in inds],rotation=45,ha='right')
                # quicksave('{} cov'.format(thesub))

# small ellipse
with initiate_plot(1,1,300) as fig:
    ax=fig.add_subplot()
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('obs noise') # correslonding to 1,1 in cov
    ax.set_ylabel('action cost') # correslonding to 0,0 in cov
    quickspine(ax)
    plot_cov_ellipse(cov[2:4,2:4],ax=ax, color='black')
    # quicksave('action cost vs noise cov')



