
# for res_asd figures
import numpy as np
from plot_ult import * 
from scipy import stats 
from sklearn import svm
import matplotlib
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


# ASD questionaire data (SCQ)----------------------------------------------
# fit to the most seperatble axis from svm

scqdf=pd.read_csv(datapath/'human/Demosgraphics.csv')

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
logs={'a':datapath/'human/fixragroup','h':datapath/'human/clusterpaperhgroup'}

invres={'a':[],'h':[]}
for isub in range(numhsub):
    dataname="hsub{}".format(str(isub))
    savename=Path(datapath/"human/{}".format(foldername))/"invhsub{}".format(str(isub))
    if savename.is_file():
        invres['h'].append(process_inv(savename,ind=31, usingbest=True))
for isub in range(numasub):
    dataname="asub{}".format(str(isub))
    savename=Path(datapath/"human/{}".format(foldername))/"invasub{}".format(str(isub))
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
