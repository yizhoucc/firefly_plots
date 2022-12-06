
# turn off obs data ------------------------------------
for filename in Path('/data/human/turnoffdata').glob('inv*'):
    dataname=Path('/data/human/turnoffdata')/filename.name[3:]

with open(filename, 'rb') as f:
    theta,_,_=process_inv(filename,removegr=False)
with open(dataname, 'rb') as f:
    states, actions, tasks, stimdur = pickle.load(f)


theta[7:9]=theta[7:9]/10

env=ffacc_real.FireFlyPaper(arg)
env.episode_len=50
env.debug=1
env.terminal_vel=0.2
phi=torch.tensor([[2],
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

# run trial with model 

# full optic flow
cond=np.where(stimdur==50)[0]
substates, subactions, subtasks, = [states[i] for i in cond], [actions[i] for i in cond], tasks[cond]
ax=plotoverhead_simple(substates,subtasks,color='b',label='subject',ax=None)
modelstates,modelactions=run_trials_multitask(agent, env, phi, theta, subtasks, ntrials=1)
T=max([len(s) for s in states])
modelstates=[m[:T] for m in modelstates]
ax=plotoverhead_simple(modelstates,subtasks,color='r',label='model',ax=ax)
ax.get_figure()

# minimal opotic flow
cond=np.where(stimdur==5)[0]
substates, subactions, subtasks, = [states[i] for i in cond], [actions[i] for i in cond], tasks[cond]
ax=plotoverhead_simple(substates,subtasks,color='g',label='subject',ax=None)
modelstates,modelactions=run_trials_multitask(agent, env, phi, theta, subtasks, ntrials=1, stimdur=stimdur[cond])
T=max([len(s) for s in states])
modelstates=[m[:T] for m in modelstates]
ax=plotoverhead_simple(modelstates,subtasks,color='r',label='model',ax=ax)
ax.get_figure()

# quicksave('{} model vs data overhead'.format(thesub),fig=ax.get_figure())

# plot control curve 
fig=plotctrl_vs(actions, modelactions, color1='b', color2='r', label1='subject', label2='model', alpha=1)
# quicksave('{} model vs data control curve '.format(thesub),fig=fig)

