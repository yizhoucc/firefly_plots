# use svm with penalty (encourage smoothness)
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from plot_ult import *
import copy


class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=16
        )
        self.encoder_output_layer = nn.Linear(
            in_features=16, out_features=16
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=16, out_features=16
        )
        self.decoder_output_layer = nn.Linear(
            in_features=16, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed

    def encode(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        return code


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, x, y):
        'Initialization'
        self.labels = y
        self.list_IDs = x

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x = self.list_IDs[index]
        y = self.labels[index]
        return x, y


def hinge_loss(outputs, labels, model,reg=1.5, alpha=0.01):
    """
    折页损失计算
    :param outputs: 大小为(N, num_classes)
    :param labels: 大小为(N)
    :return: 损失值
    """
    num_labels = len(labels)
    corrects = outputs[range(num_labels), labels].unsqueeze(0).T

    # 最大间隔
    margin = 1.0
    margins = outputs - corrects + margin
    loss = torch.sum(torch.max(margins, 1)[0]) / len(labels)

    # # 正则化强度
    loss += alpha * torch.sum(model.weight ** 2)
    # weights smooth
    loss+=reg*torch.sum(torch.abs(torch.diff(torch.diff(model.weight, axis=0))))
    # loss-=torch.abs(model.weight[0,model.weight.shape[1]//2]-model.weight[0,model.weight.shape[1]//2-1]) # adjust for disjoint where v and w concat

    return loss


def load_data(X, Y):

    data_loaders = {}
    data_sizes = {}
    totaln=len(Y)
    inds=torch.randperm(totaln)[:totaln//5]
   
    for name in ['train', 'val']:
        if name=='val':
            data_set=Dataset(X, Y)
        else:
            mask=torch.ones_like(Y)
            mask[inds]=0
            data_set=Dataset(X[mask], Y[mask])
        data_set=Dataset(X, Y)
        data_loader = DataLoader(data_set, shuffle=True, batch_size=128, num_workers=8)
        data_loaders[name] = data_loader
        data_sizes[name] = len(data_set)
        
    return data_loaders, data_sizes


def pad_to_dense(M, maxlen=0, padfirst=False):
    """Appends the minimal required amount of zeroes at the end of each 
     array in the jagged array `M`, such that `M` looses its jagedness."""
    if maxlen==0:
        maxlen = max(len(r) for r in M)
    Z = np.zeros((len(M), maxlen))
    if padfirst:
        for enu, row in enumerate(M):
            Z[enu, maxlen-min(maxlen, len(row)):] += row[:min(maxlen, len(row))]

    else:
        for enu, row in enumerate(M):
            Z[enu, :min(maxlen, len(row))] += row[:min(maxlen, len(row))]
    return Z


if __name__ == '__main__':

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
            
    alltag=[]
    allsamples_=[]
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
                paddedw=pad_to_dense([np.array(a[:,1])*sign for sign, a in zip(np.array(tasks[:,1]>0, dtype=int)*-1, actions)],maxlen=maxlen)
                # firstpaddedv=pad_to_dense([np.array(a[:,0]) for a in actions], maxlen=maxlen, padfirst=True)
                # firstpaddedw=pad_to_dense([np.array(a[:,1])*sign for sign, a in zip(np.array(tasks[:,1]>0, dtype=int)*-1, actions)], maxlen=maxlen,padfirst=True)                # allsamples_.append(np.hstack([tasks,paddedv,paddedw])) # baseline
                allsamples_.append(np.hstack([paddedv,paddedw])) # remove the task, similar res
                # allsamples_.append(np.hstack([(paddedv.T/tasks[:,0]).T,(paddedw.T/np.abs(tasks[:,1])).T])) # devide the target distance and angle

                # warpv=np.array([resample(np.array(a[:,0]), maxlen) for a in actions])
                # warpw=np.array([resample(np.array(a[:,1])*sign, maxlen) for sign, a in zip(np.array(tasks[:,1]>0, dtype=int)*-1, actions)])
                # allsamples_.append(np.hstack([warpv, warpw])) # resample instead of padding (time warpping)
                # allsamples_.append(np.hstack([tasks, warpv, warpw])) # resample instead of padding (time warpping) and target
                # allsamples_.append(np.hstack([(warpv.T/tasks[:,0]).T, (warpw.T/np.abs(tasks[:,1])).T])) # resample instead of padding (time warpping) and divde by target

                if invtag=='a':
                    alltag+=[1]*len(actions)
                else:
                    alltag+=[0]*len(actions)

    allsamples=np.vstack(allsamples_)
    alltag=np.array(alltag).astype('int') # asd is 1


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    X, Y=torch.tensor(allsamples).float(),torch.tensor(alltag)
    # X.shape
    model = AE(input_shape=X.shape[1]).to(device)
    # model = nn.Linear(X.shape[1], 2).to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    lr_schduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    data_loaders, data_sizes = load_data(X, Y)


epochs=222
for epoch in range(epochs):
    loss = 0
    for batch_features, batch_label in data_loaders['train']:

        batch_features = batch_features.view(-1, X.shape[1]).to(device)
        
        outputs = model(batch_features)
        train_loss = criterion(outputs, batch_features)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        loss += train_loss.item()
    loss = loss / len(data_loaders['train'])
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
    # visualize the reconstruction
    vis_reconstruction=True
    if vis_reconstruction and epoch%20==0:
        with torch.no_grad():
            recX=model(X)
        with initiate_plot(5,5,300) as fig:    
            
            ind=torch.randint(low=0, high=len(X), size=(1,)).item()
            ax=fig.add_subplot(321)
            ax.plot(recX[ind].clone().detach()[:200] ,label='reconstructed')
            ax.plot(X[ind][:200], label='orignal trajectory')
            ax.set_xlabel('time')
            ax.set_ylabel('v')
            ax.set_xticks([])
            ax.set_ylim(None, 1)
            quickspine(ax)
            

            ax=fig.add_subplot(322)
            ax.plot(recX[ind].clone().detach()[200:],label='reconstructed')
            ax.plot(X[ind][200:], label='orignal trajectory')
            ax.set_ylabel('w')
            ax.set_xlabel('time')
            ax.set_ylim(None, 1)
            ax.set_xticks([])
            quickspine(ax)


            ind=torch.randint(low=0, high=len(X), size=(1,)).item()
            ax=fig.add_subplot(323)
            ax.plot(recX[ind].clone().detach()[:200] ,label='reconstructed')
            ax.plot(X[ind][:200], label='orignal trajectory')
            ax.set_xlabel('time')
            ax.set_ylabel('v')            
            ax.set_xticks([])
            ax.set_ylim(None, 1)
            quickspine(ax)


            ax=fig.add_subplot(324)
            ax.plot(recX[ind].clone().detach()[200:],label='reconstructed')
            ax.plot(X[ind][200:], label='orignal trajectory')
            ax.set_ylabel('w')
            ax.set_xlabel('time')
            ax.set_ylim(None, 1)
            ax.set_xticks([])
            quickspine(ax)


            ind=torch.randint(low=0, high=len(X), size=(1,)).item()
            ax=fig.add_subplot(325)
            ax.plot(recX[ind].clone().detach()[:200] ,label='reconstructed')
            ax.plot(X[ind][:200], label='orignal trajectory')
            ax.set_ylabel('v')
            ax.set_xlabel('time')
            ax.set_xticks([])
            ax.set_ylim(None, 1)
            quickspine(ax)


            ax=fig.add_subplot(326)
            ax.plot(recX[ind].clone().detach()[200:],label='reconstructed')
            ax.plot(X[ind][200:], label='orignal trajectory')
            ax.set_ylabel('w')
            ax.set_xlabel('time')
            ax.set_ylim(None, 1)
            ax.set_xticks([])
            quickspine(ax)

            quickleg(ax)

            
# save the model
torch.save(model, "/workspaces/ae_trajectoryhid16_0410.model")


model = torch.load("/workspaces/ae_trajectoryhid16_0410.model")
# model2 = AE(input_shape=X.shape[1]).to(device)
# model2.load_state_dict(model.state_dict())
# model2.encode(batch_features).shape




vis_weights=False
if vis_weights:
    print('''
    wegihts of the svm.
    ''')

    with initiate_plot(6,2,300) as f:
        ax=f.add_subplot(121)
        im=ax.imshow(model.encoder_hidden_layer.weight.clone().detach(),cmap='bone', vmin=torch.min(model.encoder_hidden_layer.weight.clone().detach()),vmax=torch.max(model.encoder_hidden_layer.weight.clone().detach()))
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='25%', pad=0.45)
        f.colorbar(im, cax=cax, orientation='horizontal')
        ax.set_xlabel('trajecotry')
        ax.set_ylabel('hidden')

        quickallspine(ax)
        cax.set_xticks(cax.get_xlim())
        cax.set_xticklabels(['min','max'])
    
    with initiate_plot(6,2,300) as f:
        ax=f.add_subplot(121)
        im=ax.imshow(model.encoder_output_layer.weight.clone().detach(),cmap='bone', vmin=torch.min(model.encoder_output_layer.weight.clone().detach()),vmax=torch.max(model.encoder_output_layer.weight.clone().detach()))
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.15)
        f.colorbar(im, cax=cax, orientation='vertical')
        ax.set_ylabel('encoded trajectory')
        ax.set_xlabel('hidden')

        quickallspine(ax)
        cax.set_xticks(cax.get_xlim())
        cax.set_xticklabels(['min','max'])
        



# run svm 
run_svm=False
if run_svm:
    from sklearn import svm
    # Y.shape
    # X.shape
    with torch.no_grad():
        encodedX=model.encode(X)
    svmmodel = svm.SVC(kernel='linear')
    clf = svmmodel.fit(encodedX, Y)


    print('''
    project the individual thetas on to the normal vector.
    ''')
    w=clf.coef_[0]
    ticks=encodedX@w
    with initiate_plot(3,3,300) as fig:
        ax  = fig.add_subplot(111)
        ax.hist(ticks[Y==0],density=True,color='b',bins=55,label='health control',alpha=0.6)
        ax.hist(ticks[Y==1],density=True,color='r',bins=55,label='ASD',alpha=0.6)
        ax.set_xlabel('param value')
        ax.set_ylabel('probability')
        quickspine(ax)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.spines['left'].set_visible(False)
        ax.set_title("autoencoder embedding")
        # quicksave('svm on ae trajectory')


    print('''
    wegihts of the svm.
    ''')
    with initiate_plot(2,2,300) as f:
        ax=f.add_subplot(111)
        ax.plot(w,color='k')
        quickspine(ax)
        ax.set_xlabel('embedded trajectory')
        ax.set_ylabel('feature coef')
        ax.set_xticks(ax.get_xlim())
        ax.set_xticklabels([])
    

# per subject curve
import scipy.stats as stats
w=clf.coef_[0]
fig = plt.figure(figsize=(6,3))
ax1  = fig.add_subplot(121)
ax2  = fig.add_subplot(122, sharey=ax1,sharex=ax1)
for i, (s,e) in enumerate(zip(cumsum[:-1], cumsum[1:])):
    ticks=encodedX[s:e,:]@w
    if s>=astartind: # ASD
        cm_subsection = np.linspace(0,1, 25) 
        colors = [ cm.gist_heat(x) for x in cm_subsection ]
        # ax2.hist(ticks,density=True,bins=33,alpha=0.6)
        density = stats.gaussian_kde(ticks)
        n, x = np.histogram(ticks, bins=33,density=True)
        ax2.plot(x, density(x),linewidth=1, color=colors[i-numhsub])
    else: # NT
        cm_subsection = np.linspace(0, 0.8, 25) 
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





'''
0226
the autoencoder uses a kind of common template to fit all trajectories, instead of amplifying the difference in trajectories, check why
'''

'''
0410
change the hid dim from 128 to 16, because we dont need that many of features for trajectories.
result: still wont achieve a good reconstruction. similar result as before.
'''
