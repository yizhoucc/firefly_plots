import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import timeit
import random
import datetime
from plot_ult import *
from pathlib import Path

class Parameters:

    def __init__(self, data_dict):
        for k, v in data_dict.items():
            exec("self.%s=%s" % (k, v))

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, x):
        'Initialization'
        self.list_IDs = x

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x = self.list_IDs[index]
        return x


from scipy.ndimage import gaussian_filter


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


def load_data(X):

    data_loaders = {}
    data_sizes = {}
    totaln=len(X)
    inds=torch.randperm(totaln)[:totaln//5]
   
    for name in ['train', 'val']:
        if name=='val':
            data_set=Dataset(X)
        else:
            mask=torch.ones_like(Y)
            mask[inds]=0
            data_set=Dataset(X[mask])
        data_set=Dataset(X)
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


def normalize(x):
    if (np.max(x)-np.min(x))==0:
        return x
    return (x-np.min(x))/(np.max(x)-np.min(x))

forcemaxlen=50
# load data ----------------------------
asd_data_set={}
numhsub,numasub=25,14
fulltrainfolder='persub1cont'
parttrainfolder='persub3of5dp'
for invtag in ['h','a']:
    for isub in range(numhsub):
        thesub="{}sub{}".format(invtag,str(isub))
        evalname=Path(datapath/"human/{}/evaltrain_inv{}sub{}".format(parttrainfolder,invtag,str(isub)))
        fullinverseres=Path(datapath/"human/{}".format(fulltrainfolder))/"inv{}sub{}".format(invtag,str(isub))
        partinverseres=Path(datapath/"human/{}".format(parttrainfolder))/"inv{}sub{}".format(invtag,str(isub))
        # load inv res
        if fullinverseres.is_file():
            asd_data_set['res'+thesub]=process_inv(fullinverseres, usingbest=True, removegr=False)
        # load data
        if Path(datapath/'human/{}'.format(thesub)).is_file():
            with open(datapath/'human/{}'.format(thesub), 'rb') as f:
                states, actions, tasks = pickle.load(f)
            print(len(states))
            asd_data_set['data'+thesub]=states, actions, tasks
        
alltag=[]
X=[]
allsamples_=[]
seq_lengths=[]
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
maxlen=min(maxlen, forcemaxlen)       
for invtag in ['h','a']:
    for isub in range(numhsub):
        thesub="{}sub{}".format(invtag,str(isub))
        if 'data'+thesub in asd_data_set:
            _,actions,tasks=asd_data_set['data'+thesub]
            seq_lengths+=[len(a) for a in actions]
            paddedv=pad_to_dense([normalize(gaussian_filter(np.array(a[:,0]),2)) for a in actions],maxlen=maxlen)
            paddedw=pad_to_dense([normalize(gaussian_filter(np.array(a[:,1]),2)*sign) for sign, a in zip(np.array(tasks[:,1]>0, dtype=int)*-1, actions)],maxlen=maxlen)
            # paddedv=[normalize(gaussian_filter(np.array(a[:,0]),2)) for a in actions]
            # paddedw=[normalize(gaussian_filter(np.array(a[:,1]),2)*sign) for sign, a in zip(np.array(tasks[:,1]>0, dtype=int)*-1, actions)]
            # X+=[np.moveaxis(np.array([v,w]), [0,1],[1,0]) for v,w in zip(paddedv, paddedw)]
            allsamples_.append(np.moveaxis(np.array([paddedv,paddedw]),[0,2],[2,1])) # remove the task, similar res

            if invtag=='a':
                alltag+=[1]*len(actions)
            else:
                alltag+=[0]*len(actions)

allsamples=np.vstack(allsamples_)
alltag=np.array(alltag).astype('int') # asd is 1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
X, Y=torch.tensor(allsamples).float(),torch.tensor(alltag)



class AE(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.enc = nn.LSTM(input_dim, latent_dim) 
        self.dec = nn.LSTM(latent_dim, latent_dim)
        self.out = nn.Linear(latent_dim, input_dim)
  
    def forward(self, x_batch):
        x_batch.shape # [10, 203, 2]
        x_batch=torch.permute(x_batch, (1,0,2))
        x_batch.shape # [203, 10, 2]
        _, (prev_hidden, _) = self.enc(x_batch)
        prev_hidden.shape # [1, 10, 16]
        encoded = prev_hidden.repeat([x_batch.shape[0],1,1])
        encoded.shape # [203, 10, 16] 
        out, _ = self.dec(encoded)
        # out, _ = self.dec(prev_hidden)
        out.shape # [203, 10, 16]
        res=self.out(out)
       
        res.shape # [203, 10, 2]
        res=torch.permute(res, (1,0,2)) # [10, 203, 2]

        return res
    
# x_batch=X[:10]
# self=ae

from torch.autograd import Variable

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.isCuda = isCuda
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()
        
        #initialize weights
        nn.init.xavier_uniform(self.lstm.weight_ih_l0, gain=np.sqrt(3))
        nn.init.xavier_uniform(self.lstm.weight_hh_l0, gain=np.sqrt(3))

    def forward(self, input):
        tt = torch.cuda if self.isCuda else torch
        h0 = Variable(tt.FloatTensor(self.num_layers, input.size(0), self.hidden_size))
        c0 = Variable(tt.FloatTensor(self.num_layers, input.size(0), self.hidden_size))
        out, (hidden_n, _) = self.lstm(input, (h0, c0))
        out, (hidden_n, _) = self.lstm2(out, (h0, c0))
        encoded_input = self.relu(hidden_n)
        return torch.permute(encoded_input,(1,0,2))

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, isCuda,seq_len=forcemaxlen):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.isCuda = isCuda
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(output_size, output_size, num_layers, batch_first=True)
        #self.relu = nn.ReLU()
        self.act = nn.Tanh()
        self.seq_len=seq_len
        
        #initialize weights
        nn.init.xavier_uniform(self.lstm.weight_ih_l0, gain=np.sqrt(3))
        nn.init.xavier_uniform(self.lstm.weight_hh_l0, gain=np.sqrt(3))
        
    def forward(self, encoded_input):
        tt = torch.cuda if self.isCuda else torch
        h0 = Variable(tt.FloatTensor(self.num_layers, encoded_input.size(0), self.output_size))
        c0 = Variable(tt.FloatTensor(self.num_layers, encoded_input.size(0), self.output_size))
        decoded_output, hidden = self.lstm(encoded_input, (h0, c0))
        decoded_output = self.act(decoded_output)
        decoded_output=decoded_output*1.5 # allow slightly larger output range
        return decoded_output

    def forward(self, encoded_input):
        tt = torch.cuda if self.isCuda else torch
        h0 = Variable(tt.FloatTensor(self.num_layers, encoded_input.size(0), self.output_size))
        c0 = Variable(tt.FloatTensor(self.num_layers, encoded_input.size(0), self.output_size))

        # lstm(hidden_size, output_size, num_layers, batch_first=True)
        h0=c0=Variable(torch.FloatTensor(1, encoded_input.size(0), 2))
        # a=nn.LSTM(32, 2, 1, batch_first=True)

        decoded_output=Variable(tt.FloatTensor(encoded_input.size(0), encoded_input.size(1), self.output_size))
        decoded_output.size() # [5662, 60, 2])

        # Process input sequence step by step
        for t in range(encoded_input.size(1)):
            if t == 0:
                # First input step
                out_t, (h_t, c_t) = self.lstm(encoded_input[:,-1:,:], (h0, c0))
                
                # torch.zeros(encoded_input.size(0),1,encoded_input.size(2)).shape
                # out_t, (h_t, c_t) = a(encoded_input[:,-1:,:], (h0, c0))
                
            else:
                # Subsequent input steps
                out_t, (h_t, c_t) = self.lstm(torch.zeros(encoded_input.size(0),1,encoded_input.size(2)), (h_t, c_t))
                
            # Save output of each input step
            # decoded_output[:, encoded_input.size(1)-1-t, :] = self.act(out_t[:, 0, :])
            decoded_output[:, t, :] = self.act(out_t[:, 0, :])
        decoded_output=decoded_output*1.5
        
        return decoded_output
    
    def forward(self, x):
        x = x.repeat(self.seq_len, x.size(2))
        x = x.reshape((self.n_features, self.seq_len, self.input_dim))
        x, (hidden_n, cell_n) = self.lstm(x,(h0, c0))
        x = x.reshape((self.seq_len, self.hidden_dim))


        seq_len=60
        n_features=1
        input_dim=32
        out_dim=2
        batchsize=11
    
        x=torch.randn(batchsize,1,input_dim) 
        x.shape
        x = x.repeat(1,seq_len,1)
        x.shape
        x = x.reshape((batchsize, seq_len, input_dim))
        x.shape

        a=nn.LSTM(input_size=input_dim,
                hidden_size=2,
                num_layers=1,
                batch_first=True)
        x, (hidden_n, cell_n) = a(x)
        x.shape

        x = x.reshape((batchsize,seq_len, out_dim))

    def forward(self, x):

        x = x.repeat(1,self.seq_len,1)

        tt = torch.cuda if self.isCuda else torch
        h0 = Variable(tt.FloatTensor(self.num_layers, x.size(0), self.output_size))
        c0 = Variable(tt.FloatTensor(self.num_layers, x.size(0), self.output_size))

        # x, (hidden_n, cell_n) = self.lstm(x,(h0,c0))
        x, (hidden_n, cell_n) = self.lstm(x)
        x, (hidden_n, cell_n) = self.lstm2(x)
        decoded_output = self.act(x)
        return decoded_output*1.5


class LSTMAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda):
        super(LSTMAE, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, isCuda)
        self.decoder = DecoderRNN(hidden_size, input_size, num_layers, isCuda)
        
    def forward(self, input):
        encoded_input = self.encoder(input)
        decoded_output = self.decoder(encoded_input)
        return decoded_output
    




nsample=len(X)
# X.shape # [5662, 203, 2])

# ae=AE(2,16) # feature, hidden

model=LSTMAE(input_size=2, hidden_size=32, num_layers=1, isCuda=False)

optimizer=torch.optim.Adam(model.parameters(), lr=0.001)
criterion=nn.L1Loss(reduction='sum').to(device)

# training
modelname='trajectory embedding rae'
note='0416_h32'
for _ in range(40):
    pred = model(X)
    print('num of nan: ', torch.sum(torch.isnan(pred)))
    if torch.any(torch.isnan(pred)):
        break
    optimizer.zero_grad()
    loss = criterion(pred, X)
    loss.backward(retain_graph=True)
    # print([a.grad for a in model.parameters()])
    max_norm = 0.3 # Example value for maximum norm
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    optimizer.step()
    
    ind=torch.randint(low=0, high=nsample, size=(1,))
    ts=np.linspace(0,maxlen/10, maxlen)
    plt.plot(ts,X[ind].clone().detach()[0], color='k', label='actual data')
    plt.plot(ts,pred[ind].clone().detach()[0], color='r', label='LSTM AE reconstructed')
    plt.xlabel('time [s]')
    plt.ylabel('control [a.u.]')
    plt.ylim(-1.1,1.1)
    quickleg(ax=plt.gca(), bbox_to_anchor=(-0.2,0))
    quickspine(ax=plt.gca())
    plt.title('loss={:.4f}'.format(loss.clone().detach()))
    plt.show()
    
    if not torch.isnan(loss.clone().detach()):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_fn': criterion
        }, datapath/'res/{}_{}.pt'.format(modelname, note))
            
def vis(true, pred, ind=None):
    if ind is None:
        ind=torch.randint(low=0, high=len(true), size=(1,))
    ts=np.linspace(0,maxlen/10, maxlen)
    plt.plot(ts,true.clone().detach()[0], color='k', label='actual data')
    plt.plot(ts,pred.clone().detach()[0], color='r', label='LSTM AE reconstructed')
    plt.xlabel('time [s]')
    plt.ylabel('control [a.u.]')
    plt.ylim(-1.1,1.1)
    quickleg(ax=plt.gca(), bbox_to_anchor=(-0.2,0))
    quickspine(ax=plt.gca())
    plt.title('loss={:.4f}'.format(loss.clone().detach()))
    plt.show()

done=False
if done:
    pt=torch.load(datapath/'res/{}_{}.pt'.format(modelname, note))
    model=LSTMAE(input_size=2, hidden_size=64, num_layers=1, isCuda=False)
    model.load_state_dict(pt['model_state_dict'])


    with torch.no_grad():
        embed = model.encoder(X)
        embed.shape
        encodedX=embed[:,-1,:]
        Y.shape

    # run svm 

    from sklearn import svm


    svmmodel = svm.SVC(kernel='linear')
    clf = svmmodel.fit(encodedX, Y)


    print('''
    project the individual thetas on to the normal vector.
    ''')
    w=clf.coef_[0]
    ticks=encodedX@w
    with initiate_plot(3,3,300) as fig:
        ax  = fig.add_subplot(111)
        ax.hist(ticks[Y==0],density=True,color='b',bins=99,label='health control',alpha=0.6)
        ax.hist(ticks[Y==1],density=True,color='r',bins=99,label='ASD',alpha=0.6)
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
        



# test
def vis(true, pred,loss=0, ind=None):
    if ind is None:
        ind=torch.randint(low=0, high=len(true), size=(1,))
    maxlen=len(true[0])
    ts=np.linspace(0,maxlen/10, maxlen)
    plt.plot(ts,true[ind].clone().detach()[0], color='k', label='actual data')
    plt.plot(ts,pred[ind].clone().detach()[0], color='r', label='LSTM AE reconstructed')
    plt.xlabel('time [s]')
    plt.ylabel('control [a.u.]')
    plt.ylim(-0.1,1.1)
    quickleg(ax=plt.gca(), bbox_to_anchor=(-0.2,0))
    quickspine(ax=plt.gca())
    plt.title('loss={:.4f}'.format(loss))
    plt.show()

def forward(x,t=50):
    out=torch.zeros(x.size(0),t, x.size(1))
    out.shape
    # for tt in range(t):
    #     x=0.77*x+torch.rand(x.size())*0.002
    #     out[:,tt,:]=x
    # out[:,:,1]=out[:,:,1]-1 #+ torch.sin(torch.linspace(0,6,t))*0.3
    out[:,:,0]=out[:,:,0] + torch.sin(torch.linspace(0,6,t))*0.4

    return out

x=torch.rand(50,1)


# X=forward(x,t=forcemaxlen)-0.5
vis(X,X)


import copy

class Encoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )
  def forward(self, x):
    # x = x.reshape((1, self.seq_len, self.n_features))
    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)
    return torch.permute(hidden_n,(1,0,2))

class Decoder(nn.Module):
  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()
    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features
    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    self.output_layer = nn.Linear(self.hidden_dim, n_features)
  def forward(self, x):
    x = x.repeat(1,self.seq_len, 1)
    # x = x.reshape((self.n_features, self.seq_len, self.input_dim))
    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    # x = x.reshape((self.seq_len, self.hidden_dim))
    return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(RecurrentAutoencoder, self).__init__()
    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x


model = RecurrentAutoencoder(seq_len=forcemaxlen, n_features=2, embedding_dim=64)

train_dataset=X
val_dataset=X
n_epochs=2000
train_loss=0
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.L1Loss(reduction='sum').to(device)
history = dict(train=[], val=[])
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = np.inf
for epoch in range(1, n_epochs + 1):
    model = model.train()
    train_losses = []

    optimizer.zero_grad()
    seq_pred = model(train_dataset)
    loss = criterion(seq_pred, train_dataset)
    loss.backward()
    max_norm = 1 # Example value for maximum norm
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
    train_losses.append(loss.item())

    val_losses = []
    model = model.eval()
    with torch.no_grad():

        seq_pred = model(val_dataset)
        loss = criterion(seq_pred, val_dataset)
        val_losses.append(loss.item())
        vis(val_dataset,seq_pred,train_loss)

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    history['train'].append(train_loss)
    history['val'].append(val_loss)
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
model.load_state_dict(best_model_wts)

import pickle
with open(datapath/'res/lstmae_64.mw','wb+') as f:
    pickle.dump(best_model_wts,f)


# encode
with torch.no_grad():
    encodedX=model.encoder(X)

run_svm=True
if run_svm:
    from sklearn import svm
    svmmodel = svm.SVC(kernel='linear')
    clf = svmmodel.fit(encodedX[:,0,:], Y)


    print('''
    project the individual thetas on to the normal vector.
    ''')
    w=clf.coef_[0]
    ticks=(encodedX@w).view(-1)
    # ticks=normalize(ticks)
    with initiate_plot(3,3,300) as fig:
        ax  = fig.add_subplot(111)
        ax.hist(gaussian_filter(ticks[Y==0],5),density=True,color='b',bins=55,label='health control',alpha=0.6)
        ax.hist(gaussian_filter(ticks[Y==1],5),density=True,color='r',bins=55,label='ASD',alpha=0.6)
        ax.set_xlabel('param value')
        ax.set_ylabel('probability')
        quickspine(ax)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.spines['left'].set_visible(False)
        ax.set_title("autoencoder embedding")
        # quicksave('svm on lstm ae 64 trajectory')


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
    ticks=(encodedX[s:e,:]@w).view(-1)
    if s>=astartind: # ASD
        cm_subsection = np.linspace(0,1, 25) 
        colors = [ cm.gist_heat(x) for x in cm_subsection ]
        # ax2.hist(ticks,density=True,bins=33,alpha=0.6)
        density = stats.gaussian_kde(ticks)
        n, x = np.histogram(ticks, bins=33,density=True)
        ax2.plot(x, gaussian_filter(density(x),1),linewidth=1, color=colors[i-numhsub])
    else: # NT
        cm_subsection = np.linspace(0, 0.8, 25) 
        colors = [ cm.winter(x) for x in cm_subsection ]
        # ax1.hist(ticks,density=True,bins=33,alpha=0.6)
        density = stats.gaussian_kde(ticks)
        n, x = np.histogram(ticks, bins=33,density=True)
        ax1.plot(x, gaussian_filter(density(x),1), linewidth=1, color=colors[i])

    ax1.set_xlabel('param value')
    ax1.set_ylabel('probability density')
    quickspine(ax1)
    ax2.set_xlabel('param value')
    quickspine(ax2)
quicksave('svm on lstm ae 64 trajectory persub')
print('the other peak suggest asd behavior. still they are mixed')



with torch.no_grad():
    pred = model(X)
    loss = criterion(X, pred)
ind=torch.randint(0,len(X),(1,))
vis(X,pred,loss, ind=ind)



# orginal test

def forward(x,t=50):
    out=torch.zeros(x.size(0),t, x.size(1))
    out.shape
    for tt in range(t):
        x=0.77*x+torch.rand(x.size())*0.3
        out[:,tt,:]=x
    # out[:,:,1]=out[:,:,1]-1 #+ torch.sin(torch.linspace(0,6,t))*0.3
    out[:,:,0]=out[:,:,0]-1 #- torch.sin(torch.linspace(0,3,t))*0.2

    return out

x=torch.rand(66,1)

forward(x).shape
X=forward(x)


class Encoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )
  def forward(self, x):
    x = x.reshape((1, self.seq_len, self.n_features))
    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)
    return hidden_n.reshape((self.n_features, self.embedding_dim))
  
class Decoder(nn.Module):
  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()
    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features
    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    self.output_layer = nn.Linear(self.hidden_dim, n_features)
  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))
    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((self.seq_len, self.hidden_dim))
    return self.output_layer(x)
  
class RecurrentAutoencoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(RecurrentAutoencoder, self).__init__()
    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x
  

model = RecurrentAutoencoder(seq_len=forcemaxlen, n_features=1, embedding_dim=64)

def train_model(model, train_dataset, val_dataset, n_epochs):
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.L1Loss(reduction='sum').to(device)
  history = dict(train=[], val=[])
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0
  for epoch in range(1, n_epochs + 1):
    model = model.train()
    train_losses = []
    for seq_true in train_dataset:
      seq_true=seq_true.view(1,forcemaxlen,1)
      optimizer.zero_grad()
      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)
      loss = criterion(seq_pred, seq_true)
      loss.backward()
      max_norm = 0.3 # Example value for maximum norm
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

      optimizer.step()
      train_losses.append(loss.item())

    val_losses = []
    model = model.eval()
    with torch.no_grad():
      for seq_true in val_dataset:
        seq_true = seq_true.to(device)
        seq_pred = model(seq_true)
        loss = criterion(seq_pred, seq_true)
        val_losses.append(loss.item())

      ind=torch.randint(0,len(val_dataset), size=(1,))
      seq_pred = model(val_dataset[ind])
      vis(val_dataset[ind],seq_pred.unsqueeze(0))

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    history['train'].append(train_loss)
    history['val'].append(val_loss)
    if val_loss < best_loss:
      best_loss = val_loss
      best_model_wts = copy.deepcopy(model.state_dict())
    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
  model.load_state_dict(best_model_wts)

  return model.eval(), history

model, history = train_model(
  model,
  X[:-100],
  X[-100:],
  n_epochs=150
)





