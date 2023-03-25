
# use svm with penalty (encourage smoothness)
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from plot_ult import *
import copy


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


def train_model(data_loaders, model, criterion, optimizer, lr_scheduler, num_epochs=25, device=None):
    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                # print(inputs.shape)
                # print(labels.shape)
                # inputs = inputs.reshape(-1)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print(outputs.shape)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels, model)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                lr_scheduler.step()

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / data_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model


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



# 0226, simulation test
if __name__ == '__main__':

    maxlen=200
    minlen=100
    allsamples_,alltag_=[],[]

    # make asd
    alltag_+=[1]*(maxlen-minlen)
    for i in range(minlen,maxlen):
        trialv=[1]*(i//2)
        trialv+=[0.5]*(i-len(trialv))
        trialv+=[0.]*(maxlen-len(trialv))
        allsamples_.append(trialv)


    # make healthy
    alltag_+=[0]*(maxlen-minlen)
    for i in range(minlen,maxlen):
        trialv=[1]*(2*i//3)
        trialv+=[0.5]*(i-len(trialv))
        trialv+=[0.]*(maxlen-len(trialv))
        allsamples_.append(trialv)

    allsamples=np.vstack(allsamples_)
    alltag=np.array(alltag_).astype('int') # asd is 1


    # my svm, encourage smoother weights -----------------------
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    X, Y=torch.tensor(allsamples).float(),torch.tensor(alltag)
    # X.shape
    model = nn.Linear(X.shape[1], 2).to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = hinge_loss
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    lr_schduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    data_loaders, data_sizes = load_data(X, Y)


    epochs=10
    for e in range(epochs):
        train_model(data_loaders, model, criterion, optimizer, lr_schduler, num_epochs=3, device=device)
        diffs=(model.weight.clone().detach()[1]-model.weight.clone().detach()[0])[2:]


        # projected on most seperable axis svm ----------------------
        w=(model.weight.clone().detach()[1]-model.weight.clone().detach()[0])
        ticks=X[:,:]@(w)
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.hist(ticks[Y==0],density=True,color='b',bins=99,label='health control',alpha=0.6)
        ax.hist(ticks[Y==1],density=True,color='r',bins=99,label='ASD',alpha=0.6)
        ax.set_xlabel('param value')
        ax.set_ylabel('probability')
        # ax.set_xlim(-1,1)
        quickspine(ax)
        # quicksave('asd behavior trajectory svm project smooth')


        # weights of the svm ----------------------------
        maxlen=len(diffs)//4
        with initiate_plot(2,2,111) as f:
            ax=f.add_subplot(111)
            ax.plot(diffs[:maxlen], color='k')
            quickspine(ax)
            ax.set_xlabel('zero pad end')
            ax.set_ylabel('v control coef')
            ax.set_xticks(ax.get_xlim())
            ax.set_xticklabels(['start' ,'end'])

            # quicksave('v and w coef of trajectory svm smooth')




check=False
if check:
    # baseline svm (no smoothing) -------------------------------
    from sklearn import svm
    X, Y=allsamples,alltag
    X = X[np.logical_or(Y==0,Y==1)][:,:]
    Y = Y[np.logical_or(Y==0,Y==1)]
    baselinemodel = svm.SVC(kernel='linear')
    clf = baselinemodel.fit(X, Y)
    # f_importances(np.abs(clf.coef_[0][2:]),list(range(9)))
    # quicksave('svm trajectory weights')

    with initiate_plot(2,2,111) as f:
        vwcoef=(clf.coef_[0][2:])
        maxcoef=max(vwcoef)
        normvwcoef=vwcoef/maxcoef
        ax=f.add_subplot(111)
        ax.plot(np.linspace(0,len(normvwcoef[:maxlen])/10,len(normvwcoef[:maxlen]) ),(normvwcoef[:maxlen]), color='k')
        quickspine(ax)
        ax.set_xlabel('time s')
        ax.set_ylabel('v control coef')
        # ax.set_ylim(0,1)
        ax.set_xlim(-1,15)
        # quicksave('v and w coef of trajectory svm')


    print('''
    project the individual trajectory on to the normal vector.
    ''')
    # svm and curve together
    w=clf.coef_[0]
    ticks=X[:,:].dot(w)

    ticks=X@w.reshape(-1,1)+clf.intercept_
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.hist(ticks[Y==0],density=True,color='b',bins=99,label='health control',alpha=0.6)
    ax.hist(ticks[Y==1],density=True,color='r',bins=99,label='ASD',alpha=0.6)
    ax.set_xlabel('param value')
    ax.set_ylabel('probability')
    
    acc=1-(len(ticks[(Y==0) & (ticks>0).reshape(-1)])+len(ticks[(Y==1) & (ticks<0).reshape(-1)]))/200
    ax.set_title(acc)
    quickspine(ax)


'''
0226
the smooth svm has some problem, need to double check
the solution may not be unique
'''