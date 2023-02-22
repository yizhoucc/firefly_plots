
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np

# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
X.shape
y.shape
# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
testX.shape

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(testy))]

# fit a model
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)
# predict probabilities
lr_probs = model.predict_proba(testX)
lr_probs.shape
np.sum(lr_probs, axis=1)

# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]

# calculate scores using np
threshold=np.linspace(0,1,30)
def tfp(t, lr_probs,testy):
    tp=sum((lr_probs>t) & np.equal(testy, 1))
    tn=sum((lr_probs<=t) & np.equal(testy, 0))
    fp=sum((lr_probs>t) & np.equal(testy, 0))
    fn=sum((lr_probs<=t) & np.equal(testy, 1))
    tpr=tp/sum(np.equal(testy,1))
    # tpr=tp/(tp+fn)
    fpr=fp/sum(np.equal(testy,0))
    # fp/(fp+tn)
    return tpr, fpr

roc=[] # fp, tp, given threshold
for t in threshold:
    roc.append(tfp(t, lr_probs, testy))
roc=np.array(roc)    
pyplot.plot(roc[:,1],roc[:,0])

np.sum(np.abs(np.diff(roc[:,1]))*roc[:-1,0])


# calculate scores
ns_auc = roc_auc_score(testy, ns_probs)
lr_auc = roc_auc_score(testy, lr_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()# precision-recall curve and f1


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot
# generate 2 class dataset
X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
# fit a model
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)
# predict probabilities
lr_probs = model.predict_proba(testX)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# predict class values
yhat = model.predict(testX)
lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
# summarize scores
print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(testy[testy==1]) / len(testy)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

'''
获取数据 n 个 m 维的数据
随机生成 K 个 m 维的点
while(t)
    for(int i=0;i < n;i++)
        for(int j=0;j < k;j++)
            计算点 i 到类 j 的距离
    for(int i=0;i < k;i++)
        1. 找出所有属于自己这一类的所有数据点
        2. 把自己的坐标修改为这些数据点的中心点坐标
end

'''

from matplotlib import pyplot as plt

nstep=10
k=3
k_hat=4
n,m=100*k,2

x=[]
for _ in range(k):
    mu=np.random.random(m)*5
    std=np.random.random(m)+1
    x.append(np.random.normal(mu, std, size=(100,m)))
x=np.vstack(x)

mu_hat=np.random.random((k_hat, m))*5+1

for _ in range(nstep):

    tmp=[] # distance to k_hat centers
    for i in range(k_hat):
        tmp.append((np.sum(np.power(x-mu_hat[i], 2), axis=1))**0.5)
    tmp=np.vstack(tmp).T

    newmu=[] 
    assignments=[]
    for i in range(k_hat):
        assignments.append(np.where(np.min(tmp,axis=1)==tmp[:,i])[0])
        tmp2=np.mean(x[np.where(np.min(tmp,axis=1)==tmp[:,i])[0]], axis=0)
        newmu.append(tmp2)
    mu_hat=np.vstack(newmu)

    # plt.scatter(x[:,0],x[:,1],color='blue')
    plt.scatter(mu_hat[:,0], mu_hat[:,1], color='red')
    for inds in assignments:
        plt.scatter(x[inds,0],x[inds,1])
    plt.axis('equal')
    plt.show()
 

class Kmean():
    def __init__(self, x, k_hat):
        self.k_hat=k_hat
        self.x=x
        
        inds=set()
        while len(inds)<k_hat:
            inds.add(np.random.randint(low=0, high=len(x)-1))
    
        self.initmu=x[list(inds)]
    
    def findlabels(self,mus):
        distances=[] # distance to k_hat centers
        for mu in mus:
            distances.append((np.sum(np.power(x-mu, 2), axis=1))**0.5)
        distances=np.vstack(distances).T

        assignments=[]
        for i in range(self.k_hat):
           assignments.append(np.where(np.min(distances,axis=1)==distances[:,i])[0])

        return assignments

    def findcenter(self, assignments):
        newmu=[] 
        for inds in assignments:
            newmu.append(np.mean(x[inds], axis=0))
        return newmu

    def loss(self,mus):
        distances=[] # distance to k_hat centers
        for mu in mus:
            distances.append((np.sum(np.power(x-mu, 2), axis=1))**0.5)
        distances=np.vstack(distances).T
        loss=sum(np.min(distances, axis=1))
        return loss

    def fit(self,iter=3):
        ass=self.findlabels(self.initmu)

        for i in range(iter):
            
            mus=self.findcenter(ass)
            loss=self.loss(mus)
            print('iter, ', iter, 'loss: ', loss)
            ass=self.findlabels(mus)

            plt.scatter(mus[:,0], mus[:,1], color='red')
            for inds in ass:
                plt.scatter(x[inds,0],x[inds,1])
            plt.axis('equal')
            plt.show()
                        
    
model=Kmean(x, k_hat)
model.fit(iter=10)            






class LogisticRegression():

    def __init__(self, learning_rate=.1, n_iterations=4000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def initialize_weights(self, n_features):
        # 初始化参数
        # 参数范围[-1/sqrt(N), 1/sqrt(N)]
        limit = np.sqrt(1 / n_features)
        w = np.random.uniform(-limit, limit, (n_features, 1))
        b = 0
        self.w = np.insert(w, 0, b, axis=0)

    def fit(self, X, y):
        m_samples, n_features = X.shape
        self.initialize_weights(n_features)
        # 为X增加一列特征x1，x1 = 0
        X = np.insert(X, 0, 1, axis=1)
        y = np.reshape(y, (m_samples, 1))

        # 梯度训练n_iterations轮
        for i in range(self.n_iterations):
            h_x = X.dot(self.w)
            y_pred = sigmoid(h_x)
            w_grad = X.T.dot(y_pred - y)
            self.w = self.w - self.learning_rate * w_grad

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        h_x = X.dot(self.w)
        y_pred = np.round(sigmoid(h_x))
        return y_pred.astype(int)




l1=[[1,2] ,[3,9]]

l2 = [[3,4],[7,11],[14,16]]

res=[]
if l1[0][0]<l2[0][0]:
    res.append(l1[0])
else:
    res.append(l2[0])

p1,p2=0,0

while p1<len(l1) and p2<len(l2):
    s1,e1 = l1[p1]
    s2,e2 = l2[p2]
    print(l1[p1],l2[p2],p1,p2)
    if s1<s2:
        if s1<=res[-1][1] and e1>res[-1][1]:
            res[-1][1]=e1
        elif s1>res[-1][1]:
            res.append([s1,e1])
        p1+=1
    else:
        if s2<=res[-1][1] and e2>res[-1][1]:
            res[-1][1]=e2
        elif s2>res[-1][1]:
            res.append([s2,e2])
        p2+=1

if p1<len(l1):
    res+=l1[p1:]

if p2<len(l2):
    res+=l2[p2:]
print(res)















