import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)




ntokens = vocabsize  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout)

# [seq_len,batch_size,embedding_size] 
x=torch.randn(size=(66, 20))
x=x.int()
x=torch.abs(x)

model(x, torch.ones(size=(66,66)))


data=x
seq_len = data.size(0)
print(seq_len,'seq len')
src_mask = generate_square_subsequent_mask(66)
# y=model(data, src_mask)
# y.shape
src = (model.encoder(data)* math.sqrt(model.d_model))
print(src.shape,'src 1') # the encoding?
src = model.pos_encoder(src)
print(src.shape,'src 2')
output = model.transformer_encoder(src, src_mask)
print(output.shape,'oput 1')
output = model.decoder(output)
print(output.shape,'out 2')


# make simulation data --------------------------------------------

# load env and agent
# behavior stats results
import os
os.chdir('..')
from stable_baselines3 import TD3
from firefly_task import ffacc_real
from env_config import Config
import numpy as np
from plot_ult import suppress

arg = Config()
env=ffacc_real.FireFlyPaper(arg)
env.debug=True
env.terminal_vel=0.1
env.episode_len=40
agent_=TD3.load('trained_agent/paper.zip')
agent=agent_.actor.mu.cpu()

# task configurations (distance, gain, pro noise)
possible_phi=[]
taskdistance=np.linspace(0.5,1,5)
taskgain=np.linspace(0.5,1.5,5)
tasknoise=np.linspace(0.1,1,5)
for a in taskdistance:
    for b in taskgain:
        for c in tasknoise:
            possible_phi.append([a,b,c])
possible_phi=np.array(possible_phi)

# agent assumptions (gain, pro noise, obs noise, action cost)
possible_theta=[]
agentgain=np.linspace(0.3,1.2,5)
agentnoise=np.linspace(0.1,1,5)
agentobs=np.linspace(0.1,1,5)
agentcost=np.linspace(0,1,5)
for a in agentgain:
    for b in agentnoise:
        for c in agentobs:
            for d in agentcost:
                possible_theta.append([a,b,c,d])
possible_theta=np.array(possible_theta)

# get data
import pandas as pd
data=pd.DataFrame(columns=['dist','taskgain','tasknoise','again','anoise','aobsnoise','acost', 'traj' ])

for dist,taskgain,tasknoise in possible_phi:
    for again,anoise,aobsnoise,acost in possible_theta:
        phi=torch.tensor([[taskgain],   
                        [2],   
                        [tasknoise],   
                        [0.001],   
                        [0.9],   
                        [0.9],   
                        [0.13],   
                        [0.9],   
                        [0.9],   
                        [0.5],   
                        [0.5]])
        theta=torch.tensor([[again],   
                        [2],   
                        [anoise],   
                        [0.001],   
                        [aobsnoise],   
                        [0.9],   
                        [0.13],   
                        [acost],   
                        [0.9],   
                        [0.1],   
                        [0.1]])
        phi, theta=phi.float(), theta.float()
        env.reset(phi=phi, theta=theta, goal_position=[dist,0], vctrl=0, wctrl=0)
        
        traj=[]

        with suppress():
            done=False
            while not done:
                with torch.no_grad():
                    a=agent(env.decision_info)
                a[0,1]=0 # foce the 1d, ingore the angular action
                traj.append(a[0,0].item())
                _,_,done,_=env.step(a)

        row={'dist':dist,'taskgain':taskgain,'tasknoise':tasknoise,'again':again,'anoise':anoise,'aobsnoise':aobsnoise,'acost':acost, 'traj':traj}
        data=pd.concat([data, pd.Series(row).to_frame().T],ignore_index=True)
        print('{:.1f}'.format(len(data)/len(possible_phi)/len(possible_theta)*100))

# save for later use
import pickle
with open(datapath/'transformer_data', 'wb+') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# process trajectory (discretilize)
def process(x, reso=10):
    res=x*reso
    res=res.int()
    return res


maxlen=max([len(s) for s in data.traj])

def padding(x,maxlen,padvalue, endvalue):
    for row in x:
        row+=([padvalue]*(maxlen-len(row)))
        row+=endvalue
    return x

reso=10 # discretilize resolution. smallest unit is 1/reso
vocabsize=reso+2 # plus the pad and end value

ntokens = vocabsize  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout)


