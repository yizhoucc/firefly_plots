
print('''toy example acc state dependent noise''')
from plot_ult import *
import numpy as np
from matplotlib import pyplot as plt
from plot_ult import quickleg, quickspine, quicksave

def geta(tau,dt=0.01):
    a = np.exp(-dt/tau)
    return a

def getprocessnoisescalar(tau):
    a=geta(tau)
    s=(a**2-a*2+1)
    return s

statenames=[
    'd',
    'v',
    'pv',
    'pn',
    'ov',
    'on',
]



# standard integraltion model ---------------
class DynamicSystem:

    def __init__(self, B, initalstate, a=0.4, pa=0.2, oa=0.2,dt=0.01) -> None:
        self.B=B
        self.dt=dt
        self.a, self.pa,self.oa=a,pa,oa
        self.x=[initalstate]
        initA=self.computeA(initalstate[1,0],initalstate[2,0])
        self.A=[initA]
    
    def step_(self,u, newnoise, t=-1):
        nextx=self.A[t]@self.x[t] + self.B@u + newnoise 
        self.A.append(self.computeA(self.x[t][1,0],self.x[t][2,0]))
        self.x.append(nextx)

    def step(self,x, u, A=None, B=None,t=-1): # prediction
        if A is None:
            A=self.A[t]
        if B is None:
            B=self.B
        return A@x + B@u

    def computeA(self, v, pv):
        # x = [d, v, pv, pn, ov, on]
        a,pa,oa=self.a,self.pa,self.oa
        A=np.array([
            [1, 0, self.dt,0,0,0],
            [0, a, 0, 0,0,0],
            [0,1,0,v,0,0],
            [0,0,0,2*pa-pa**2,0,0],
            [0,0,1,0,0,pv],
            [0,0,0,0,0,2*oa-oa**2]
        ])
        return A

tau,ptau,otau=1.,0.3,0.6
noiselevel=100
a,pa,oa=geta(tau),geta(ptau),geta(otau)
B=np.array([[0],[1-a],[0], [0],[0], [0]])
x0=np.array([[0],[0],[0],[0],[0], [0]])
world=DynamicSystem(B, x0,a=a,pa=pa,oa=oa)


for _ in range(99):
    newnoise=np.zeros_like(world.x[-1]).astype('float32')
    newnoise[3,0]=np.random.normal(0,1)*noiselevel*getprocessnoisescalar(ptau)
    newnoise[5,0]=np.random.normal(0,1)*noiselevel*getprocessnoisescalar(otau)
    # control=1
    control=np.random.normal(0.7,0.1)
    world.step_(np.ones((1,1))*control, newnoise) 

xs=np.array(world.x)[:,:,0].T
for i in range(len(xs)):
    with initiate_plot(2,2,100) as f:
        plt.plot(xs[i])
        plt.title(statenames[i])
        plt.show()



class Belief:

    def __init__(self, dynamic_system, C, horizon=200,q=0.1, rp=0.1,ro=0.1, a=0.1) -> None:
         # belief is a simpler model. 
        # d, pv
        self.system=dynamic_system
        self.C=C
        self.horizon = horizon
        # self.PN,self.ON=[],[]
        self.P = [np.zeros((2,2))]
        self.S=[np.zeros((2,2))]
        self.Kf = []
        self.x=[np.array([[0],[0]])]
        self.q=q # prediction noise (var)
        self.Q=np.zeros((2,2))
        self.Q[1,1]=q**2
        self.rp=rp # obs noise of process optic flow
        self.ro=ro # obs noise of obs (disruptor) flow
        self.y=[]  
        self.a=a
        self.gain=1
        self.dt=self.system.dt
        self.A=np.array([
            [1, self.dt],
            [0, a],])

    def observe(self, t=-1):
        # observe the optic flow as a single flow
        pn=self.rp
        on=self.ro
        pv=self.system.x[-1][2]
        ov=self.system.x[-1][4]
        mu=(pn*ov+on*pv)/(pn+on)
        var=pn*on/(pn+on)
        return mu, var

    # the belief, linear quadratic estimator (Kalman filter) 
    def lqe(self,t=-1,C=None):

        obs, r = self.observe()
        R=r

        C=self.C if C is None else C
        A=self.A
        self.Kf.append(self.P[t] @ C.T @ np.linalg.pinv(C @ self.P[t] @ C.T + R))
        self.S.append(self.P[t] - self.Kf[t] @ C @ self.P[t])
        if t < self.horizon - 1:
            self.P.append(A @ self.S[t] @ A.T +self.Q) # this is P_t+1


    def step(self,u,C=None):
        y, var=self.observe()
        self.y.append(y)
        self.lqe() # get the current kalman gain
        k=self.Kf[-1]
        # predicted_x=self.system.step(self.x[-1],u)
        
        # predicted_x=(1-self.a)*self.x[-1][1]*self.system.dt + self.a*u
        B=np.array([[0],[self.a]])
        predicted_x=self.A@self.x[-1] + B*u
        # q = self.q
        # mu=(obs*q+prediction*r)/(q+r)
        # var=q*r/(q+r)

        err=(y-predicted_x[1])
        estimated_x=predicted_x+k*(err)
        self.x.append(estimated_x)

tau,ptau,otau=0.8,0.5,0.3
noiselevel=66
a,pa,oa=geta(tau),geta(ptau),geta(otau)
B=np.array([[0],[1-a],[0], [0],[0], [0]])
x0=np.array([[0],[0],[0],[0],[0], [0]])
world=DynamicSystem(B, x0,a=a,pa=pa,oa=oa)

C=np.array([[0,1]]) # can observe v
q=111
rp=0.2
ro=0.2
monkey=Belief(world,C,q=q,rp=rp, ro=ro)

for t in range(monkey.horizon-1):
    newnoise=np.zeros_like(world.x[-1]).astype('float32')
    newnoise[3,0]=np.random.normal(0,1)*noiselevel*getprocessnoisescalar(ptau)
    newnoise[5,0]=np.random.normal(0,1)*noiselevel*getprocessnoisescalar(otau)
    # control=0.3
    if t<150:
        control=1
    else: control=-1
    # control=np.random.normal(0.3,0.2)
    world.step_(np.ones((1,1))*control, newnoise) 
    monkey.step(np.ones((1,1))*control)

# self=monkey

# monkey's belief about v
monkey_vs=np.array([d[1] for d in monkey.x]).reshape(-1)
monkey_vs_uncertainty=np.array([p[1,1]**0.5 for p in monkey.S])
plt.fill_between(list(range(monkey.horizon)), monkey_vs-monkey_vs_uncertainty,monkey_vs+monkey_vs_uncertainty, alpha=0.4, label='belief')
plt.plot([d[2] for d in world.x], label='actual optic flow {}'.format(rp))
plt.plot([d[4] for d in world.x], label='disruptor optic flow {}'.format(ro))
# plt.plot([d[1] for d in world.x], label='prediction from ctrl {}'.format(q))
quickleg(plt.gca(),bbox_to_anchor=(0,0))
plt.title('velocity')
plt.xlabel('time')
plt.show()

# monkey's belief about d
monkey_ds=np.array([d[0] for d in monkey.x]).reshape(-1)
monkey_ds_uncertainty=np.array([p[0,0]**0.5 for p in monkey.S])
plt.fill_between(list(range(monkey.horizon)), monkey_ds-monkey_ds_uncertainty,monkey_ds+monkey_ds_uncertainty, alpha=0.4, label='belief')
i=0
plt.plot([d[i] for d in world.x], label='actual')
plt.legend()
plt.title('distance')
plt.xlabel('time')
plt.show()


# # actual state
# for i in range(len(world.x[-1])):
#     plt.plot([d[i] for d in world.x], label='actual')
#     plt.legend()
#     plt.title(statenames[i])
#     plt.xlabel('time')
#     plt.show()


# new, mixing opticlow  -----------------------------

class DynamicSystem:

    def __init__(self, B, initalstate, a=0.4, pa=0.2, oa=0.2,dt=0.01) -> None:
        self.B=B
        self.dt=dt
        self.a, self.pa,self.oa=a,pa,oa
        self.x=[initalstate]
        initA=self.computeA(initalstate[1,0],initalstate[2,0])
        self.A=[initA]
    
    def step_(self,u, newnoise, t=-1):
        nextx=self.A[t]@self.x[t] + self.B@u + newnoise 
        self.A.append(self.computeA(self.x[t][1,0],self.x[t][2,0]))
        self.x.append(nextx)

    def step(self,x, u, A=None, B=None,t=-1): # prediction
        if A is None:
            A=self.A[t]
        if B is None:
            B=self.B
        return A@x + B@u

    def computeA(self, v, pv):
        # x = [d, v, pv, pn, ov, on]
        a,pa,oa=self.a,self.pa,self.oa
        A=np.array([
            [1, 0, self.dt,0,0,0],
            [0, a, 0, 0,0,0],
            [0,1,0,v,0,0],
            [0,0,0,2*pa-pa**2,0,0],
            [0,0,1,0,0,pv],
            [0,0,0,0,0,2*oa-oa**2]
        ])
        return A

class Belief:

    def __init__(self, dynamic_system, C, horizon=200,q=0.1, rp=0.1,ro=0.1, a=0.1) -> None:
         # belief is a simpler model. 
        # d, pv
        self.system=dynamic_system
        self.C=C
        self.horizon = horizon
        # self.PN,self.ON=[],[]
        self.P = [np.zeros((2,2))]
        self.S=[np.zeros((2,2))]
        self.Kf = []
        self.x=[np.array([[0],[0]])]
        self.q=q # prediction noise (var)
        self.Q=np.zeros((2,2))
        self.Q[1,1]=q**2
        self.rp=rp # obs noise of process optic flow
        self.ro=ro # obs noise of obs (disruptor) flow
        self.y=[]  
        self.a=a
        self.gain=1
        self.dt=self.system.dt
        self.A=np.array([
            [1, self.dt],
            [0, a],])

    def observe(self, t=-1):
        # observe the optic flow as a single flow
        pn=self.rp
        on=self.ro
        pv=self.system.x[-1][2]
        ov=self.system.x[-1][4]

        samples=np.array([np.random.normal(pv, pn, size=(50)),
        np.random.normal(ov, on, size=(50))]).reshape(-1)
        mu, var=np.mean(samples), np.var(samples)

        sample = np.random.uniform(np.min(samples), np.max(samples))
        if (sample-pv)**2>(sample-ov)**2:
            return pv, pn
        else:
            return ov,on

        return mu, var

    # the belief, linear quadratic estimator (Kalman filter) 
    def lqe(self,t=-1,C=None):

        obs, r = self.observe()
        R=r

        C=self.C if C is None else C
        A=self.A
        self.Kf.append(self.P[t] @ C.T @ np.linalg.pinv(C @ self.P[t] @ C.T + R))
        self.S.append(self.P[t] - self.Kf[t] @ C @ self.P[t])
        if t < self.horizon - 1:
            self.P.append(A @ self.S[t] @ A.T +self.Q) # this is P_t+1


    def step(self,u,C=None):
        y, var=self.observe()
        self.y.append(y)
        self.lqe() # get the current kalman gain
        k=self.Kf[-1]
        # predicted_x=self.system.step(self.x[-1],u)
        
        # predicted_x=(1-self.a)*self.x[-1][1]*self.system.dt + self.a*u
        B=np.array([[0],[self.a]])
        predicted_x=self.A@self.x[-1] + B*u
        # q = self.q
        # mu=(obs*q+prediction*r)/(q+r)
        # var=q*r/(q+r)

        err=(y-predicted_x[1])
        estimated_x=predicted_x+k*(err)
        self.x.append(estimated_x)

tau,ptau,otau=0.8,0.5,0.3
noiselevel=66
a,pa,oa=geta(tau),geta(ptau),geta(otau)
B=np.array([[0],[1-a],[0], [0],[0], [0]])
x0=np.array([[0],[0],[0],[0],[0], [0]])
world=DynamicSystem(B, x0,a=a,pa=pa,oa=oa)

C=np.array([[0,1]]) # can observe v
q=999.7
rp=0.2
ro=0.2
monkey=Belief(world,C,q=q,rp=rp, ro=ro)

for t in range(monkey.horizon-1):
    newnoise=np.zeros_like(world.x[-1]).astype('float32')
    newnoise[3,0]=np.random.normal(0,1)*noiselevel*getprocessnoisescalar(ptau)
    newnoise[5,0]=np.random.normal(0,1)*noiselevel*getprocessnoisescalar(otau)
    # control=0.3
    if t<150:
        control=1
    else: control=-1
    # control=np.random.normal(0.3,0.2)
    world.step_(np.ones((1,1))*control, newnoise) 
    monkey.step(np.ones((1,1))*control)


# monkey's belief about v
monkey_vs=np.array([d[1] for d in monkey.x]).reshape(-1)
monkey_vs_uncertainty=np.array([p[1,1]**0.5 for p in monkey.S])
# plt.fill_between(list(range(monkey.horizon)), [d[2,0]-q**0.5 for d in world.x],[d[1,0]+q**0.5 for d in world.x], alpha=0.2, label='prediction from ctrl {}'.format(q))
plt.fill_between(list(range(monkey.horizon)), [d[2,0]-rp**0.5*pa for d in world.x],[d[2,0]+rp**0.5*pa for d in world.x], alpha=0.4, label='actual optic flow {}'.format(rp))
plt.fill_between(list(range(monkey.horizon)), [d[4,0]-ro**0.5*oa for d in world.x],[d[4,0]+ro**0.5*oa for d in world.x], alpha=0.4, label='disruptor optic flow {}'.format(rp))
plt.fill_between(list(range(monkey.horizon)), monkey_vs-monkey_vs_uncertainty,monkey_vs+monkey_vs_uncertainty, alpha=0.6, label='belief')
quickleg(plt.gca(),bbox_to_anchor=(0,0))
plt.title('velocity')
plt.xlabel('time')
plt.show()


# monkey's belief about d
monkey_ds=np.array([d[0] for d in monkey.x]).reshape(-1)
monkey_ds_uncertainty=np.array([p[0,0]**0.5 for p in monkey.S])
plt.fill_between(list(range(monkey.horizon)), monkey_ds-monkey_ds_uncertainty,monkey_ds+monkey_ds_uncertainty, alpha=0.4, label='belief')
i=0
plt.plot([d[i] for d in world.x], label='actual')
plt.legend()
plt.title('distance')
plt.xlabel('time')
plt.show()


# check manual intergration of noise, compare with total uncertainty
ds_var_inte=np.power(np.cumsum(np.array([p[1,1] for p in monkey.S])), 0.5)*0.1

ds_var_inte=[0]
for p in monkey.S:
    ds_var_inte.append((ds_var_inte[-1]*(1-a)+a*p[1,1])**0.5)
ds_var_inte.pop()

plt.plot(monkey_ds_uncertainty)
plt.plot(ds_var_inte)

monkey_ds=np.array([d[0] for d in monkey.x]).reshape(-1)
plt.fill_between(list(range(monkey.horizon)), monkey_ds-ds_var_inte,monkey_ds+ds_var_inte, alpha=0.4, label='belief')
i=0
plt.plot([d[i] for d in world.x], label='actual')
plt.legend()
plt.title('distance')
plt.xlabel('time')
plt.show()



# do not use prediction and kf ----------------
from scipy import stats
class Belief:

    def __init__(self, dynamic_system, C, horizon=200,q=0.1, rp=0.1,ro=0.1, a=0.1) -> None:
         # belief is a simpler model. 
        # d, pv
        self.system=dynamic_system
        self.C=C
        self.horizon = horizon
        # self.PN,self.ON=[],[]
        self.P = [np.zeros((2,2))]
        self.S=[np.zeros((2,2))]
        self.Kf = []
        self.x=[np.array([[0],[0]])]
        self.q=q # prediction noise (var)
        self.Q=np.zeros((2,2))
        self.Q[1,1]=q**2
        self.rp=rp # obs noise of process optic flow
        self.ro=ro # obs noise of obs (disruptor) flow
        self.y=[]  
        self.a=a
        self.gain=1
        self.dt=self.system.dt
        self.A=np.array([
            [1, self.dt],
            [0, a],])

    def observe(self, t=-1):
        # observe the optic flow as a single flow
        pn=self.rp
        on=self.ro
        pv=self.system.x[-1][2]
        ov=self.system.x[-1][4]

        samples=np.array([np.random.normal(pv, pn, size=(50)),
        np.random.normal(ov, on, size=(50))]).reshape(-1)
        mu, var=np.mean(samples), np.var(samples)

        # if t%50==0:
        if pv-ov>0.2:
            x = np.linspace(mu - 3*var**0.5, mu + 3*var**0.5, 100)
            plt.plot(x, stats.norm.pdf(x, mu, var**0.5))
            plt.plot(x, stats.norm.pdf(x, pv, pn**0.5))
            plt.plot(x, stats.norm.pdf(x, ov, on**0.5))
            plt.show()

        return mu, var

    def step(self,u,C=None):
        y, var=self.observe()
        self.y.append(y)

        # d=d+v*dt
        x_=np.zeros((2,1))
        x_[0]=x_[0]+y*self.dt
        # v=y
        x_[1]=y
        self.x.append(x_)

        #d var
        S_=np.zeros((2,2))
        S_[0,0]=self.S[t][0,0]+var*0.01
        S_[1,1]=var
        self.S.append(S_)


tau,ptau,otau=0.8,0.5,0.3
noiselevel=66
a,pa,oa=geta(tau),geta(ptau),geta(otau)
B=np.array([[0],[1-a],[0], [0],[0], [0]])
x0=np.array([[0],[0],[0],[0],[0], [0]])
world=DynamicSystem(B, x0,a=a,pa=pa,oa=oa)

C=np.array([[0,1]]) # can observe v
q=11.7
rp=0.2
ro=0.2
monkey=Belief(world,C,q=q,rp=rp, ro=ro)

for t in range(monkey.horizon-1):
    newnoise=np.zeros_like(world.x[-1]).astype('float32')
    newnoise[3,0]=np.random.normal(0,1)*noiselevel*getprocessnoisescalar(ptau)
    newnoise[5,0]=np.random.normal(0,1)*noiselevel*getprocessnoisescalar(otau)
    # control=0.3
    if t<150:
        control=1
    else: control=-1
    # control=np.random.normal(0.3,0.2)
    world.step_(np.ones((1,1))*control, newnoise) 
    monkey.step(np.ones((1,1))*control)


# monkey's belief about v
monkey_vs=np.array([d[1] for d in monkey.x]).reshape(-1)
monkey_vs_uncertainty=np.array([p[1,1]**0.5 for p in monkey.S])
# plt.fill_between(list(range(monkey.horizon)), [d[2,0]-q**0.5 for d in world.x],[d[1,0]+q**0.5 for d in world.x], alpha=0.2, label='prediction from ctrl {}'.format(q))
plt.fill_between(list(range(monkey.horizon)), [d[2,0]-rp**0.5*pa for d in world.x],[d[2,0]+rp**0.5*pa for d in world.x], alpha=0.4, label='actual optic flow {}'.format(rp))
plt.fill_between(list(range(monkey.horizon)), [d[4,0]-ro**0.5*oa for d in world.x],[d[4,0]+ro**0.5*oa for d in world.x], alpha=0.4, label='disruptor optic flow {}'.format(rp))
plt.fill_between(list(range(monkey.horizon)), monkey_vs-monkey_vs_uncertainty,monkey_vs+monkey_vs_uncertainty, alpha=0.6, label='belief')
quickleg(plt.gca(),bbox_to_anchor=(0,0))
plt.title('velocity')
plt.xlabel('time')
plt.show()


# monkey's belief about d
monkey_ds=np.array([d[0] for d in monkey.x]).reshape(-1)
monkey_ds_uncertainty=np.array([p[0,0]**0.5 for p in monkey.S])
plt.fill_between(list(range(monkey.horizon)), monkey_ds-monkey_ds_uncertainty,monkey_ds+monkey_ds_uncertainty, alpha=0.4, label='belief')
i=0
plt.plot([d[i] for d in world.x], label='actual')
plt.legend()
plt.title('distance')
plt.xlabel('time')
plt.show()


ds_var_inte=np.power(np.cumsum(np.array([p[1,1] for p in monkey.S])), 0.5)*0.1

ds_var_inte=[0]
for p in monkey.S:
    ds_var_inte.append((ds_var_inte[-1]*(1-a)+a*p[1,1])**0.5)
ds_var_inte.pop()

plt.plot(monkey_ds_uncertainty)
plt.plot(ds_var_inte)

monkey_ds=np.array([d[0] for d in monkey.x]).reshape(-1)
plt.fill_between(list(range(monkey.horizon)), monkey_ds-ds_var_inte,monkey_ds+ds_var_inte, alpha=0.4, label='belief')
i=0
plt.plot([d[i] for d in world.x], label='actual')
plt.legend()
plt.title('distance')
plt.xlabel('time')
plt.show()

