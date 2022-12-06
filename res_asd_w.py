# why angular?

from matplotlib import pyplot as plt
import numpy as np
from plot_ult import *

# assumptions
# orgin is the camera
dt=1
c=-0.1 # the ground level
zmax=5
fov=pi/4
# fc=0.16
sensorsize=0.024197724949859098
fc=sensorsize/np.tan(fov/2)
userandom=False
# self motion
v=0.2*0.1
w=0.6*0.1
samplereso=2
fov_v=np.arctan(np.tan(pi/4)*16/9)
screenx=np.tan(fov/2)*fc
screeny=screenx
fov_world=[(-screenx, c, fc), (screenx, c, fc), (np.tan(fov/2)*zmax, c, zmax), (-1*np.tan(fov/2)*zmax, c, zmax),(-screenx, c, fc)]
# v=(np.random.random()*1+0.7)*dt
# w=(np.random.random()*2-1)*dt


def world2screen(P,fc=fc):
    X,Y,Z=P
    x,y=fc*X/Z, fc*Y/Z
    return x,y

def scrren2world(p,fc=fc,Y=c, c=c):
    x,y=p
    Z=fc*c/y
    X=x*Z/fc
    return X,Y,Z


# randome some 3d dots
dot3dx,dot3dz=[],[]
if userandom:
    # random
    ndots=100
    while len(dot3dx)<ndots:
        x=np.random.random()*2-1
        z=np.random.random()*2+fc
        if (x**2+z**2)**0.5<2:
            dot3dx.append(x)
            dot3dz.append(z)
else:
    # sample
    for x in np.linspace(-(zmax*np.tan(fov/2)),zmax*np.tan(fov/2),int(2*zmax*np.tan(fov/2)*samplereso)):
        for z in np.linspace(0, zmax, int(zmax*samplereso)):
            # x=x*1.1
            # if (x**2+z**2)**0.5<3: # render region
            dot3dx.append(x)
            dot3dz.append(z)
    ndots=len(dot3dx)
dot3dx,dot3dz = np.array(dot3dx).reshape(1,-1), np.array(dot3dz).reshape(1,-1)
# dot3dx=np.random.random(size=(1,ndots))*2-1
# dot3dz=np.random.random(size=(1,ndots))*2+fc
dot3d=np.vstack([dot3dx,c*np.ones((1,ndots)),dot3dz])

dot2d=[]
for i in range(ndots):
    x,y=world2screen(dot3d.T[i])
    dot2d.append([x,y])
dot2d=np.stack(dot2d).T

recovered3d=[]
for i in range(ndots):
    p=dot2d.T[i]
    x,y=p
    if -screenx<x<screenx and -screeny<y<-screeny*0.05:
        recovered3d.append(scrren2world(dot2d.T[i]))
recovered3d=np.stack(recovered3d).T

with initiate_plot(7,3,300) as f:
    ax=f.add_subplot(131)
    ax.scatter(dot3d[0],dot3d[2], s=1)
    quickspine(ax)
    ax.set_xlabel('world X')
    ax.set_ylabel('world Z')
    ax.set_title('overhead')
    ax=f.add_subplot(132)
    ax.scatter(dot2d[0],dot2d[1], s=1)
    ax.set_aspect('equal', adjustable='box')
    
    ax.set_xlim(-screenx,screenx)
    ax.set_ylim(-screeny,screeny)
    # quickspine(ax)
    ax.set_xlabel('screen x')
    ax.set_ylabel('screen y')
    ax.set_title('screen view')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot([-1,1],[0,0],'k')
    ax.set_aspect('equal', adjustable='box')

    ax=f.add_subplot(133)
    ax.scatter(recovered3d[0],recovered3d[2], s=1)
    quickspine(ax)
    ax.set_xlabel('world X')
    ax.set_ylabel('world Z')
    ax.set_title('in fov dots overhead')
    ax.set_aspect('equal', adjustable='box')
    for a,b in zip(fov_world[1:], fov_world[:-1]):
        sx,_,sz=a
        ex,_,ez=b
        ax.plot([sx,ex],[sz,ez],'k')
    plt.tight_layout()



# compute the motion vectors for each dot (Tx, Tz)
vs=np.vstack([np.zeros((1,ndots)),np.zeros((1,ndots)),np.ones((1,ndots))*v])
tempws=[]
for x,y,z in dot3d.T:
    unitvectorxz=np.array([-z,x])/np.linalg.norm([-z,x])
    unitvector=np.hstack([unitvectorxz[0], 0, unitvectorxz[1]])
    r=(x**2+z**2)**0.5
    wsize=np.tan(w)*r
    tempws.append(unitvector*wsize)
ws=np.vstack(tempws).T
T=vs+ws

dot2dend=[]
for i in range(ndots):
    x,y=world2screen(dot3d.T[i]+T.T[i])
    dot2dend.append([x,y])
dot2dend=np.stack(dot2dend).T

T2d=dot2dend-dot2d

# plot the motion field
with initiate_plot(9,3,300) as f:
    ax=f.add_subplot(131)
    for i in range(ndots):
        sx,sy,sz=dot3d.T[i]
        ex,ey,ez=dot3d.T[i]+T.T[i]
        ax.plot([sx,ex], [sz,ez])
    quickspine(ax)
    ax.set_xlabel('world X')
    ax.set_ylabel('world Z')
    ax.plot([0,0],[0,v], 'k')
    ax.plot([0,-w],[0,0],'k')
    ax.set_title('overhead, v and w')
    ax.set_aspect('equal', adjustable='box')

    ax=f.add_subplot(132,sharey=ax)
    for i in range(ndots):
        sx,sy,sz=dot3d.T[i]
        ex,ey,ez=dot3d.T[i]+vs.T[i]
        ax.plot([sx,ex], [sz,ez])
    quickspine(ax)
    ax.set_xlabel('world X')
    ax.set_ylabel('world Z')
    ax.plot([0,0],[0,v], 'k')
    ax.plot([0,-w],[0,0],'k')
    ax.set_title('overhead, v')
    ax.set_aspect('equal', adjustable='box')
    
    ax=f.add_subplot(133,sharey=ax)
    for i in range(ndots):
        sx,sy,sz=dot3d.T[i]
        ex,ey,ez=dot3d.T[i]+ws.T[i]
        ax.plot([sx,ex], [sz,ez])
    quickspine(ax)
    ax.set_xlabel('world X')
    ax.set_ylabel('world Z')
    ax.plot([0,0],[0,v], 'k')
    ax.plot([0,-w],[0,0],'k')
    ax.set_title('overhead, w')
    ax.set_aspect('equal', adjustable='box')

# traslate on to 2d image
with initiate_plot(7,3,300) as f:
    ax=f.add_subplot(121)
    for i in range(ndots):
        sx,sy,sz=dot3d.T[i]
        ex,ey,ez=dot3d.T[i]+T.T[i]
        ax.plot([sx,ex], [sz,ez])
    quickspine(ax)
    ax.set_xlabel('world X')
    ax.set_ylabel('world Z')
    ax.plot([0,0],[0,v], 'k')
    ax.plot([0,-w],[0,0],'k')
    ax.set_title('overhead view')
    ax.plot()
    ax.set_aspect('equal', adjustable='box')

    ax=f.add_subplot(122)
    for i in range(ndots):
        sx,sy=dot2d.T[i]
        if -screenx<sx<screenx and -screeny<sy<-screeny*0.1:
            ex,ey=dot2dend.T[i]
            ax.plot([sx,ex], [sy,ey])
    ax.set_xlim(-screenx,screenx)
    ax.set_ylim(-screeny,screeny)
    # quickspine(ax)
    ax.set_xlabel('screen x')
    ax.set_ylabel('screen y')
    ax.set_title('screen view')
    ax.plot([-1,1],[0,0])
    ax.set_aspect('equal', adjustable='box')

def v2t(p, v, fc=fc, c=c):
    x,y=p
    vx,vy=v
    # Z=fc*c/y
    # Tv=-Z*vy/y
    Tv=-1*fc*c/y*vy/y
    # Tw=(-Z*vx+Z*vy*x/y)/fc/Z
    Tw= (-1*fc*c/y*vx+ fc*c/y*vy*x/y) /fc /(fc*c/y)
    return Tv, Tw

def jocob(p, fc=fc, c=c):
    x,y=p
    jacobian_matrix= np.array(
        [[0 ,   -fc*c/y/y],
        [-1/fc,   1/(y+c)]
        ])
    return jacobian_matrix

# recoverT=[]
# for i in range(ndots):
#     p=dot2d.T[i]
#     v=T2d.T[i]
#     recoverT.append(v2t(p, v))
# recoverT=np.array(recoverT).T

# recoverTjacobian=[]
# for i in range(ndots):
#     p=dot2d.T[i]
#     J=jocob(p)
#     v=T2d.T[i]
#     recoverTjacobian.append(J@v.reshape(-1,1))
# recoverTjacobian=np.array(recoverT)


# TODO remove the dots that not in the field of view

# traslate on to 2d image
with initiate_plot(7,3,300) as f:
    ax=f.add_subplot(121)
    for i in range(ndots):
        sx,sy,sz=dot3d.T[i]
        ex,ey,ez=dot3d.T[i]+T.T[i]
        ax.plot([sx,ex], [sz,ez])
    # ax.axis('equal')
    quickspine(ax)
    ax.set_xlabel('world X')
    ax.set_ylabel('world Z')
    ax.plot([0,0],[0,v], 'k')
    ax.plot([0,-w],[0,0],'k')
    ax.set_title('overhead uncertainty')
    ax.set_aspect('equal', adjustable='box')

    ax=f.add_subplot(122)
    for i in range(ndots):
        x,y=dot2d.T[i]
        if -screenx<x<screenx and -screeny<y<-screeny*0.05:
            sx,sy=dot2d.T[i]
            ex,ey=dot2dend.T[i]
            ax.plot([sx,ex], [sy,ey])
    # ax.axis('equal')
    ax.set_xlim(-screenx,screenx)
    ax.set_ylim(-screeny,screeny)
    # quickspine(ax)
    ax.set_xlabel('screen x')
    ax.set_ylabel('screen y')
    ax.set_title('screen view uncertainty')
    ax.plot([-1,1],[0,0])
    ax.set_aspect('equal', adjustable='box')

recoverTuncertainty=[]
for i in range(ndots):
    p=dot2d.T[i]
    x,y=p
    if -screenx<x<screenx and -screeny<y<-screeny*0.1:
        J=jocob(p)
        sigma2=J@np.eye(2)@J.T
        recoverTuncertainty.append(sigma2)
infoTv=[1/mat[0,0] for mat in recoverTuncertainty]
infoTw=[1/mat[1,1] for mat in recoverTuncertainty]
print('jacobian on I (regardless of size), the information for Tv and Tw are: \n', sum(infoTv), sum(infoTw))



def screen_cov(p,motion=None,size=None, use='size', ratio=100):
    _,y=p
    size=abs(y)/screeny if size is None else size
    if use=='both':
        sigma2=(1/size)**2*ratio + (motion)**2*(1-ratio)
    elif use=='size':
        sigma2=(1/size)**2
    elif use=='motion':
        sigma2=(motion)**2
    cov=np.array([[sigma2,0],[0,sigma2]])
    return cov


recoverTuncertainty=[]
fovdots=[]
for i in range(ndots):
    p=dot2d.T[i]
    x,y=p
    if -screenx<x<screenx and -screeny<y<-screeny*0.1:
        J=jocob(p)
        sigma2=J@screen_cov(p, size=None, motion=np.linalg.norm(T2d.T[i]), use='size')@J.T
        recoverTuncertainty.append(sigma2)
        fovdots.append(i)
infoTv=[1/mat[0,0] for mat in recoverTuncertainty]
infoTw=[1/mat[1,1] for mat in recoverTuncertainty]
print('jacobian on cov(vx, vy), the information for Tv and Tw are: \n', sum(infoTv), sum(infoTw))


with initiate_plot(7,3,300) as f:
    ax=f.add_subplot(131)
    for j in range(len(fovdots)):
        i=fovdots[j]
        sx,sy,sz=dot3d.T[i]
        ex,ey,ez=dot3d.T[i]+(vs.T[i]/np.linalg.norm(vs.T[i]))*np.power(infoTv[j],0.3)
        ax.plot([sx,ex], [sz,ez],color='k',alpha=0.5)
    for j in range(len(fovdots)):
        i=fovdots[j]
        sx,sy,sz=dot3d.T[i]
        ex,ey,ez=dot3d.T[i]+(ws.T[i]/np.linalg.norm(ws.T[i]))*np.power(infoTw[j],0.3)
        ax.plot([sx,ex], [sz,ez],color='k',alpha=0.5)
        # ax.scatter(sx,sz)
    # ax.axis('equal')
    quickspine(ax)
    ax.set_xlabel('world X')
    ax.set_ylabel('world Z')
    # ax.plot([0,0],[0,v], 'k')
    # ax.plot([0,-w],[0,0],'k')
    ax.set_title('overhead, info V and W')

    ax=f.add_subplot(132)
    for j in range(len(fovdots)):
        i=fovdots[j]
        sx,sy,sz=dot3d.T[i]
        ex,ey,ez=dot3d.T[i]+(vs.T[i]/np.linalg.norm(vs.T[i]))*np.power(infoTv[j],0.3)
        ax.plot([sx,ex], [sz,ez],color='k',alpha=0.5)    
    quickspine(ax)
    ax.set_xlabel('world X')
    ax.set_ylabel('world Z')
    # ax.plot([0,0],[0,v], 'k')
    # ax.plot([0,-w],[0,0],'k')
    ax.set_title('overhead, info V')

    ax=f.add_subplot(133)
    for j in range(len(fovdots)):
        i=fovdots[j]
        sx,sy,sz=dot3d.T[i]
        ex,ey,ez=dot3d.T[i]+(ws.T[i]/np.linalg.norm(ws.T[i]))*np.power(infoTw[j],0.3)
        ax.plot([sx,ex], [sz,ez],color='k',alpha=0.5)  
    quickspine(ax)
    ax.set_xlabel('world X')
    ax.set_ylabel('world Z')
    # ax.plot([0,0],[0,v], 'k')
    # ax.plot([0,-w],[0,0],'k')
    ax.set_title('overhead, info W')




# # calculate the information for v and w, assuming the information of dot is motion^2
# info2d=(np.linalg.norm(T2d,axis=0))**2
# info3dratiov=(np.linalg.norm(vs,axis=0))**2/((np.linalg.norm(ws,axis=0))**2 + (np.linalg.norm(vs,axis=0))**2)
# info3dratiow=(np.linalg.norm(ws,axis=0))**2/((np.linalg.norm(ws,axis=0))**2 + (np.linalg.norm(vs,axis=0))**2)

# info3dv=info3dratiov*info2d
# info3dw=info3dratiow*info2d

# print('use motion,  sumed info v and info w \n', sum(info3dv), sum(info3dw))

# # use size but not motion
# info2d=1/(np.linalg.norm(np.delete(dot3d,1,0),axis=0))**2
# info3dratiov=(np.linalg.norm(vs,axis=0))**2/((np.linalg.norm(ws,axis=0))**2 + (np.linalg.norm(vs,axis=0))**2)
# info3dratiow=(np.linalg.norm(ws,axis=0))**2/((np.linalg.norm(ws,axis=0))**2 + (np.linalg.norm(vs,axis=0))**2)
# info3dv=info3dratiov*info2d
# info3dw=info3dratiow*info2d
# print('use size, sumed info v and info w \n', sum(info3dv), sum(info3dw))

