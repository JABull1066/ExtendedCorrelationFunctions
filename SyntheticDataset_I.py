import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from helperFunctions import *

sns.set_style('white')
sns.set(font_scale=5)

# Generate synthetic data - strong clustering on LHS, weak exclusion on RHS
np.random.seed(1107)
seeds1 = 0.1 + 0.8*np.random.rand(20,2)*np.array([0.5,1])
labs = []
contuslabs = []
points = np.empty(shape=(0,2))
sigma = 0.02
# Cluster process on LHS
for i in range(len(seeds1)):
    p = seeds1[i,:] + np.random.randn(20,2)*sigma
    labs.extend(['$C_1$' for v in range(10)])
    contuslabs.extend([np.nan for v in range(10)])
    # contuslabs.extend([0.25 + np.random.rand()*0.25 for v in range(10)])
    labs.extend(['$C_2$' for v in range(10)])
    contuslabs.extend([np.random.rand()*0.5 for v in range(10)])
    points = np.vstack((points,p))

# Different cluster processes on RHS
seeds2 = 0.1 + 0.8*np.random.rand(20,2)*np.array([0.5,1]) + [0.4,0]
for i in range(len(seeds2)):
    p = seeds2[i,:] + np.random.randn(10,2)*sigma
    if i < len(seeds2)*0.5:
        labs.extend(['$C_1$' for v in range(10)])
        contuslabs.extend([np.nan for v in range(10)])
        # contuslabs.extend([np.random.rand()*0.25 for v in range(10)])
    else:
        labs.extend(['$C_2$' for v in range(10)])
        contuslabs.extend([0.5 + np.random.rand()*0.5 for v in range(10)])
    points = np.vstack((points,p))


points = points*1000

pc = generatePointCloud('Synthetic Dataset I',points,domain=[[0,1000],[0,1000]])
pc.addLabels('Celltype','categorical',labs,cmap='tab10')
pc.addLabels('$m$','continuous',contuslabs,cmap='RdBu_r')

visualisePointCloud(pc,'Celltype',markerSize=100)#,showBoundary=True)
visualisePointCloud(pc,'$m$',markerSize=100,cmap='Oranges')#,showBoundary=True)
c1mask = np.asarray(labs) == '$C_1$'
plt.scatter(points[c1mask,0],points[c1mask,1],s=100,zorder=-1)
#%% Calculate cross-PCFs - Synthetic Dataset I (Fig 2)
a = '$C_1$'
b = '$C_2$'
maxR=500
annulusStep = 1
annulusWidth = 10
r, pcf, contributions = pairCorrelationFunction(pc, 'Celltype', [a,b], maxR=maxR,annulusStep=annulusStep,annulusWidth=annulusWidth)
r2, pcf2, contributions2 = pairCorrelationFunction(pc, 'Celltype', [b,a], maxR=maxR,annulusStep=annulusStep,annulusWidth=annulusWidth)

plt.figure(figsize=(18,18))
plt.gca().axhline(1,c='k',linestyle=':',lw=3)
plt.plot(r,pcf,lw=7,label='$g_{C_1 C_2}(r)$',linestyle='-')
plt.plot(r2,pcf2,lw=7,label='$g_{C_2 C_1}(r)$',linestyle=(0,(1,1.5)))
plt.xlabel('Radius, $r$ ($\mu$m)')
plt.ylim([0,7])
plt.legend()

#%% Calculate TCMs - Synthetic Dataset I (Fig 2)
tcm = topographicalCorrelationMap(pc,'Celltype','$C_1$','Celltype','$C_2$',radiusOfInterest=50,maxCorrelationThreshold=5.0,kernelRadius=150,kernelSigma=50,visualiseStages=False)

plt.figure(figsize=(20,20))
l = int(np.ceil(np.max(np.abs([tcm.min(),tcm.max()]))))
plt.imshow(tcm,cmap='RdBu_r',vmin=-l,vmax=l,origin='lower')
plt.colorbar(label='$\Gamma_{C_1 C_2}(r=50)$')
ax = plt.gca()
ax.grid(False)

tcm = topographicalCorrelationMap(pc,'Celltype','$C_2$','Celltype','$C_1$',radiusOfInterest=50,maxCorrelationThreshold=5.0,kernelRadius=150,kernelSigma=50,visualiseStages=False)

plt.figure(figsize=(20,20))
l = int(np.ceil(np.max(np.abs([tcm.min(),tcm.max()]))))
plt.imshow(tcm,cmap='RdBu_r',vmin=-l,vmax=l,origin='lower')
plt.colorbar(label='$\Gamma_{C_2 C_1}(r=50)$')
ax = plt.gca()
ax.grid(False)


#%% Calculate wPCFs - Synthetic Dataset I (Fig 4)

def weightingFunction(p,l_B):
    weights = 1-np.abs(p-l_B)/0.2
    weights = np.maximum(weights, np.zeros(np.shape(weights)))
    return weights

w = [weightingFunction(0.5, v) for v in np.linspace(0,1,101)]
plt.figure()
plt.plot(np.linspace(0,1,101),w)

r, targetP, wPCF = weightedPairCorrelationFunction(pc, 'Celltype', '$C_1$', '$m$',maxR=maxR,annulusStep=annulusStep,annulusWidth=annulusWidth,targetP=np.arange(0,1.01,0.01),weightingFunction=weightingFunction)
plotWeightedPCF(r,targetP,wPCF,vmin=0,vmax=2,ax=None,cm='plasma')

plt.figure(figsize=(18,18))
for v in np.arange(0,len(targetP),25):
    plt.plot(r,wPCF[:,v],label=f'$m = {targetP[v]}$',c=plt.cm.Oranges(v/len(targetP)),lw=5)
plt.legend()
plt.xlabel('Radius (r)')
plt.ylabel('wPCF($r$, $C_1$, $m$)')
plt.gca().axhline(1,c='k',linestyle=':',lw=3)
plt.ylim([0,10.5])
