import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import random
import time
from functools import partial
from helperFunctions import *
from smallestEnclosingCircle import make_circle
sns.set_style('white')
sns.set(font_scale=5)
#%%
# Load in a point cloud
df = pd.read_csv('./ROI-0_MainText.csv')
labels = {'T Helper Cell' : 1,
                   'Regulatory T Cell' : 2,
                   'Cytotoxic T Cell' : 3,
                   'Macrophage' : 4,
                   'Neutrophil' : 5,
                   'Epithelium' : 6
                    }
    
points = np.asarray([df['x'],df['y']]).transpose()


pc = generatePointCloud('ROI',points,domain=[[0,1000],[0,1000]])
pc.addLabels('Celltype','categorical',df.Celltype,cmap='tab10')

pc.changeIndividualLabelColor('Celltype', 'Epithelium', plt.cm.tab10(0))
pc.changeIndividualLabelColor('Celltype', 'T Helper Cell', plt.cm.tab10(1))
pc.changeIndividualLabelColor('Celltype', 'Macrophage', plt.cm.tab10(2))
pc.changeIndividualLabelColor('Celltype', 'Neutrophil', plt.cm.tab10(4))
pc.changeIndividualLabelColor('Celltype', 'Cytotoxic T Cell', plt.cm.tab10(3))
pc.changeIndividualLabelColor('Celltype', 'Regulatory T Cell', plt.cm.tab10(5))

visualisePointCloud(pc,'Celltype',markerSize=100)

#%% All cross-PCFs (Fig 5)
maxR=150
annulusStep = 10
annulusWidth = 10

avals = []
bvals = []
pcfs = []
for a in pc.labels['Celltype']['categories']:
    for b in pc.labels['Celltype']['categories']:
        print(a,b)
        r, pcf, contributions = pairCorrelationFunction(pc, 'Celltype', [a,b], maxR=maxR,annulusStep=annulusStep,annulusWidth=annulusWidth)
        avals.append(a)
        bvals.append(b)
        pcfs.append(pcf)

sns.set(font_scale=2.4)
fig, ax = plt.subplots(nrows=len(pc.labels['Celltype']['categories']), ncols=len(pc.labels['Celltype']['categories']),sharex=True,sharey=True)
it = 0
for i,a in enumerate(pc.labels['Celltype']['categories']):
    for j,b in enumerate(pc.labels['Celltype']['categories']):
        ax[i,j].plot(r,pcfs[it],lw=5)
        if i == 5:
            ax[i,j].set_xlabel(b)
        if j == 0:
            ax[i,j].set_ylabel(a)
        ax[i,j].set_ylim([0,10])
        ax[i,j].axhline(1,linestyle=':',c='k',lw=3)
        it = it + 1


#%% Selected cross-PCFs - g_{ThE}
sns.set(font_scale=5)
a = 'T Helper Cell'
b = 'Epithelium'
maxR=150
annulusStep = 1
annulusWidth = 20
r, pcf, contributions = pairCorrelationFunction(pc, 'Celltype', [a,b], maxR=maxR,annulusStep=annulusStep,annulusWidth=annulusWidth)

plt.figure(figsize=(18,18))
plt.plot(r, pcf, lw=5)
plt.gca().axhline(1,c='k',linestyle=':',lw=3)
plt.xlabel('$r$ ($\mu$m)')
plt.ylabel('$g_{ThE}(r)$')




#%%Selected cross-PCFs - g_{ThM}
a = 'T Helper Cell'
b = 'Macrophage'

maxR=150
annulusStep = 1
annulusWidth = 20
r, pcf, contributions = pairCorrelationFunction(pc, 'Celltype', [a,b], maxR=maxR,annulusStep=annulusStep,annulusWidth=annulusWidth)

plt.figure(figsize=(18,18))
plt.plot(r, pcf, lw=5)
plt.gca().axhline(1,c='k',linestyle=':',lw=3)
plt.xlabel('$r$ ($\mu$m)')
plt.ylabel('$g_{ThM}(r)$')

#%% TCM example
typea = 'T Helper Cell'
typeb = 'Macrophage'

tcm = topographicalCorrelationMap(pc,'Celltype',typea,'Celltype',typeb,radiusOfInterest=50,maxCorrelationThreshold=5.0,kernelRadius=150,kernelSigma=50,visualiseStages=False)

#%
plt.figure(figsize=(20,20))
l = int(np.ceil(np.max(np.abs([tcm.min(),tcm.max()]))))
plt.imshow(tcm,cmap='RdBu_r',vmin=-l,vmax=l,origin='lower')
plt.colorbar(label='$\Gamma_{C_1 C_2}(r=50)$')
ax = plt.gca()
ax.grid(False)


#%% NCF example

maxR = 150
celltypes = ['T Helper Cell','Macrophage','Neutrophil']

ns = {v:np.sum(df.Celltype == v) for v in celltypes}
circles, triplets = neighbourhoodCorrelationFunction(pc,'Celltype',celltypes,maxR=maxR)


step = 5
bins=np.arange(0,maxR+step,step)
vals_observed, rs = np.histogram(circles,bins=bins)

redoBootstrap = False
if redoBootstrap:
    bootstrappedRadii = []
    nBootstrap = 100000000
    for i in range(nBootstrap):
        if i % 100000 == 0:
            print(i)
        temp = make_circle(1000*np.random.rand(3,2))
        bootstrappedRadii.append(temp[2])
        vals_bootstrap, rs = np.histogram(bootstrappedRadii,bins=bins)
        vals_bootstrap = vals_bootstrap/nBootstrap
else:

    if (step == 5) & ~redoBootstrap:
        vals_bootstrap = np.array([     6,     86,    372,    890,   1957,   3455,   5578,   8315,
                12064,  16432,  21767,  28480,  36075,  45048,  54542,  65789,
                78113,  90920, 105552, 121675, 138988, 157609, 177132, 198158,
               221789, 245811, 270577, 296563, 323350, 353548])/100000000
    elif (step == 10) & ~redoBootstrap:
        vals_bootstrap = np.array([    92,   1262,   5412,  13893,  28496,  50247,  81123, 120331,
               169033, 227227, 296597, 375290, 467600, 567140, 676898])/100000000
    else:
        print('Bootstrapping values not pre-calculated for this radial step value, set redoBoostrap=True')
        assert(1==2) # I'm not good at errors
    vals_bootstrap = ns[celltypes[0]]*ns[celltypes[1]]*ns[celltypes[2]]*vals_bootstrap

plt.figure(figsize=(18,18))
plt.plot(rs[1:],vals_bootstrap,label='Expectation (CSR)',lw=5)
plt.plot(rs[1:],vals_observed,label='Observation',lw=5)
plt.xlabel('$r$ ($\mu$m)')
plt.ylabel('Number')
plt.legend()

plt.figure(figsize=(18,18))
plt.plot(rs[1:],vals_observed/vals_bootstrap,lw=5)
plt.gca().axhline(1,c='k',linestyle=':',lw=3)
plt.xlabel('$r$ ($\mu$m)')
plt.ylabel('NCF$_{Th M N}(r)$')


#%% wPCF example
pc.addLabels('Epithelium','categorical',df.Epithelium,cmap='tab10')
pc.addLabels('CD4','continuous',df.CD4,cmap='Oranges')

pc.changeIndividualLabelColor('Epithelium', 'Positive', plt.cm.tab10(0))
pc.changeIndividualLabelColor('Epithelium', 'Negative', [0.8,0.8,0.8,1])
visualisePointCloud(pc,'Epithelium',markerSize=100)
visualisePointCloud(pc,'CD4',markerSize=50,vmax=24,cmap='Oranges')

def weightingFunction(p,l_B):
    weights = 1-np.abs(p-l_B)/2
    weights = np.maximum(weights, np.zeros(np.shape(weights)))
    return weights

targetP = np.arange(0,25,1)
w = [weightingFunction(5, v) for v in targetP]
plt.figure()
plt.plot(targetP,w)

# Values used in manuscript:
    # annulusStep = 1
    # annulusWidth = 20
    # I've changed them here so that if you're running this code you won't get bored and think it's broken
    # If you want to reproduce the exact plot from the paper, stick these values in and go and get a cup of tea (it doesn't take that long, but it's a lot more calculations)
annulusStep = 10
annulusWidth = 20

r, targetP, wPCF = weightedPairCorrelationFunction(pc, 'Epithelium', 'Positive', 'CD4',maxR=150,annulusStep=annulusStep,annulusWidth=annulusWidth,targetP=targetP,weightingFunction=weightingFunction)
plotWeightedPCF(r,targetP,wPCF,vmin=0,vmax=5,ax=None,cm='plasma')


# Plot the cross-PCF over the top of the cross-sections
maxR = 150
a = 'Epithelium'
b = 'T Helper Cell'
r, pcf, contributions = pairCorrelationFunction(pc, 'Celltype', [a,b], maxR=maxR,annulusStep=annulusStep,annulusWidth=annulusWidth)
plt.figure(figsize=(18,18))
plt.gca().axhline(1,c='k',linestyle=':',lw=5)
for v in np.linspace(0,targetP[-1],5):
    v = int(np.floor(v))
    plt.plot(r,wPCF[:,v],label=f'CD4 = {targetP[v]}',c=plt.cm.viridis(v/len(targetP)),lw=7)
plt.plot(r,pcf,c='r',linestyle='--',lw=7,label='$g_{ETh}(r)$')
plt.legend()
plt.xlabel('Radius ($r$)')
plt.ylabel('wPCF($r$, $E$, CD4)')







