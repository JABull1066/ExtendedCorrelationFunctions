import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import random
from helperFunctions import *
sns.set_style('white')
sns.set(font_scale=5)

# open a file, where you ant to store the data
with open('./ROI-0_MainText.pkl', 'rb') as file:
    # dump information to that file
    [img, names] = pickle.load(file)

#%% Visualise some of the channels
for i in range(8):
    plt.figure(figsize=(25,25))
    plt.imshow(img[:,:,[i]])
    plt.title(f'{names[i]}, (layer {i})')
    plt.colorbar()

plt.figure(figsize=(25,25))
plt.imshow(img[:,:,5],cmap='Greys_r',origin='lower')
plt.gca().grid(False)
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.colorbar(label='E-Cadherin intensity')

plt.figure(figsize=(25,25))
plt.imshow(img[:,:,6],cmap='Oranges',origin='lower')
plt.gca().grid(False)
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.colorbar(label='CD4 intensity')
#%% 
# Format image as a point cloud where each pixel is a point
[h,w,c] = np.shape(img)
X,Y = np.mgrid[0:h,0:w]
points = np.vstack([X.ravel(), Y.ravel()]).T
opal780 = img[:,:,5].ravel()
opal690 = img[:,:,2].ravel()
opal520 = img[:,:,6].ravel()
opal620 = img[:,:,4].ravel()

gap = 5 # Step to cut down on pixels - downsample to take every {gap} pixels
mask = [True if np.remainder(v[0],gap) + np.remainder(v[1],gap) == 0 else False for v in points]
points = points[mask,:]
opal780 = opal780[mask]
opal520 = opal520[mask]
opal620 = opal620[mask]
opal690 = opal690[mask]

pc = generatePointCloud('rawImage',points)
pc.addLabels('opal780_continuous','continuous',opal780)
pc.addLabels('opal690_continuous','continuous',opal690)
pc.addLabels('opal520_continuous','continuous',opal520)
pc.addLabels('opal620_continuous','continuous',opal620)

opal690 = np.asarray(['Positive' if v > 50 else 'Negative' for v in opal690])
opal780 = np.asarray(['Positive' if v > 15 else 'Negative' for v in opal780])
pc.addLabels('opal780','categorical',opal780)
pc.addLabels('opal690','categorical',opal690)

s = 50
visualisePointCloud(pc,'opal780_continuous',markerSize=s)
visualisePointCloud(pc,'opal780',markerSize=s)
visualisePointCloud(pc,'opal690_continuous',markerSize=s)
visualisePointCloud(pc,'opal690',markerSize=s)
visualisePointCloud(pc,'opal520_continuous',markerSize=s)
visualisePointCloud(pc,'opal620_continuous',markerSize=s)
# ms.pointcloud.visualise(pc,'opal520',markerSize=100)

#%% Visualise points for wPCF
plt.figure(figsize=(18,18))
plt.scatter(points[:,1],points[:,0],c=opal520,cmap='Oranges',s=15,vmax=255)

plt.figure(figsize=(18,18))
plt.scatter(points[:,1],points[:,0],c=opal520,cmap='Oranges',s=180,vmax=255)
plt.xlim([0,350])
plt.ylim([0,350])

plt.figure(figsize=(18,18))
m = opal780 == 'Positive'
plt.scatter(points[m,1],points[m,0],c=plt.cm.tab10(0),s=15)

plt.figure(figsize=(18,18))
plt.scatter(points[m,1],points[m,0],c=plt.cm.tab10(0),s=180)
plt.xlim([0,350])
plt.ylim([0,350])

#%% Calculate wPCF example
def weightingFunction(p,l_B):
    weights = 1-np.abs(p-l_B)/20
    weights = np.maximum(weights, np.zeros(np.shape(weights)))
    return weights

w = [weightingFunction(100, v) for v in np.arange(0,256,10)]
plt.figure(figsize=(18,18))
plt.plot(np.arange(0,256,10),w)
plt.xlabel('Opal 520 pixel intensity')
plt.ylabel('$w(100,m)$')

r, targetP, wPCF = weightedPairCorrelationFunction(pc, 'opal780', 'Positive', 'opal520_continuous',maxR=150,annulusStep=1,annulusWidth=20,targetP=np.arange(0,256,10),weightingFunction=weightingFunction)

plotWeightedPCF(r,targetP,wPCF,vmin=0,vmax=5,ax=None,cm='plasma')
plt.xlabel('Radius ($r$) mm')
plt.ylabel('Opal 520 pixel intensity')
#%%
plt.figure(figsize=(18,18))
plt.gca().axhline(1,c='k',linestyle=':',lw=5)
for v in np.arange(0,len(targetP),5):
    plt.plot(r,wPCF[:,v],label=f'Opal 520 = {targetP[v]}',c=plt.cm.viridis(v/len(targetP)),lw=7)
plt.xlabel('Radius (r)')
plt.ylabel('wPCF($r$, Opal780+, Opal 520 intensity)')

# Precalculated cross-PCF from other script
pcf = np.array([[0.20429162, 0.22399941, 0.24010333, 0.25625299, 0.27236   ,
        0.29171182, 0.31463267, 0.33475687, 0.34921048, 0.37726578,
        0.39739166, 0.41121297, 0.42588562, 0.44879098, 0.46668775,
        0.48730773, 0.50644911, 0.52425707, 0.53535951, 0.54828698,
        0.56579706, 0.57886946, 0.59137167, 0.60662379, 0.62132189,
        0.63224445, 0.63854459, 0.6468266 , 0.66068365, 0.66589117,
        0.67403342, 0.68863575, 0.69845456, 0.70670399, 0.71445507,
        0.71865879, 0.72695555, 0.73438324, 0.74831405, 0.75561609,
        0.76117898, 0.7727954 , 0.78788214, 0.7925001 , 0.79646984,
        0.80643062, 0.82224841, 0.83095756, 0.83610324, 0.84595257,
        0.85946499, 0.86549036, 0.87719453, 0.88302217, 0.8939241 ,
        0.90613212, 0.91177284, 0.92189585, 0.92685154, 0.93801303,
        0.94783566, 0.95467844, 0.95843387, 0.96747524, 0.97801698,
        0.98577862, 0.99035913, 0.99669481, 1.00364678, 1.0102888 ,
        1.01019283, 1.01540835, 1.02098542, 1.02923899, 1.03217602,
        1.03644155, 1.04294502, 1.04572668, 1.05101171, 1.05481911,
        1.05717736, 1.05746795, 1.05848478, 1.0638925 , 1.06916618,
        1.06891695, 1.07074238, 1.07427882, 1.07408042, 1.07608619,
        1.08122714, 1.08360769, 1.08506193, 1.08727362, 1.09024254,
        1.09212946, 1.09287076, 1.09307599, 1.09308764, 1.09299525,
        1.09453158, 1.10086845, 1.10189787, 1.10163299, 1.10288117,
        1.10378342, 1.10343622, 1.10354607, 1.10651554, 1.10725975,
        1.10760118, 1.10905861, 1.11254872, 1.11097115, 1.11006967,
        1.10822749, 1.10870717, 1.11067936, 1.11174225, 1.11669595,
        1.11400388, 1.1116594 , 1.11564804, 1.11615543, 1.11116315,
        1.11285759, 1.11326792, 1.11623673, 1.11670278, 1.11699024,
        1.11799444, 1.11618197, 1.11500768, 1.11747422, 1.11728253,
        1.12016528, 1.12015596, 1.12027903, 1.11935979, 1.11693191,
        1.12138176, 1.12213556, 1.11944196, 1.11970962, 1.12204672,
        1.12259548, 1.12576367, 1.12535695, 1.12446693, 1.12641286,
        1.12774177]])
newr = np.arange(0,151,1)
plt.plot(newr,pcf.T,c='r',linestyle='--',lw=7,label='$g_{ETh}(r)$')

plt.legend()
