from helperFunctions import *
import numpy as np
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt

sns.set(font_scale=5)
#%% Make synthetic data which correlates pairwise but not three-wise
np.random.seed(1536)
clusterCentres = np.asarray([[0.2,0.2],[0.5,0.8],[0.8,0.2]])
labelsPresent = [['$C_1$','$C_2$'],['$C_1$','$C_3$'],['$C_2$','$C_3$']]

points = np.empty(shape=(0,2))
labels = []

# background noise
nBackground = 0
for l in ['$C_1$','$C_2$','$C_3$']:
    points = np.vstack([points, np.random.rand(nBackground,2)])
    labels.extend([l for v in range(nBackground)])

sigma = 0.05
nPoints = 25
for j in range(3):
    mu = clusterCentres[j]
    l = labelsPresent[j]
    points = np.vstack([points, mu + sigma*np.random.randn(2*nPoints,2)])
    labels.extend([l[0] for v in range(nPoints)])
    labels.extend([l[1] for v in range(nPoints)])

    
pc_pairwise = generatePointCloud('Pairwise Correlation Only', points*1000,domain=([[0,1000],[0,1000]]))
pc_pairwise.addLabels('Celltype', 'categorical', labels)



#%% Make synthetic data which correlates pairwise AND three-wise in the same way
clusterCentres = np.asarray([[0.2,0.2],[0.5,0.8],[0.8,0.2]])
labelsPresent = [['$C_1$','$C_2$','$C_3$'],['$C_1$','$C_2$','$C_3$'],['$C_1$','$C_2$','$C_3$']]

points = np.empty(shape=(0,2))
labels = []

# background noise
nBackground = 0
for l in ['$C_1$','$C_2$','$C_3$']:
    points = np.vstack([points, np.random.rand(nBackground,2)])
    labels.extend([l for v in range(nBackground)])

sigma = 0.05
nPoints = 25
for j in range(3):
    mu = clusterCentres[j]
    l = labelsPresent[j]
    points = np.vstack([points, mu + sigma*np.random.randn(3*nPoints,2)])
    labels.extend([l[0] for v in range(nPoints)])
    labels.extend([l[1] for v in range(nPoints)])
    labels.extend([l[2] for v in range(nPoints)])

    
pc_threewise = generatePointCloud('Three-way correlation', points*1000,domain=([[0,1000],[0,1000]]))
pc_threewise.addLabels('Celltype', 'categorical', labels)


#%% Visualise point clouds
visualisePointCloud(pc_pairwise,'Celltype',markerSize=100)
visualisePointCloud(pc_threewise,'Celltype',markerSize=100)

#%% Synthetic Dataset II - Pairwise cross-PCFs

for pc in [pc_pairwise, pc_threewise]:
    plt.figure(figsize=(20,20))
    plt.gca().axhline(1,c='k',linestyle=':',lw=4)
    for cellpairs in [['$C_1$','$C_2$'],['$C_1$','$C_3$'],['$C_2$','$C_3$']]:
        maxR = 1000
        annulusStep = 10
        annulusWidth = 50
        r, pcf, contributions = pairCorrelationFunction(pc, 'Celltype', cellpairs, maxR=maxR,annulusStep=annulusStep,annulusWidth=annulusWidth)
        label = '$g_{C_'+cellpairs[0][3]+' C_'+cellpairs[1][3]+'}(r)$'
        plt.plot(r,pcf,lw=5,label=label)
        plt.xlabel('Radius, $r$ ($\mu$m)')
        plt.ylim([0,15])
        plt.xlim([0,1000])
    plt.title(pc.name)
    plt.legend()
    
#%% Synthetic Dataset II - Neighbourhood correlation functions
maxR = 300

for pc in [pc_pairwise, pc_threewise]:
    circles, triplets = neighbourhoodCorrelationFunction(pc,'Celltype',['$C_1$','$C_2$','$C_3$'],maxR=maxR)
    order = np.arange(len(circles))
    np.random.shuffle(order)
        
    
    drawCircles = False
    if drawCircles:
        visualisePointCloud(pc,'Celltype',markerSize=100)
        for i in range(len(order)):
            circle = circles[order[i]]
            col = plt.cm.plasma(circle[2]/maxR)
            ec = [v for v in col]
            ec[3] = 0.25
            circle = plt.Circle((circle[0], circle[1]), circle[2], ec=col, fc=[0,0,0,0],zorder=-1)
            plt.gca().add_patch(circle)
            
        plt.gca().axis('equal')
         
    circles = np.asarray(circles)
    
    # Use bootstrapping to get predicted number of circles under CSR
    nA = 2*nPoints + nBackground
    nB = 2*nPoints + nBackground
    nC = 2*nPoints + nBackground
    
    redoBoostrap = False
    # Set this flag to true if you want to regenerate the distribution under CSR 
    # Otherwise we load in precalculated values for speed
    if redoBoostrap:
        bootstrappedRadii = []
        nBootstrap = 1000000
        for i in range(nBootstrap):
            if i % 10000 == 0:
                print(i)
            points_temp = np.random.rand(3,2)
            pc_temp = generatePointCloud('temp',points_temp,domain=[[0,1],[0,1]])
            pc_temp.addLabels('Celltype','categorical',['$C_1$','$C_2$','$C_3$'])
            circles_temp, triplets_temp = neighbourhoodCorrelationFunction(pc_temp,'Celltype',['$C_1$','$C_2$','$C_3$'],maxR=2)
            bootstrappedRadii.append(circles_temp[0][2])
        bootstrappedRadii = [v*1000 for v in bootstrappedRadii] # As this was generated in mm, not mu m
            
        vals_bootstrap, rs = np.histogram(bootstrappedRadii,bins=bins)
        vals_bootstrap = nA*nB*nC*vals_bootstrap/nBootstrap
    
    else:
        vals_observed, rs = np.histogram(circles,bins=bins)
        vals_bootstrap = np.asarray([0.375,1.375,4.875,17.625,39.25,65.875,103.75,143.75,208.875,276.5,364.625,464,578.25,713.75,856.125,997.625,1151.5,1325.88,1489.88,1684.62,1875.88,2075.88,2280.38,2495.5,2682.62,2887.62,3072.5,3305.75,3487.88,3617])
    
    plt.figure(figsize=(18,18))
    plt.plot(rs[1:],vals_bootstrap,label='Expectation (CSR)',lw=5)
    plt.plot(rs[1:],vals_observed,label='Observation',lw=5)
    plt.xlabel('$r$ (mm)')
    plt.ylabel('Number')
    plt.legend()
    plt.title(pc.name)
    
    plt.figure(figsize=(18,18))
    plt.plot(rs[1:],vals_observed/vals_bootstrap,lw=5)
    plt.gca().axhline(1,c='k',linestyle=':',lw=5)
    plt.xlabel('$r$ ($\mu$m)')
    plt.ylabel('NCF$_{C_1 C_2 C_3}(r)$')
    plt.title(pc.name)
