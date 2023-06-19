import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns
import random
from shapely.geometry import Polygon
from scipy.spatial.distance import cdist


class pointcloud:
    def __init__(self, 
    name, 
    points,
    domain=None, 
    unitOfLength=None
    ):
        self.name = name
        self.points = points #todo ensure points are an (n,d) numpy array for d = 2 or 3
        self.nPoints = np.shape(points)[0]
        self.dimension = np.shape(points)[1]       
        self.labels = {}
        self.nLabels = 0
        self.nLabels_categorical = 0
        self.nLabels_continuous = 0
        
        self.summaryStatistics = None

        if domain is None:
            # We estimate domain size by taking most extreme values and rounding up to nearest 1 unit
            maxDomain = np.ceil(np.max(self.points,axis=0))
            minDomain = np.floor(np.min(self.points,axis=0))
            self.domain = np.stack((minDomain,maxDomain)).transpose()
        else:
            self.domain = np.asarray(domain)
            if np.shape(self.domain)[0] != self.dimension:
                raise RuntimeError('Specified domain should be nDimensions by 2, specifying domain min and max values in each dimension')
            if np.shape(self.domain)[1] != 2:
                raise RuntimeError('Specified domain should be nDimensions by 2, specifying domain min and max values in each dimension')
            for d in range(self.dimension):
                if self.domain[d,0] >= self.domain[d,1]:
                    raise RuntimeError(f'Specified domain minimum value in dimension {d} ({self.domain[d,0]}) must be lower than maximum value ({self.domain[d,1]})')
        if self.dimension == 2:
            v = [[self.domain[0,0],self.domain[1,0]],
                 [self.domain[0,0],self.domain[1,1]],
                 [self.domain[0,1],self.domain[1,1]],
                 [self.domain[0,1],self.domain[1,0]]]
            self.boundaryPolygon = Polygon(v)
        self.domainVolume = np.prod(self.domain[:,1] - self.domain[:,0],axis=0)
        self.density = self.nPoints / self.domainVolume
        

    def __str__(self):
        return f"Name: {self.name}, nPoints: {self.nPoints}"

    def addLabels(self, labelName, labelType, labels,cmap=None):
        if self.nPoints != len(labels):
            raise ValueError(f"Expected a list of {self.nPoints} labels, received {len(labels)}")
        if labelType in ['categorical','continuous']:
            self.labels[labelName] = {'Type':labelType,
                                        'labels':labels}
            self.nLabels = self.nLabels + 1
            
            if labelType == 'categorical':
                self.nLabels_categorical = self.nLabels_categorical + 1
                unique = np.unique(labels)
                labelToInteger = {unique[v] : v for v in range(len(unique))}
                self.labels[labelName]['categories'] = unique
                self.labels[labelName]['nCategories'] = len(unique)
                self.labels[labelName]['labelToInteger'] = labelToInteger
                self.labels[labelName]['integerToLabel'] = {labelToInteger[v] : v for v in labelToInteger.keys()}
                self.labels[labelName]['numericalLabels'] = np.asarray([labelToInteger[labels[v]] for v in range(len(labels))])
                if cmap is None:
                    # Use default colormap
                    cmap = 'tab10'
                self.labels[labelName]['integerToColor'] = {v: plt.cm.get_cmap(cmap)(v) for v in self.labels[labelName]['integerToLabel'].keys()}
                colArray = np.asarray([v for v in self.labels[labelName]['integerToColor'].values()])
                self.labels[labelName]['cmap'] = ListedColormap(colArray)
                
            else:
                self.nLabels_continuous = self.nLabels_continuous + 1
                self.labels[labelName]['numericalLabels'] = np.asarray(labels)
                self.labels[labelName]['cmap'] = 'plasma'
        else:
            raise ValueError('labelType must be categorical or continuous')

    def changeIndividualLabelColor(self, labelName, labelToUpdate, newColor):
        assert(len(newColor) == 4)
        labelIntegerValue = self.labels[labelName]['labelToInteger'][labelToUpdate]
        self.labels[labelName]['integerToColor'][labelIntegerValue] = newColor
        # Update cmap
        colArray = np.asarray([v for v in self.labels[labelName]['integerToColor'].values()])
        self.labels[labelName]['cmap'] = ListedColormap(colArray)
        
def generatePointCloud(name, points,domain=None,unitOfLength=None):
    pc = pointcloud(name, points, domain, unitOfLength)
    return pc

def visualisePointCloud(pc,labelForVisualisation=None,cmap=None,markerSize=None,vmin=None,vmax=None):
    from matplotlib import colors
    if pc.dimension != 2:
        raise RuntimeError('Visualisation currently only possible in 2D')
    if labelForVisualisation not in pc.labels.keys() and labelForVisualisation != None:
        raise ValueError('labelForVisualisation must be a label!')

    shuffleOrder = np.arange(len(pc.points))
    random.shuffle(shuffleOrder)

    if markerSize is None:
        markerSize = 20
    else:
        if isinstance(markerSize, (int,float)):
            if markerSize < 0:
                raise ValueError('markerSize must be a positive number')

    plt.figure(figsize=(24,18))
    if labelForVisualisation is None:
        plt.scatter(pc.points[shuffleOrder,0],pc.points[shuffleOrder,1],s=markerSize,cmap=cmap)
    else:
        norm = None
        if cmap is None:
            cmap = pc.labels[labelForVisualisation]['cmap']
        labelType = pc.labels[labelForVisualisation]['Type']
        if labelType == 'categorical':
            nCategories = pc.labels[labelForVisualisation]['nCategories']
            # if cmap is None:
            #     cmap = pc.labels[labelForVisualisation]['cmap']
            cmap = plt.cm.get_cmap(cmap,nCategories)
            norm = colors.BoundaryNorm(np.arange(-0.5, nCategories+0.5, 1), cmap.N)
        plt.scatter(pc.points[shuffleOrder,0],pc.points[shuffleOrder,1],c=pc.labels[labelForVisualisation]['numericalLabels'][shuffleOrder],s=markerSize,cmap=cmap,norm=norm,vmin=vmin,vmax=vmax)
    plt.gca().axis('equal')
    plt.xlim(pc.domain[0])
    plt.ylim(pc.domain[1])
    if labelForVisualisation != None:
        cbar=plt.colorbar(label=labelForVisualisation)
        if labelType == 'categorical':
            cbar.set_ticks(list(pc.labels[labelForVisualisation]['labelToInteger'].values()))
            cbar.set_ticklabels(list(pc.labels[labelForVisualisation]['labelToInteger'].keys()))
    plt.tight_layout()
    return plt.gcf(), plt.gca()

def pairCorrelationFunction(pc,labelName,categoriesToPlot,maxR=0.5,annulusStep=0.025,annulusWidth=0.025):
# First we check that the chosen label is categorical
    labelType = pc.labels[labelName]['Type']
    if labelType != 'categorical':
        raise RuntimeError(f'The label {labelName} is not a categorical label.')
    categories = pc.labels[labelName]['categories']
    labelA = categoriesToPlot[0]
    labelB = categoriesToPlot[1]
    if labelA not in categories:
        raise RuntimeError(f'The category {labelA} is not associated with the label {labelName}.')
    if labelB not in categories:
        raise RuntimeError(f'The category {labelB} is not associated with the label {labelName}.')

    
    i_A = pc.labels[labelName]['labelToInteger'][labelA]
    i_B = pc.labels[labelName]['labelToInteger'][labelB]
    
    # Points to include A: All points within pc.domain
    # Points to include B: All points within pc.domain
    p_A = pc.points[pc.labels[labelName]['numericalLabels'] == i_A,:]
    p_B = pc.points[pc.labels[labelName]['numericalLabels'] == i_B,:]
    if np.shape(p_A)[0] == 0:
        raise RuntimeError(f'No cells with {labelA} found within PCF domain')
    if np.shape(p_B)[0] == 0:
        raise RuntimeError(f'No cells with {labelB} found within PCF domain')
    # Get annulus areas (within domain) around p_A
    areas_A = getAnnulusAreasAroundPoints(p_A, maxR, annulusStep,annulusWidth, pc.domain)
    density_B = np.shape(p_B)[0]/pc.domainVolume
    
    distances_AtoB = cdist(p_A, p_B, metric='euclidean')
    radii, g, contributions = crossPCF(distances_AtoB, areas_A, density_B, maxR, annulusStep, annulusWidth)

    return radii, g, contributions

def weightedPairCorrelationFunction(pc,categoricalLabelName,categoricalLabelToPlot,continuousLabelName,maxR=0.5,annulusWidth=0.025,annulusStep=0.025,targetP=None,weightingFunction=None):
    # First we check that the chosen label is categorical
    labelTypeA = pc.labels[categoricalLabelName]['Type']
    if labelTypeA != 'categorical':
        raise RuntimeError(f'The label {labelTypeA} is not a categorical label.')
    labelTypeB = pc.labels[continuousLabelName]['Type']
    if labelTypeB != 'continuous':
        raise RuntimeError(f'The label {labelTypeB} is not a continuous label.')
    categoriesA = pc.labels[categoricalLabelName]['categories']
    labelA = categoricalLabelToPlot
    if labelA not in categoriesA:
        raise RuntimeError(f'The category {labelA} is not associated with the label {categoricalLabelName}.')

    # Now calculate wPCF
    i_A = pc.labels[categoricalLabelName]['labelToInteger'][labelA]
    p_A = pc.points[pc.labels[categoricalLabelName]['numericalLabels'] == i_A,:]

    # Get all points with a valid (i.e., not a nan) value for continuousLabelName
    p_B = pc.points[~np.isnan(pc.labels[continuousLabelName]['numericalLabels']),:]
    l_B = pc.labels[continuousLabelName]['numericalLabels'][~np.isnan(pc.labels[continuousLabelName]['numericalLabels'])]
    
    if targetP is None:
        targetP = np.linspace(np.min(l_B),np.max(l_B),101)
    if weightingFunction is None:
        def weightingFunction(p,l_B):
            weights = 1-5*np.abs(p-l_B)
            weights = np.maximum(weights, np.zeros(np.shape(weights)))
            return weights

    # Get annulus areas (within domain) around p_A
    if np.shape(p_A)[0]>0:
        areas_A = getAnnulusAreasAroundPoints(p_A, maxR, annulusStep, annulusWidth, pc.domain)

    PCF_radii_lower = np.arange(0, maxR+annulusStep, annulusStep)
    PCF_radii_upper = np.arange(annulusWidth, maxR + annulusWidth + annulusStep, annulusStep)
    
    wPCF = np.ones(shape=(len(PCF_radii_lower),len(targetP)))
    distances_AtoB = cdist(p_A, p_B, metric='euclidean')
    N_A = np.shape(p_A)[0]
    N_B = np.shape(p_B)[0]
    for ind_j, targetP_j in enumerate(targetP):
        weights_j = weightingFunction(targetP_j, l_B)
    
        totalWeight_j = sum(weights_j) # W_Y
        density_j = N_B / pc.domainVolume # N_Y / A
    
        wPCF_row = np.zeros(shape=(len(PCF_radii_lower)))
        
        for annulus in range(len(PCF_radii_lower)):
            inner = PCF_radii_lower[annulus]
            outer = PCF_radii_upper[annulus]
    
            # Find pairwise distances within this radius
            distanceMask = np.logical_and((distances_AtoB >= inner),(distances_AtoB < outer))
            for i in range(N_A):
                fillIndices = np.where(distanceMask[i,:])[0]
                m_i = sum(weights_j[fillIndices]) # m_i
                wPCF_row[annulus] = wPCF_row[annulus] + m_i*N_B/(totalWeight_j*density_j*areas_A[i,annulus])
    
        wPCF[:,ind_j] = wPCF_row / N_A
    return PCF_radii_lower, targetP, wPCF

def neighbourhoodCorrelationFunction(pc,labelName,categoriesToPlot,maxR=0.5):
# First we check that the chosen label is categorical
    if pc.dimension != 2:
        raise NotImplementedError('Currently only implemented for 2D point clouds')

    labelType = pc.labels[labelName]['Type']
    if labelType != 'categorical':
        raise RuntimeError(f'The label {labelName} is not a categorical label.')
    categories = pc.labels[labelName]['categories']

    for category in categoriesToPlot:
        if category not in categories:
            raise RuntimeError(f'The category {category} is not associated with the label {labelName}.')
    
    # idea: For each combination of cells from the different categories, calculate the radius of the smallest enclosing circle
    nCategories = len(categoriesToPlot)
    assert(nCategories > 1)
    pointsToCompare = []
    for category in categoriesToPlot:
        i = pc.labels[labelName]['labelToInteger'][category]
        p = pc.points[pc.labels[labelName]['numericalLabels'] == i,:]
        pointsToCompare.append(p)

    # pointsToCompare contains nCategories lists of nPointsCategoryX x 2 points
    # We want to get all possible combinations of 1 point from each list, such that no two elements are more than 2*maxR apart

    #TODO this assumes n = 3
    assert(nCategories == 3)
    # Prefiltering
    AB = cdist(pointsToCompare[0],pointsToCompare[1]) < maxR*2
    AC = cdist(pointsToCompare[0],pointsToCompare[2]) < maxR*2
    BC = cdist(pointsToCompare[1],pointsToCompare[2]) < maxR*2

    
    ab_candidates = np.where(AB)
    allCandidates = []
    for i in range(np.shape(ab_candidates)[1]):
        target = [ab_candidates[0][i], ab_candidates[1][i]]
        # Check this against BC and AC to see if we have a candidate
        successes = np.where(AC[target[0],:] & BC[target[1],:])[0]
        if len(successes) > 0:
            allCandidates.extend([[target[0],target[1],v] for v in successes])
    allCandidates = np.asarray(allCandidates)
    # allCandidates is now a nCandidates x 3 array, where nCandidates is the number of triplets which are pairwise within a distance of 2*maxR
    # For each triplet, get the points and calculate the smallest enclosing circle
    from smallestEnclosingCircle import make_circle
    nCandidates = np.shape(allCandidates)[0]
    circles = []
    triplets = []
    for i in range(nCandidates):
        (center_x, center_y, radius) = make_circle([pointsToCompare[v][allCandidates[i,v]] for v in range(nCategories)])
        if radius < maxR:
            circles.append([center_x, center_y, radius])
            triplets.append([pointsToCompare[v][allCandidates[i,v]] for v in range(nCategories)])

    return circles, triplets

def plotWeightedPCF(radii,targetP,wPCF,vmin=0,vmax=8,ax=None,cm='plasma'):
    equalColormap = True
    
    if ax == None:
        plt.figure(figsize=(18,18))
        ax = plt.gca()
        fig = plt.gcf()
    if equalColormap:
        # sample the colormaps that you want to use. Use 128 from each so we get 256
        # colors in total
    
        one = 1/vmax
        nCols = 1000
        threshold = one*nCols
        #print(threshold)
        colors1 = plt.cm.Greens(np.linspace(0, 1, round(threshold)))
        map = plt.cm.get_cmap(cm)
        colors2 = map(np.linspace(0, 1, nCols - round(threshold)))
    
        # combine them and build a new colormap
        colors = np.vstack((colors1, colors2))
        cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        
    else:
        cmap = 'inferno'
    ax.imshow(wPCF.transpose(),origin='lower',extent=[0,1,0,1],cmap=cmap,vmin=vmin,vmax=vmax)
    
    ax.set_xlabel('Radius $(r)$')
    tickProps = [0,0.25,0.5,0.75,1]
    ax.set_xticks(tickProps)
    ax.set_xticklabels([v*(radii[-1]) for v in tickProps])
    ax.set_ylabel('$P$')
    tickProps = [0,0.25,0.5,0.75,1]
    ax.set_yticks(tickProps)
    ax.set_yticklabels([v*(targetP[-1]) for v in tickProps])
    return fig, ax

def topographicalCorrelationMap(pc,labelNameA,labelA,labelNameB,labelB,radiusOfInterest=0.1,maxCorrelationThreshold=5.0,kernelRadius=150,kernelSigma=50,visualiseStages=False):
    
    for labelName in [labelNameA,labelNameB]:
        labelType = pc.labels[labelName]['Type']
        if labelType != 'categorical':
            raise RuntimeError(f'The label {labelName} is not a categorical label.')
    
    if labelA not in pc.labels[labelNameA]['categories']:
        raise RuntimeError(f'The category {labelA} is not associated with the label {labelNameA}.')
    if labelB not in pc.labels[labelNameB]['categories']:
        raise RuntimeError(f'The category {labelB} is not associated with the label {labelNameB}.')

    i_A = pc.labels[labelNameA]['labelToInteger'][labelA]
    i_B = pc.labels[labelNameB]['labelToInteger'][labelB]

    # Points to include A: All points within pc.domain
    # Points to include B: All points within pc.domain
    p_A = pc.points[pc.labels[labelNameA]['numericalLabels'] == i_A,:]
    p_B = pc.points[pc.labels[labelNameB]['numericalLabels'] == i_B,:]

    # Get areas around A, calculate pairwise A-B distances
    areas = []
    for i in range(len(p_A)):
        area = returnAreaOfCircleInDomainAroundPoint(i,p_A,radiusOfInterest,pc.domain[0],pc.domain[1])
        areas.append(area)
    density_B = np.shape(p_B)[0]/pc.domainVolume
    areas_A = np.asarray(areas)
    distances_AtoB = cdist(p_A, p_B, metric='euclidean')
    contributions = distances_AtoB <= radiusOfInterest
    
    BnearA_observed = np.sum(contributions,axis=1)/areas_A # observed per unit area
    marks = BnearA_observed/density_B
    
    
    if visualiseStages:
        s=100
        plt.figure(figsize=(20,20))
        plt.scatter(p_A[:,0],p_A[:,1],c=marks,cmap='viridis',s=s)
        plt.colorbar()
        plt.gca().axis('equal')
        
    # Map PCF interpretation to [-1,1]
    minCorrelationThreshold = 1/maxCorrelationThreshold
    
    transformedMarks = np.copy(marks)
    transformedMarks[transformedMarks < minCorrelationThreshold] = minCorrelationThreshold
    transformedMarks[transformedMarks > maxCorrelationThreshold] = maxCorrelationThreshold
    
    transformedMarks[transformedMarks<1] = -1/transformedMarks[transformedMarks<1]
    # That gives us values in [-maxPCFthreshold,-1] U [1,maxPCFthreshold]
    # Now map to [-1,1]
    transformedMarks[transformedMarks<0] = (transformedMarks[transformedMarks<0]+1)/(maxCorrelationThreshold-1)
    transformedMarks[transformedMarks>0] = (transformedMarks[transformedMarks>0]-1)/(maxCorrelationThreshold-1)
    
    if visualiseStages:
        plt.figure(figsize=(20,20))
        plt.scatter(p_A[:,0],p_A[:,1],c=transformedMarks,cmap='RdBu_r',vmin=-1,vmax=1,s=s)
        plt.colorbar()
        plt.gca().axis('equal')
                          
                        
    x, y = np.meshgrid(np.arange(-kernelRadius,kernelRadius+0.1,1),np.arange(-kernelRadius,kernelRadius+0.1,1))
    dst = np.sqrt(x*x + y*y)
    kernel = np.exp(-( dst**2 / ( 2.0 * kernelSigma**2 ) ) )
    
    xrange = [int(pc.domain[0][0])-kernelRadius, int(pc.domain[0][1])+1+kernelRadius]
    yrange = [int(pc.domain[1][0])-kernelRadius, int(pc.domain[1][1])+1+kernelRadius]
    heatmap = np.zeros(shape=(xrange[1]-xrange[0],yrange[1]-yrange[0]))
    
    def addWeightedContribution(heatmap, weight, coordinate, xrange, yrange, kernel,kernelRadius):
        x0 = int(coordinate[0]) - kernelRadius - xrange[0]
        x1 = x0 + 2*kernelRadius + 1
        y0 = int(coordinate[1]) - kernelRadius - yrange[0]
        y1 = y0 + 2*kernelRadius + 1
        heatmap[x0:x1,y0:y1] = heatmap[x0:x1,y0:y1] + kernel*weight
        return heatmap
    
    for i in range(len(p_A)):
        coordinate = p_A[i,:]
        weight = transformedMarks[i]
        heatmap = addWeightedContribution(heatmap, weight, coordinate, xrange, yrange, kernel, kernelRadius)
    
    topographicalCorrelationMap = heatmap[kernelRadius:-kernelRadius,kernelRadius:-kernelRadius]

    if visualiseStages:
        l = int(np.ceil(np.max([topographicalCorrelationMap.min(),topographicalCorrelationMap.max()])))
        fig, ax = plotTopographicalCorrelationMap(pc,topographicalCorrelationMap.T,ax=None,cmap='RdBu_r',colorbarLimit=l)
    
    return topographicalCorrelationMap.T

def plotTopographicalCorrelationMap(pc,topographicalCorrelationMap,ax=None,cmap='RdBu_r',colorbarLimit=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    if ax == None:
        plt.figure(figsize=(18,18))
        ax = plt.gca()
    if colorbarLimit == None:
        colorbarLimit = int(np.ceil(np.max([topographicalCorrelationMap.min(),topographicalCorrelationMap.max()])))
    extent = [pc.domain[0,0],pc.domain[0,1],pc.domain[1,0],pc.domain[1,1]]
    im = ax.imshow(topographicalCorrelationMap,origin='lower',cmap=cmap,extent=extent,vmin=-colorbarLimit,vmax=colorbarLimit)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.gcf().colorbar(im, cax=cax, orientation='vertical')
    return plt.gcf(), plt.gca()










#%% Helper functions
# Used in quadratCorrelationMatrix
def getCFromSigma_inv(Sigma_inv):
    C = np.zeros(np.shape(Sigma_inv))
    for i in range(len(Sigma_inv)):
        for j in range(len(Sigma_inv)):
            C[i,j] = -Sigma_inv[i,j] / np.sqrt(Sigma_inv[i,i] * Sigma_inv[j,j])
    return C

# Used in quadratCorrelationMatrix
def changeSomeElements(matrix):
    
    # Select elements of a submatrix (a b; c d) such that elements in the same row/column are from the same row/column in matrix
    n, m = np.shape(matrix)
    rows = random.sample(range(n), 2)
    cols = random.sample(range(m), 2)
    a = matrix[rows[0],cols[0]]
    b = matrix[rows[0],cols[1]]
    c = matrix[rows[1],cols[0]]
    d = matrix[rows[1],cols[1]]
    
    # Now find the smallest values on the diagonals
    minDiag1 = min(a,d)
    minDiag2 = min(b,c)
    if minDiag1 == 0 and minDiag2 == 0:
        return matrix,False
    else:
        # At least one diagonal doesn't include 0. We subtract from that diagonal. wlog pick diag1 if both are fine
        if minDiag1 > 0:
            if minDiag1 == 1:
                k = 1
            else:
                # Choose k between 1 and minDiag1
                k = np.random.randint(1,minDiag1+1)
            new_a = a - k
            new_d = d - k
            new_b = b + k
            new_c = c + k
        else:
            if minDiag2 == 1:
                k = 1
            else:
                # Choose k between 1 and minDiag2
                k = np.random.randint(1,minDiag2+1)
            new_a = a + k
            new_d = d + k
            new_b = b - k
            new_c = c - k
        #print(rows, cols)
        matrix[rows[0],cols[0]] = new_a
        matrix[rows[0],cols[1]] = new_b
        matrix[rows[1],cols[0]] = new_c
        matrix[rows[1],cols[1]] = new_d
        return matrix,True


def crossPCF(distances_AtoB, areas_A, density_B, maxR, annulusStep, annulusWidth):
    N_A = np.shape(distances_AtoB)[0]

    PCF_radii_lower = np.arange(0, maxR + annulusStep, annulusStep)
    PCF_radii_upper = np.arange(annulusWidth, maxR + annulusStep + annulusWidth, annulusStep)

    crossPCF_AtoB = np.ones(shape=(len(PCF_radii_lower),1))
    contributions = np.zeros(shape=(N_A,len(PCF_radii_lower)))
    for annulus in range(len(PCF_radii_lower)):
        inner = PCF_radii_lower[annulus]
        outer = PCF_radii_upper[annulus]

        # Find pairwise distances within this radius
        distanceMask = np.logical_and((distances_AtoB > inner),(distances_AtoB <= outer))
        for i in range(N_A):
            # For each point in pA
            # Find pairwise distances to points in pB within this radius
            fillIndices = np.where(distanceMask[i,:])[0]
            contribution = len(fillIndices)/(density_B*areas_A[i,annulus])
            crossPCF_AtoB[annulus] = crossPCF_AtoB[annulus] + contribution
            contributions[i,annulus] = contributions[i,annulus] + contribution
        crossPCF_AtoB[annulus] = crossPCF_AtoB[annulus] / N_A
    return PCF_radii_lower, crossPCF_AtoB, contributions

def getAnnulusAreasAroundPoints(points_i, maxR, annulusStep, annulusWidth, domain):
    # We want to populate a table the same size as distances, which contains the area of the annulus containing that contribution
    # i.e., "at distance D(i->j) from point i, what is area of containing annulus?"
    domainX = domain[0,:]
    domainY = domain[1,:]
    vfunc_returnAreaOfCircleInDomainAroundPoint = np.vectorize(returnAreaOfCircleInDomainAroundPoint,excluded=['points','domainX','domainY'])
    PCF_radii_lower = np.arange(0, maxR+annulusStep, annulusStep)
    PCF_radii_upper = np.arange(annulusWidth, maxR + annulusWidth + annulusStep, annulusStep)
    # PCF_radii_lower = np.arange(0, maxR, dr)
    # PCF_radii_upper = np.arange(dr, maxR + dr, dr)

    allAreas = np.zeros(shape=(len(points_i),len(PCF_radii_lower)))

    for annulus in range(len(PCF_radii_lower)):
        inner = PCF_radii_lower[annulus]
        outer = PCF_radii_upper[annulus]

        areas_in = vfunc_returnAreaOfCircleInDomainAroundPoint(index=np.arange(len(points_i)), points=points_i, r=inner, domainX=domainX, domainY=domainY)
        areas_out = vfunc_returnAreaOfCircleInDomainAroundPoint(index=np.arange(len(points_i)), points=points_i, r=outer, domainX=domainX, domainY=domainY)

        areas = areas_out - areas_in
        if not np.all(areas >= 0):
            raise RuntimeError(f'Negative areas calculated for point {np.argwhere(areas < 0)}.')
        allAreas[:,annulus] = areas
    return allAreas

def returnAreaOfCircleInDomainAroundPoint(index, points, r, domainX, domainY):
    point = points[index,:]
    area = returnAreaOfCircleInDomain(point[0], point[1], r, domainX, domainY)
    return area

def getAnnulusAreasAroundPoints_polygon(points_i, maxR, annulusStep, annulusWidth, polygon):
    # We want to populate a table the same size as distances, which contains the area of the annulus containing that contribution
    # i.e., "at distance D(i->j) from point i, what is area of containing annulus?"
    vfunc_returnAreaOfCircleInDomainAroundPoint = np.vectorize(returnAreaOfCircleInDomainAroundPoint_polygon,excluded=['points','r','polygon'])
    PCF_radii_lower = np.arange(0, maxR+annulusStep, annulusStep)
    PCF_radii_upper = np.arange(annulusWidth, maxR + annulusWidth + annulusStep, annulusStep)

    allAreas = np.zeros(shape=(len(points_i),len(PCF_radii_lower)))

    for annulus in range(len(PCF_radii_lower)):
        inner = PCF_radii_lower[annulus]
        outer = PCF_radii_upper[annulus]

        areas_in = vfunc_returnAreaOfCircleInDomainAroundPoint(index=np.arange(len(points_i)), points=points_i, r=inner, polygon=polygon)
        areas_out = vfunc_returnAreaOfCircleInDomainAroundPoint(index=np.arange(len(points_i)), points=points_i, r=outer, polygon=polygon)

        areas = areas_out - areas_in
        if not np.all(areas >= 0):
            raise RuntimeError(f'Negative areas calculated for point {np.argwhere(areas < 0)}.')
        allAreas[:,annulus] = areas
    return allAreas

def returnAreaOfCircleInDomainAroundPoint_polygon(index, points, r, polygon):
    px, py = points[index,:]
    circle = Point(px,py).buffer(r)
    return polygon.intersection(circle).area


def returnAreaOfCircleInDomain(x0, y0, r, domainX, domainY):
    intersectionPoints = returnIntersectionPoints(x0, y0, r, domainX, domainY)
    if not intersectionPoints:
        area = np.pi * r ** 2
    else:
        # Need to calculate area from intersection Points
        intersectionPoints.append(intersectionPoints[0])
        area = 0
        for v in range(len(intersectionPoints) - 1):
            a = intersectionPoints[v]
            b = intersectionPoints[v + 1]
            # Find out if this is a segment or a triangle
            isTriangle = False

            # Check if point b is anticlockwise from point a on the same line
            if a[0] == b[0] or a[1] == b[1]:
                if a[0] == b[0]:  # On a vertical line
                    if a[0] == domainX[0]:
                        # LHS
                        if b[1] < a[1]:
                            isTriangle = True
                    else:
                        # RHS
                        if b[1] > a[1]:
                            isTriangle = True
                else:  # On a horizontal line
                    if a[1] == domainY[0]:
                        # bottom
                        if b[0] > a[0]:
                            isTriangle = True
                    else:
                        # top
                        if a[0] > b[0]:
                            isTriangle = True

            # If points are on the same line moving anticlockwise, then return the area of the triangle formed by a, b and the centre
            if isTriangle:
                # Points share a border: return area of triangle between them
                area = area + 0.5 * np.abs(a[0] * (b[1] - y0) + b[0] * (y0 - a[1]) + x0 * (a[1] - b[1]))
            else:
                # Else, return the area of the circle segment between them
                # We need to be careful to take the angle between v1 and v2 in an anticlockwise direction
                v1 = [x0 - a[0], y0 - a[1]]
                v2 = [x0 - b[0], y0 - b[1]]

                theta = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
                # Normalise to 0, 2pi
                if theta < 0:
                    theta = theta + 2 * np.pi

                area = area + 0.5 * theta * r ** 2
    return area


def returnIntersectionPoints(x0, y0, r, domainX, domainY):
    # Calculate the points of intersection between a circle of radius r centred at (x0,y0)
    # and the box boundaries x = domainX[0], y = domainY[0], x = domainX[1] and y = domainY[1]
    # This also includes corners which are within the domain

    # Find intersections with each of the 4 domain edges - gives max of 8 intersections. Then take floor/ceil to take points outside of domain to domain corners
    # We assume for now that domain edges are parallel to coordinate axes
    # i.e, vertices are [ [domainX[0],domainY[0]], [domainX[1],domainY[0]], [domainX[0],domainY[1]], [domainX[1],domainY[1]]  ]

    # Line of form ax + by = c
    # circle of form (x - x0)^2 + (y - y0)^2 = r^2
    # Need r^2 (a^2 + b^2) - (c - ax0 - by0)^2 > 0
    # See page 17 of https://www2.mathematik.tu-darmstadt.de/~ehartmann/cdgen0104.pdf
    def calculateUsefulValues(rSquared, a, b, c, x0, y0):
        cPrime = c - a*x0 - b*y0
        return rSquared*(a**2 + b**2) - cPrime**2, cPrime
    def getIntersections(val, a, b, cPrime, x0, y0):
        temp = np.sqrt(val)/(a**2 + b**2)
        point1 = [x0 + a*cPrime + b*temp, y0 + b*cPrime - a*temp]
        point2 = [x0 + a*cPrime - b*temp, y0 + b*cPrime + a*temp]
        return [point1, point2]

    rSquared = r**2
    intersectionPoints = []
    # Move around anti-clockwise from upper left corner
    # LEFT HAND SIDE
    # x = domainX[0], i.e. a=1, b=0, c=domainX[0]
    val, cPrime = calculateUsefulValues(rSquared, 1, 0, domainX[0], x0, y0)
    if val > 0:
        intersections = getIntersections(val, 1, 0, cPrime, x0, y0)
        intersectionPoints.extend([intersections[1],intersections[0]])
    # BOTTOM
    # y = domainY[0], i.e. a=0, b=1, c=domainY[0]
    val, cPrime = calculateUsefulValues(rSquared, 0, 1, domainY[0], x0, y0)
    if val > 0:
        intersections = getIntersections(val, 0, 1, cPrime, x0, y0)
        intersectionPoints.extend([intersections[1],intersections[0]])
    # RIGHT HAND SIDE
    # x = domainX[1], i.e. a=1, b=0, c=domainX[1]
    val, cPrime = calculateUsefulValues(rSquared, 1, 0, domainX[1], x0, y0)
    if val > 0:
        intersections = getIntersections(val, 1, 0, cPrime, x0, y0)
        intersectionPoints.extend(intersections)
    # BOTTOM
    # y = domainY[0], i.e. a=0, b=1, c=domainY[1]
    val, cPrime = calculateUsefulValues(rSquared, 0, 1, domainY[1], x0, y0)
    if val > 0:
        intersections = getIntersections(val, 0, 1, cPrime, x0, y0)
        intersectionPoints.extend(intersections)
    
    temp = [[np.max([np.min([point[0],domainX[1]]),domainX[0]]), np.max([np.min([point[1],domainY[1]]),domainY[0]])] for point in intersectionPoints]
    intersectionPoints = [temp[v] for v in range(len(temp)) if temp[v] not in temp[:v]]
    return intersectionPoints



