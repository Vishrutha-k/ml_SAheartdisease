import numpy as np
import xlrd
from pylab import *
from matplotlib.pyplot import figure, show, xlabel, ylabel, plot, legend, bar, title
from sklearn.mixture import GaussianMixture
from scipy.linalg import svd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.neighbors import NearestNeighbors

def clusterplot(X, clusterid, centroids='None', y='None', covars='None'):
    '''
    CLUSTERPLOT Plots a clustering of a data set as well as the true class
    labels. If data is more than 2-dimensional it should be first projected
    onto the first two principal components. Data objects are plotted as a dot
    with a circle around. The color of the dot indicates the true class,
    and the cicle indicates the cluster index. Optionally, the centroids are
    plotted as filled-star markers, and ellipsoids corresponding to covariance
    matrices (e.g. for gaussian mixture models).

    Usage:
    clusterplot(X, clusterid)
    clusterplot(X, clusterid, centroids=c_matrix, y=y_matrix)
    clusterplot(X, clusterid, centroids=c_matrix, y=y_matrix, covars=c_tensor)
    
    Input:
    X           N-by-M data matrix (N data objects with M attributes)
    clusterid   N-by-1 vector of cluster indices
    centroids   K-by-M matrix of cluster centroids (optional)
    y           N-by-1 vector of true class labels (optional)
    covars      M-by-M-by-K tensor of covariance matrices (optional)
    '''
    
    X = np.asarray(X)
    cls = np.asarray(clusterid)
    if type(y) is str and y=='None':
        y = np.zeros((X.shape[0],1))
    else:
        y = np.asarray(y)
    if type(centroids) is not str:
        centroids = np.asarray(centroids)
    K = np.size(np.unique(cls))
    C = np.size(np.unique(y))
    ncolors = np.max([C,K])
    
    # plot data points color-coded by class, cluster markers and centroids
    #hold(True)
    colors = [0]*ncolors
    for color in range(ncolors):
        colors[color] = cm.brg(color/(ncolors-1))[:3]
    for i,cs in enumerate(np.unique(y)):
        plot(X[(y==cs).ravel(),0], X[(y==cs).ravel(),1], 'o', markeredgecolor='k', markerfacecolor=colors[i],markersize=6, zorder=2)
    for i,cr in enumerate(np.unique(cls)):
        plot(X[(cls==cr).ravel(),0], X[(cls==cr).ravel(),1], 'o', markersize=12, markeredgecolor=colors[i], markerfacecolor='None', markeredgewidth=3, zorder=1)
    if type(centroids) is not str:        
        for cd in range(centroids.shape[0]):
            plot(centroids[cd,0], centroids[cd,1], '*', markersize=22, markeredgecolor='k', markerfacecolor=colors[cd], markeredgewidth=2, zorder=3)
    # plot cluster shapes:
    if type(covars) is not str:
        for cd in range(centroids.shape[0]):
            x1, x2 = gauss_2d(centroids[cd],covars[cd,:,:])
            plot(x1,x2,'-', color=colors[cd], linewidth=3, zorder=5)
    #hold(False)

    # create legend        
    legend_items = np.unique(y).tolist()+np.unique(cls).tolist()+np.unique(cls).tolist()
    for i in range(len(legend_items)):
        if i<C: legend_items[i] = 'Class: {0}'.format(legend_items[i]);
        elif i<C+K: legend_items[i] = 'Cluster: {0}'.format(legend_items[i]);
        else: legend_items[i] = 'Centroid: {0}'.format(legend_items[i]);
    legend(legend_items, numpoints=1, markerscale=.75, prop={'size': 9})
    
def gauss_2d(centroid, ccov, std=2, points=100):
    ''' Returns two vectors representing slice through gaussian, cut at given standard deviation. '''
    mean = np.c_[centroid]; tt = np.c_[np.linspace(0, 2*np.pi, points)]
    x = np.cos(tt); y=np.sin(tt); ap = np.concatenate((x,y), axis=1).T
    d, v = np.linalg.eig(ccov); d = std * np.sqrt(np.diag(d))
    bp = np.dot(v, np.dot(d, ap)) + np.tile(mean, (1, ap.shape[1])) 
    return bp[0,:], bp[1,:]

def gausKernelDensity(X,width):
    '''
    GAUSKERNELDENSITY Calculate efficiently leave-one-out Gaussian Kernel Density estimate
    Input: 
      X        N x M data matrix
      width    variance of the Gaussian kernel
    
    Output: 
      density        vector of estimated densities
      log_density    vector of estimated log_densities
    '''
    X = np.mat(np.asarray(X))
    N,M = X.shape

    # Calculate squared euclidean distance between data points
    # given by ||x_i-x_j||_F^2=||x_i||_F^2-2x_i^Tx_j+||x_i||_F^2 efficiently
    x2 = np.square(X).sum(axis=1)
    D = x2[:,[0]*N] - 2*X.dot(X.T) + x2[:,[0]*N].T

    # Evaluate densities to each observation
    Q = np.exp(-1/(2.0*width)*D)
    # do not take density generated from the data point itself into account
    Q[np.diag_indices_from(Q)]=0
    sQ = Q.sum(axis=1)
    
    density = 1/((N-1)*np.sqrt(2*np.pi*width)**M+1e-100)*sQ
    log_density = -np.log(N-1)-M/2*np.log(2*np.pi*width)+np.log(sQ)
    return np.asarray(density), np.asarray(log_density)
    
def data_import():
    doc = xlrd.open_workbook('data.xlsx')
    sheet = doc.sheet_by_index(0)
    
    data = []
    
    for row in range (1,sheet.nrows):
        for x in range(1,5):
            data.append(sheet.cell_value(row,x))
        for x in range(6,12):
            data.append(sheet.cell_value(row,x))
     
    X = np.asarray(data)
    X = X.reshape(int(len(X)/10),10)
    
    y = X[:,9]
    X = np.delete(X,[4,9],axis=1)
    
    return X,y

def do_svd(X):
    N0, M0 = X.shape
    
    Y = X - np.ones((N0,1))*X.mean(0)
    U,S,V = svd(Y,full_matrices=False)
    V = V.T
    
    Z = np.dot(Y,V)
    
    Z = np.delete(Z,[1,3,4,5,6,7,8,9],axis=1)
    N, M = Z.shape
    
    return Z,N,M

def GMM(X,y):
    Z,N,M = do_svd(X)
    # Number of clusters
    K = 2
    cov_type = 'diag'       
    # type of covariance, you can try out 'diag' as well
    reps = 10                
    # number of fits with different initalizations, best result will be kept
    # Fit Gaussian mixture model
    gmm = GaussianMixture(n_components=K, covariance_type=cov_type, n_init=reps).fit(Z)
    cls = gmm.predict(Z)    
    # extract cluster labels
    cds = gmm.means_        
    # extract cluster centroids (means of gaussians)
    covs = gmm.covariances_
    # extract cluster shapes (covariances of gaussians)
    if cov_type == 'diag':    
        new_covs = np.zeros([K,M,M])    
    
    count = 0    
    for elem in covs:        
        temp_m = np.zeros([M,M])        
        for i in range(len(elem)):            
            temp_m[i][i] = elem[i]        
        
        new_covs[count] = temp_m        
        count += 1
            
    covs = new_covs
    # Plot results:
    figure(figsize=(14,9))
    clusterplot(Z, clusterid=cls, centroids=cds, y=y, covars=covs)
    xlabel('PCA1')
    ylabel('PCA3')
    show()
    
    
def hierarchical(X,y):
    Z,N,M = do_svd(X)
    
    # Perform hierarchical/agglomerative clustering on data matrix
    Method = 'weighted'
    Metric = 'euclidean'
    
    Z1 = linkage(Z, method=Method, metric=Metric)
    
    # Compute and display clusters by thresholding the dendrogram
    Maxclust = 2
    cls = fcluster(Z1, criterion='maxclust', t=Maxclust)
    figure(1,figsize=(14,9))
    clusterplot(Z, cls.reshape(cls.shape[0],1), y=y)
    xlabel('PCA1')
    ylabel('PCA3')
    
    # Display dendrogram
    max_display_levels=6
    figure(2,figsize=(10,4))
    dendrogram(Z1, truncate_mode='level', p=max_display_levels)
    
    show()
    
def KNN(X):
    Z,N,M = do_svd(X)
    ### K-nearest neigbor average relative density
    # Neighbor to use:
    K = 5
    # Compute the average relative density
    
    knn = NearestNeighbors(n_neighbors=K).fit(X)
    D, i = knn.kneighbors(X)
    density = 1./(D.sum(axis=1)/K)
    avg_rel_density = density/(density[i[:,1:]].sum(axis=1)/K)
    
    # Sort the avg.rel.densities
    i_avg_rel = avg_rel_density.argsort()
    np.savetxt('rank.txt',
               i_avg_rel[:30],fmt='%.i',
               delimiter=' ',
               newline=',')
    avg_rel_density = avg_rel_density[i_avg_rel]
    
    # Plot k-neighbor estimate of outlier score (distances)
    figure(1,figsize=(14,9))
    bar(range(20),avg_rel_density[:20])
    title('KNN average relative density: Outlier score')
    xlabel('19 most extreme outliers')
    ylabel('KNN Average Relative Density')
    
    figure(2,figsize=(14,9))
    bar(range(len(avg_rel_density)),avg_rel_density)
    title('KNN average relative density: Outlier score')
    xlabel('All observations')
    ylabel('KNN Average Relative Density')

def main():
    X,y=data_import()
    GMM(X,y)
    hierarchical(X,y)
    KNN(X)
    
main()