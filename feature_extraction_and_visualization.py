from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, savefig
from scipy.linalg import svd
import numpy as np
import xlrd


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
    
    return np.reshape(X,(int(len(X)/10),10))


def PCA(X):
    N = len(X)
    
    Y = X - np.ones((N,1))*X.mean(0)
    U,S,V = svd(Y,full_matrices=False)
    V = V.T
    
    Z = np.dot(Y,V)
    
    return Z


def plot_PCA(X,Z):
    CHD = X[:,9]
    
    PCA1 = Z[:,0]
    PCA3 = Z[:,2]
    
    sickPressure = []
    healthyPressure = []
    sickLdl = []
    healthyLdl = []
    
    for x in range(len(CHD)):
        if (CHD[x] == 1):
            sickPressure.append(PCA1[x])
            sickLdl.append(PCA3[x])
        else:
            healthyPressure.append(PCA1[x])
            healthyLdl.append(PCA3[x])
            
    figure(1)
    plot(sickPressure,sickLdl,'.r')
    plot(healthyPressure,healthyLdl,'.b')
    xlabel('PCA1')
    ylabel('PCA3')
    savefig('PCA.png', dpi=900)
    show()
    
    
def plot_variance(X):
    N = len(X)
    
    Y = X - np.ones((N,1))*X.mean(0)
    U,S,V = svd(Y,full_matrices=False)
    rho = (S*S) / (S*S).sum() 
    
    figure(1)
    plot(range(1,len(rho)+1),rho,'o-')
    title('Variance explained by principal components');
    xlabel('Principal component');
    ylabel('Variance explained');
    savefig('variance_explained.png',dpi=900)
    show()

    
def main():
    X = data_import()
    Z = PCA(X)
    
    plot_variance(X)
    plot_PCA(X,Z)
    
main()