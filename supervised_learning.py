import numpy as np
import xlrd
import sklearn.linear_model as lm
from scipy import stats
from sklearn import tree
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, subplot, hist, tight_layout, ylim


attributeNames = [
    'Systolic blood pressure',
    'Cumulative tobacco use',
    'Low densiity lipoprotein cholesterol',
    'Adiposity',
    'Type A behavior',
    'Obesity',
    'Alcohol consumption',
    'Age'
    ]

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

def linear_regression(X,y):
    # Normalize data
    X = stats.zscore(X);
    
    # Split dataset into features and target vector
    blood_pressure = attributeNames.index('Systolic blood pressure')
    y = X[:,blood_pressure]
    
    X_cols = list(range(0,blood_pressure)) + list(range(blood_pressure+1,len(attributeNames)))
    X = X[:,X_cols]
    
    # Fit ordinary least squares regression model
    model = lm.LinearRegression()
    model.fit(X,y)
    
    # Predict alcohol content
    y_est = model.predict(X)
    residual = y_est-y
    
    return y_est, residual

def plot_linear_regression(y,y_est,residual):
    # Display scatter plot
    figure()
    subplot(2,1,1)
    plot(y, y_est, '.')
    xlabel('Systolic blood pressure (true)'); ylabel('Systolic blood pressure (estimated)');
    subplot(2,1,2)
    hist(residual,40)
    tight_layout()
    
    show()
    
def logistic_regression(X,y):
    # Fit logistic regression model
    model = lm.logistic.LogisticRegression()
    model = model.fit(X,y)
    
    y_est = model.predict(X)
    y_est_white_prob = model.predict_proba(X)[:, 0] 
    
    # Evaluate classifier's misclassification rate over entire training data
    misclass_rate = sum(np.abs(y_est - y)) / float(len(y_est))
    
    print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))
    
    return y_est,y_est_white_prob

def plot_logistic_regression(y,y_est_white_prob):
    class0_ids = np.nonzero(y==0)[0].tolist()
    plot(class0_ids, y_est_white_prob[class0_ids], '.r')
    class1_ids = np.nonzero(y==1)[0].tolist()
    plot(class1_ids, y_est_white_prob[class1_ids], '.b')
    xlabel('Data object (patient)'); ylabel('Predicted prob. of class Sick');
    legend(['Sick', 'Healthy'])
    ylim(-0.01,1.5)
    
    show()
    
def decision_tree(X,y):
    dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=4)
    dtc = dtc.fit(X,y)
    out = tree.export_graphviz(dtc, out_file='tree_gini_SAHD_data.gvz', feature_names=attributeNames)
    
    
def main():
    X,y = data_import()
    
    y_est_1, residual = linear_regression(X,y)
    plot_linear_regression(y,y_est_1,residual)
    
    y_est_2,y_est_white_prob = logistic_regression(X,y)
    plot_logistic_regression(y,y_est_white_prob)
    
    decision_tree(X,y)
    
main()