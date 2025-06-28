import numpy as np
import sklearn
from scipy.linalg import khatri_rao
from sklearn.linear_model import LogisticRegression


# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map, my_decode etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your models using training CRPs
	# X_train has 8 columns containing the challenge bits
	# y_train contains the values for responses
	
	# THE RETURNED MODEL SHOULD BE ONE VECTOR AND ONE BIAS TERM
	# If you do not wish to use a bias term, set it to 0
    X_train = my_map(X_train)
    log_model = LogisticRegression(C=100.0, penalty='l1', solver='liblinear', max_iter=100,tol=0.0001,fit_intercept=False)
    log_model.fit(X_train, y_train)
    w = log_model.coef_.flatten()
    return w, 0


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
	
    X = np.c_[X, np.ones(X.shape[0])]
    KR = khatri_rao(X.T, X.T) 

    return KR.T


################################
# Non Editable Region Starting #
################################
def my_decode( w ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to invert a PUF linear model to get back delays
	# w is a single 65-dim vector (last dimension being the bias term)
	# The output should be four 64-dimensional vectors
	
	# Reiterated code (unchanged, as it already meets all requirements)



    W = w[:-1]  
    B = w[-1]   
    
    A = np.zeros((65, 256)) 
    
    for i in range(64):
        p, q, r, s = 4*i, 4*i+1, 4*i+2, 4*i+3
        
        
        if i == 0:
            A[i, p] = 0.5
            A[i, q] = -0.5
            A[i, r] = 0.5
            A[i, s] = -0.5
        else:
            A[i, p] = 0.5
            A[i, q] = -0.5
            A[i, r] = 0.5
            A[i, s] = -0.5
            
            
            prev_p, prev_q, prev_r, prev_s = 4*(i-1), 4*(i-1)+1, 4*(i-1)+2, 4*(i-1)+3
            A[i, prev_p] += 0.5
            A[i, prev_q] += -0.5
            A[i, prev_r] += -0.5
            A[i, prev_s] += 0.5
    
    
    i = 63
    p, q, r, s = 4*i, 4*i+1, 4*i+2, 4*i+3
    A[64, p] = 0.5
    A[64, q] = -0.5
    A[64, r] = -0.5
    A[64, s] = 0.5
    
    m, n = A.shape
    x = np.zeros(n)  
    
    for _ in range(1000):
        grad = A.T @ (A @ x - w) 
        
       
        x_new = x - 0.01 * grad  
        x_new = np.maximum(x_new, 0) 
        
      
        if np.linalg.norm(x_new - x) < 1e-6:
            break
        x = x_new
    
    #residual = np.linalg.norm(A @ x - w)
    
    P = x[0::4]
    Q = x[1::4]
    R = x[2::4]
    S = x[3::4]
    
    return P, Q, R, S

