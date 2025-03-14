import numpy as np
import scipy, scipy.spatial
import cvxopt, cvxopt.solvers

def getGaussianKernel(X1, X2, scale):
    D = scipy.spatial.distance.cdist(X1, X2, 'sqeuclidean')
    return np.exp(-D / (2 * scale**2))

def getQPMatrices(K, Y, C):
    # Prepare matrices
    n = Y.shape[0]
    P = Y[:,np.newaxis]*K*Y[np.newaxis,:]
    
    # # alternatively
    # #-----------
    # diag = np.diag(Y)
    # P = np.matmul(diag, np.matmul(K, diag))
    # #-----------
    
    q = -np.ones([n])
    G = np.concatenate([-np.identity(n), np.identity(n)])
    h = np.concatenate([np.zeros([n]), C * np.ones([n])])
    A = Y.reshape((1, n))
    b = np.array([0.0])
    
    # Convert to CVXOPT matrices
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)
    A = cvxopt.matrix(A)
    b = cvxopt.matrix(b)
    
    return P, q, G, h, A, b

def getTheta(K, Y, alpha, C):
    # First we need to find a support vector with 0 < alpha_i < C.
    # Instead of looking at all possible alpha's in a loop, we use  the midpoint heuristic.
    # Note: the value lying closer to C / 2 is more likely to satisfy this condition.
    # Considering the absolute difference np.abs ensures that the value \alpha_i does
    # not lie to close to the boundaries 0 or C improving upon the numerical stability.
    sv = np.argmin(np.abs(alpha - C / 2.0))
    theta = Y[sv] - np.dot(K[sv,:], alpha * Y)
    return theta

def fit(self, X, Y):
    K = getGaussianKernel(X, X, self.scale)
    P, q, G, h, A, b = getQPMatrices(K, Y, self.C)
    
    alpha = np.array(cvxopt.solvers.qp(P, q, G, h, A, b)['x']).flatten()
    
    th = 1e-6 * alpha.mean()
    ind = alpha > th # determine (robust) support vectors (alternatively set th = 0)
    self.X, self.Y, self.alpha = X[ind], Y[ind], alpha[ind]
    
    self.theta = getTheta(K, Y, alpha, self.C)

def predict(self, X):
    K = getGaussianKernel(X, self.X, self.scale)
    Y = np.sign(np.dot(K, self.alpha * self.Y) + self.theta)
    return Y