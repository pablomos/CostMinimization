import scipy.sparse as sp
import scipy.sparse.linalg as la
import numpy as np
import program as pr

# Create a mock sparse matrix
def create_mock_sparse_matrix():
    # The matrix will be [[3, 0, 0,], [0, 5, 0], [0, 0, 7]]
    rows = [0, 1, 2]
    cols = [0, 1, 2]
    vals = [3, 5, 7]
    sparseMatrix = sp.csr_matrix((vals, (rows, cols)), shape = (3, 3))
    return sparseMatrix

# Solve a mock LSQR problem
def mock_lsqr():
    # The vector b will be [11, 13, 17]
    A = create_mock_sparse_matrix()
    b = [11, 13, 17]
    retVal = la.lsqr(A, b)
    x = retVal[0]
    diffVec = A * x - b
    diffVecSq = diffVec * diffVec
    sumOfTerms = np.sum(diffVecSq)
    L = 0.5 * sumOfTerms
    return (x,L)

# Try LSQR with a subset of the real data
def lsqr_subset():
    (A, b) = pr.reduce_data(1000, 1000)
    aSparse = sp.csr_matrix((A[2], (A[0], A[1])), shape = (1000, 1000))
    retVal = la.lsqr(aSparse, b)
    x = retVal[0]
    diffVec = aSparse * x - b
    L = 0.5 * np.sum(diffVec * diffVec)
    return (x, L)

# Try LSQR with the entire data
def lsqr(aFile, bFile, nCols, aTol = 1e-5, bTol = 1e-9):
    # Read the data
    (aLists,b) = pr.read_data(aFile, bFile)
    A = sp.csr_matrix((aLists[2], (aLists[0], aLists[1])), shape = (len(b), nCols))
    retVal = la.lsqr(A, b, atol = aTol, btol = bTol, show = True)
    x = retVal[0]
    diffVec = pr.difference_vector(aLists, x, b)
    L = pr.cost(diffVec)
    return (x, L, retVal[1], retVal[2], aLists, b)
    
