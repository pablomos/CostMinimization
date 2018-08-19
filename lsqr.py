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
    (x, reason, nIters) = lsqr_single_sparse(A, b, 1e-8, 1e-8, False)
    diffVec = A * x - b
    diffVecSq = diffVec * diffVec
    sumOfTerms = np.sum(diffVecSq)
    L = 0.5 * sumOfTerms
    return (x,L)

# Try LSQR with a subset of the real data
def lsqr_subset():
    (A, b) = pr.reduce_data(1000, 1000)
    (aSparse, (x, reason, nIters)) = lsqr_single(A, b, 1000, 1e-8, 1e-8, False)
    diffVec = aSparse * x - b
    L = 0.5 * np.sum(diffVec * diffVec)
    return (x, L)

# Compute LSQR for a given A,b
# A is of the form (rows, cols, vals)
def lsqr_single(A, b, nCols, aTol, bTol, verbose):
    aSparse = sp.csr_matrix((A[2], (A[0], A[1])), shape = (len(b), nCols))
    return (aSparse, lsqr_single_sparse(aSparse, b, aTol, bTol, verbose))

# Compute LSQR for a given A, b
# A is a Python sparse matrix
def lsqr_single_sparse(A, b, aTol, bTol, verbose):
    return la.lsqr(A, b, atol = aTol, btol = bTol, show = verbose)[:3]

# Try LSQR with the entire data
def lsqr(aFile, bFile, nCols, aTol = 1e-5, bTol = 1e-9):
    # Read the data
    (aLists,b) = pr.read_data(aFile, bFile)
    (aSparse, (x, reason, nIters)) = lsqr_single(aLists, b, nCols, aTol, bTol, True)
    diffVec = pr.difference_vector(aLists, x, b)
    L = pr.cost(diffVec)
    return (x, L, reason, nIters, aLists, b)

# Try LSQR with multiple b vectors
'''def lsqr_test():
    nTests = 100

    # Generate nTests random x vectors
    xs = [[rd.randint(0, 100)/10. for j in range(100000)] for i in range(nTests)]
    # Generate the corresponding b vectors
    A = pr.read_data('Archive/a.txt', 'Archive/b.txt')[0]
    bs = [np.zeros(300000) for i in range(len(xs))]
    for aRow in range(len(A[0])):
        row = A[0][aRow]
        col = A[1][aRow]
        val = A[2][aRow]
        for i in range(len(xs)):
            x = xs[i]
            b = bs[i]
            b[row] += (val * x[col])

    # For each b vector, find x using lsqr
    for b in bs:
        

    # Compare the x obtained with the one used in the input using program.py

    # Make sure the cost was less than 1e-3 in all cases

    # Look at the number of iterations for each test and see if they are roughly constant

    # Compute the time it took for each iteration

    # Compute one of the rows manually'''
