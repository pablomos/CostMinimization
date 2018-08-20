import scipy.sparse as sp
import program as pr

# Create CSR matrix from lists of rows, columns and values
'''
There are various sparse matrix classes available in the package. This one was the fastest in a test run in a separate program.
'''
def convert_to_sparse_matrix(mal, nRows, nCols):
    return sp.csr_matrix((mal[2], (mal[0], mal[1])), shape = (nRows, nCols))

# Create a mock sparse matrix
def create_mock_sparse_matrix():
    # The matrix will be [[3, 0, 0,], [0, 5, 0], [0, 0, 7]]
    rows = [0, 1, 2]
    cols = [0, 1, 2]
    vals = [3, 5, 7]
    return convert_to_sparse_matrix((rows, cols, vals), 3, 3)

# Solve a mock LSQR problem
def mock_lsqr():
    # The vector b will be [11, 13, 17]
    A = create_mock_sparse_matrix()
    b = [11, 13, 17]
    (x, reason, nIters, r) = lsqr_single_sparse(A, b, 1e-8, 1e-8, False, None)
    return (x, 0.5 * r * r)

# Compute LSQR for a given A,b
# A is of the form (rows, cols, vals)
def lsqr_single(A, b, nCols, aTol, bTol, verbose, maxIter):
    aSparse = convert_to_sparse_matrix(A, len(b), nCols)
    return (aSparse, lsqr_single_sparse(aSparse, b, aTol, bTol, verbose, maxIter))

# Compute LSQR for a given A, b
# A is a Python sparse matrix
'''
This method seems to implement the LSQR algorithm presented in Paige & Saunders (1982): https://web.stanford.edu/class/cme324/paige-saunders2.pdf
We also tried LSMR, featured in the same package, but it was slower
'''
def lsqr_single_sparse(A, b, aTol, bTol, verbose, maxIter):
    r = sp.linalg.lsqr(A, b, atol = aTol, btol = bTol, show = verbose, iter_lim = maxIter)[:4]
    return r

# Try LSQR with the entire data
# aTol and bTol represent the actual error in the input data
# The optimal number of maxIter found in find_iters()
def lsqr(aFile, bFile, nCols, aTol = 1e-6, bTol = 1e-7, verbose = True, maxIter = 150):
    # Read the data
    (aLists,b) = pr.read_data(aFile, bFile)
    (aSparse, (x, reason, nIters, r)) = lsqr_single(aLists, b, nCols, aTol, bTol, verbose, maxIter)
    L = 0.5 * r * r
    retVal = (x, L, reason, nIters, aLists, b, aSparse)
    return retVal
