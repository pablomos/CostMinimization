import scipy.sparse as sp
import scipy.sparse.linalg as la
import numpy as np
import program as pr
import matplotlib.pyplot as mp
import random as rd

# Create CSR matrix from lists of rows, columns and values
def convert_to_sparse_matrix(mal, nRows, nCols):
    return sp.csr_matrix((mal[2], (mal[0], mal[1])), shape = (nRows, nCols))

# Create a mock sparse matrix
def create_mock_sparse_matrix():
    # The matrix will be [[3, 0, 0,], [0, 5, 0], [0, 0, 7]]
    rows = [0, 1, 2]
    cols = [0, 1, 2]
    vals = [3, 5, 7]
    return convert_to_sparse_matrix((rows, cols, vals))

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
    aSparse = convert_to_sparse_matrix(A, len(b), nCols)
    return (aSparse, lsqr_single_sparse(aSparse, b, aTol, bTol, verbose))

# Compute LSQR for a given A, b
# A is a Python sparse matrix
def lsqr_single_sparse(A, b, aTol, bTol, verbose):
    return la.lsqr(A, b, atol = aTol, btol = bTol, show = verbose)[:3]

# Compute the cost function; A must be of the form (rows, cols, vals)
def cost(A, b, x):
    return pr.cost(pr.difference_vector(A, x, b))

# Try LSQR with the entire data
def lsqr(aFile, bFile, nCols, aTol = 1e-5, bTol = 1e-9):
    # Read the data
    (aLists,b) = pr.read_data(aFile, bFile)
    (x, reason, nIters) = lsqr_single(aLists, b, nCols, aTol, bTol, True)[1]
    L = cost(aLists, b, x)
    return (x, L, reason, nIters, aLists, b)

# Try LSQR with multiple b vectors
# 100 tests with tolerances 1e-7, 1e-9 all passed with ~400 iterations
def lsqr_test(aTol, bTol, nTests):
    lenX = 100000
    maxXValue = 10
    maxL = 1e-3
    maxFracDiff = 1e-2

    # Generate nTests random x vectors
    print("Generating random xs...")
    xs = [[rd.randint(0, maxXValue * 10)/10. for j in range(lenX)] for i in range(nTests)]
    # Generate the corresponding b vectors
    print("Reading the real data...")
    aAndRealB = pr.read_data('Archive/a.txt', 'Archive/b.txt')
    A = aAndRealB[0]
    realB = aAndRealB[1]
    print("Initialising the corresponding bs...")
    bs = [np.zeros(len(realB)) for i in range(len(xs))]
    print("Filling the values of bs...")
    for aRow in range(len(A[0])):
        row = A[0][aRow]
        col = A[1][aRow]
        val = A[2][aRow]
        for i in range(len(xs)):
            x = xs[i]
            b = bs[i]
            b[row] += (val * x[col])

    # For each b vector, find x using lsqr
    print("Converting to sparse matrix...")
    aSparse = convert_to_sparse_matrix(A, len(realB), lenX)
    predictedXs = []
    iters = []
    costs = []
    fracDiffs = []
    for i in range(len(bs)):
        print("Test " + str(i))
        b = bs[i]
        (x, reason, nIters) = lsqr_single_sparse(aSparse, b, aTol, bTol, False)
        predictedXs.append(x)
        iters.append(nIters)

        # Compare the x obtained with the one used in the input using program.py
        pr.compare_x_arrays(x, xs[i])

        # Make sure the cost was less than maxL in all cases
        L = cost(A, b, x)
        if L > maxL:
            print("The cost is too high: " + str(L))
        costs.append(L)

        # Compute one of the rows manually
        firstRow = A[0][0]
        firstColumn = A[1][0]
        firstValue = A[2][0]
        absDiff = (firstValue * x[firstColumn] - b[firstRow])
        if (b[firstRow] == 0 and absDiff != 0) or (b[firstRow] != 0 and absDiff / b[firstRow] > maxFracDiff):
            print("Difference is too large!")
        fracDiffs.append(fracDiff)

    # Look at the number of iterations for each test and see if they are roughly constant
    mp.plot(range(len(bs)), iters)
    mp.show()

    # Look at the costs for each test
    mp.plot(range(len(bs)), costs)
    mp.show()
