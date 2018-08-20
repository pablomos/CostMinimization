import scipy.sparse as sp
import numpy as np
import program as pr
import matplotlib.pyplot as mp
import random as rd

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

# Try LSQR with a subset of the real data
def lsqr_subset():
    (A, b) = pr.reduce_data(1000, 1000)
    (x, reason, nIters, r) = lsqr_single(A, b, 1000, 1e-8, 1e-8, False, None)[1]
    L = 0.5 * r * r
    return (x, L)

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

# Generate random b vectors; realB must be a NumPy array
def gen_rand_b(A, realB, lenX, nTests, xMax, sc):
    realBNorm = np.sqrt(np.sum(realB * realB))

    # Generate nTests random x vectors
    print("Generating random xs...")
    xs = np.array([[rd.randint(1, xMax * sc)/float(sc) for j in range(lenX)] for i in range(nTests)])
    # Generate the corresponding b vectors
    print("Initialising the corresponding bs...")
    fakeBs = [np.zeros(len(realB)) for i in range(len(xs))]
    print("Filling the values of bs...")
    for aRow in range(len(A[0])):
        row = A[0][aRow]
        col = A[1][aRow]
        val = A[2][aRow]
        for i in range(len(xs)):
            x = xs[i]
            b = fakeBs[i]
            b[row] += (val * x[col])
    for i in range(len(fakeBs)):
        fakeB = fakeBs[i]
        fakeX = xs[i]
        fakeBNorm = np.sqrt(np.sum(fakeB * fakeB))
        factor = realBNorm / fakeBNorm
        fakeB *= factor
        fakeX *= factor
    return (xs, fakeBs)

# Try LSQR with multiple b vectors
def lsqr_test(aTol, bTol, nTests):
    lenX = 100000
    maxL = 1e-3
    maxFracDiff = 1e-2
    maxXValue = 10
    multFac = 1. / bTol

    print("Reading the real data...")
    aAndRealB = pr.read_data('Archive/a.txt', 'Archive/b.txt')
    A = aAndRealB[0]
    realB = np.array(aAndRealB[1])

    print('Getting random b vectors...')
    (xs, bs) = gen_rand_b(A, realB, lenX, nTests, maxXValue, multFac)

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
        (x, reason, nIters, r) = lsqr_single_sparse(aSparse, b, aTol, bTol, False, None)
        predictedXs.append(x)
        iters.append(nIters)

        # Compare the x obtained with the one used in the input using program.py
        xFracDiff = pr.compare_x_arrays(x, xs[i])[2]
        if xFracDiff > maxFracDiff:
            print('ERROR: wrong x result for iteration ' + str(i))

        # Make sure the cost was less than maxL in all cases
        L = pr.cost(pr.difference_vector(A, x, b))
        if L > maxL:
            print("The cost is too high: " + str(L))
        costs.append(L)

        # Compute one of the rows manually
        firstRow = A[0][0]
        firstColumn = A[1][0]
        firstValue = A[2][0]
        absDiff = (firstValue * x[firstColumn] - b[firstRow])
        if (b[firstRow] == 0 and absDiff != 0) or (b[firstRow] != 0 and absDiff / b[firstRow] > maxFracDiff):
            print("Difference is too large! " + str(x[firstColumn]) + ", " + str(b[firstRow]))
        fracDiffs.append(absDiff)
        if b[firstRow] != 0:
            fracDiffs[-1] /= b[firstRow]

    # Look at the number of iterations for each test and see if they are roughly constant
    mp.plot(range(len(bs)), iters)
    mp.show()

    # Look at the costs for each test
    mp.plot(range(len(bs)), costs)
    mp.show()

def test_tol():
    aT = 1e-6
    bT = 1e-6

    (x, L, reason, itn, A, b, aSparse) = lsqr('Archive/A.txt', 'Archive/b.txt', 100000, aT, bT)
    b = np.array(b)
    normA = 0
    for val in A[2]:
        normA += (val * val)
    normA = np.sqrt(normA)
    normX = np.sqrt(np.sum(x * x))
    normR = np.sqrt(2*L)
    normB = np.sqrt(np.sum(b * b))
    
    RHS = aT * normA * normX + bT * normB
    LHS = normR
    print(str(LHS) + ", " + str(RHS))
    print("Norm A: " + str(normA))
    print("Norm X: " + str(normX))
    print("Norm R: " + str(normR))
    print("Norm B: " + str(normB))
    print("L: " + str(L))
