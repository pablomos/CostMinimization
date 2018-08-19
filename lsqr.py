import scipy.sparse as sp
import scipy.sparse.linalg as la
import numpy as np
import program as pr
import matplotlib.pyplot as mp
import random as rd
import time as tm

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
def lsqr_single(A, b, nCols, aTol, bTol, verbose, maxIter):
    t0 = tm.time()
    aSparse = convert_to_sparse_matrix(A, len(b), nCols)
    t1 = tm.time()
    print(str(t1 - t0) + " s converting to a sparse matrix")
    return (aSparse, lsqr_single_sparse(aSparse, b, aTol, bTol, verbose, maxIter))

# Compute LSQR for a given A, b
# A is a Python sparse matrix
def lsqr_single_sparse(A, b, aTol, bTol, verbose, maxIter):
    t0 = tm.time()
    r = la.lsqr(A, b, atol = aTol, btol = bTol, show = verbose, iter_lim = maxIter)[:4]
    t1 = tm.time()
    print(str(t1 - t0) + " s running the actual algorithm")
    return r

# Compute the cost function; A must be of the form (rows, cols, vals)
def cost(A, b, x):
    return pr.cost(pr.difference_vector(A, x, b))

# Try LSQR with the entire data
# aTol and bTol represent the actual error in the input data
# The optimal number of maxIter found in find_iters()
def lsqr(aFile, bFile, nCols, aTol = 1e-6, bTol = 1e-7, verbose = True, maxIter = 150):
    tStart = tm.time()
    t0 = tm.time()
    # Read the data
    (aLists,b) = pr.read_data(aFile, bFile)
    t1 = tm.time()
    print(str(t1 - t0) + " s before lsqr_single")
    (aSparse, (x, reason, nIters, r)) = lsqr_single(aLists, b, nCols, aTol, bTol, verbose, maxIter)
    t0 = tm.time()
    L = 0.5 * r * r
    t1 = tm.time()
    print(str(t1 - t0) + ' s calculating the cost ' + str(L))
    retVal = (x, L, reason, nIters, aLists, b, aSparse)
    tEnd = tm.time()
    print(str(tEnd - tStart) + " s in total")
    return retVal

# Generate random b vectors
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
        (x, reason, nIters) = lsqr_single_sparse(aSparse, b, aTol, bTol, False)
        predictedXs.append(x)
        iters.append(nIters)

        # Compare the x obtained with the one used in the input using program.py
        xFracDiff = pr.compare_x_arrays(x, xs[i])[2]
        if xFracDiff > maxFracDiff:
            print('ERROR: wrong x result for iteration ' + str(i))

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

# This doesn't work because it's meant for square matrices
def gcrot():
    aFile = 'Archive/a.txt'
    bFile = 'Archive/b.txt'
    nCols = 100000

    (A,b) = pr.read_data(aFile, bFile)
    aSparse = convert_to_sparse_matrix(A, len(b), len(b))
    (x, info) = la.gcrotmk(aSparse, b)
    return (x, info)

def lsmr(aTol, bTol):
    aFile = 'Archive/a.txt'
    bFile = 'Archive/b.txt'
    nCols = 100000

    (A,b) = pr.read_data(aFile, bFile)
    aSparse = convert_to_sparse_matrix(A, len(b), nCols)
    r = la.lsmr(aSparse, b, show = True, atol = aTol, btol = bTol)
    return (r, A, b)

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

# Compare lsqr and lsmr
# This method shows a consistently slower performance for LSMR
def compare_algos():
    aFile = 'Archive/A.txt'
    bFile = 'Archive/b.txt'
    nCols = 100000
    maxL = 1e-3
    aTol = 1e-6
    bTol = 1e-7
    maxErrorL = 1e-2
    nFakeTests = 10
    maxXValue = 10
    multFac = 1. / aTol
    
    # Read the data
    (aLists,b) = pr.read_data(aFile, bFile)
    # Convert to a sparse matrix
    aSparse = convert_to_sparse_matrix(aLists, len(b), nCols)

    # Try LSQR
    t0 = tm.time()
    r = la.lsqr(aSparse, b, atol = aTol, btol = bTol, show = False)
    t1 = tm.time()
    print('LSQR: ' + str(t1 - t0) + ' s')
    x = r[0]
    dv = pr.difference_vector(aLists, x, b)
    L = pr.cost(dv)
    res = 0.5 * r[3] * r[3]
    if np.abs(L - res) / res > maxErrorL or L > maxL:
        print('ERROR: wrong cost for LSQR')
    
    # Try LSMR
    t0 = tm.time()
    r = la.lsmr(aSparse, b, atol = aTol, btol = bTol, show = False)
    t1 = tm.time()
    print('LSMR: ' + str(t1 - t0) + ' s')
    x = r[0]
    dv = pr.difference_vector(aLists, x, b)
    L = pr.cost(dv)
    res = 0.5 * r[3] * r[3]
    if np.abs(L - res) / res > maxErrorL or L > maxL:
        print('ERROR: wrong cost for LSMR')

    # Now try with fake data
    (xs, bs) = gen_rand_b(aLists, np.array(b), nCols, nFakeTests, maxXValue, multFac)
    lsqrTimes = []
    lsmrTimes = []
    for i in range(len(bs)):
        t0 = tm.time()
        r = la.lsmr(aSparse, bs[i], atol = aTol, btol = bTol)
        t1 = tm.time()
        lsmrTimes.append(t1 - t0)
        x = r[0]
        dv = pr.difference_vector(aLists, x, bs[i])
        L = pr.cost(dv)
        res = 0.5 * r[3] * r[3]
        if np.abs(L - res) / res > maxErrorL or L > maxL:
            print('ERROR: wrong cost for LSMR')
        dx = np.array([(x[j] - xs[i][j]) for j in range(len(x))])
        if np.sqrt(np.sum(dx * dx)) / np.sqrt(np.sum(xs[i] * xs[i])) > maxErrorL:
            print('ERROR: wrong x result for LSMR')
        
        t0 = tm.time()
        r = la.lsqr(aSparse, bs[i], atol = aTol, btol = bTol)
        t1 = tm.time()
        lsqrTimes.append(t1 - t0)
        x = r[0]
        dv = pr.difference_vector(aLists, x, bs[i])
        L = pr.cost(dv)
        res = 0.5 * r[3] * r[3]
        if np.abs(L - res) / res > maxErrorL or L > maxL:
            print('ERROR: wrong cost for LSQR')
        dx = np.array([(x[j] - xs[i][j]) for j in range(len(x))])
        if np.sqrt(np.sum(dx * dx)) / np.sqrt(np.sum(xs[i] * xs[i])) > maxErrorL:
            print('ERROR: wrong x result for LSQR')
    
    # Plot the times
    mp.plot(range(len(lsqrTimes)), lsqrTimes, color='red')
    mp.plot(range(len(lsmrTimes)), lsmrTimes, color='blue')
    mp.show()

# Find the optimal number of iterations
def find_iters():
    aFile = 'Archive/A.txt'
    bFile = 'Archive/b.txt'
    nCols = 100000
    maxL = 1e-3
    maxLerror = 1e-2

    iterLims = [50, 100, 150, 200, 250, 300, 350, None]

    (aLists, b) = pr.read_data(aFile, bFile)
    aSparse = convert_to_sparse_matrix(aLists, len(b), nCols)
    
    Ls = []
    times = []
    for iterLim in iterLims:
        print("Up to " + str(iterLim) + " iterations...")
        t0 = tm.time()
        res = la.lsqr(aSparse, b, atol = 1e-6, btol = 1e-7, iter_lim = iterLim)
        t1 = tm.time()
        L = pr.cost(pr.difference_vector(aLists, res[0], b))
        Lemp = 0.5 * res[3] * res[3]
        if np.abs(L - Lemp) / Lemp > maxLerror:
            print("ERROR: wrong cost for " + str(iterLim) + " iterations")
        Ls.append(L)
        times.append(t1 - t0)
    
    mp.semilogy(range(len(Ls)), Ls)
    mp.show()
    mp.plot(range(len(times)), times)
    mp.show()

# Try different kinds of spare matrix classes
# CSR and CSC seem roughly equivalent; COO is a bit slower; DOK is too slow
def find_matrix_class():
    aFile = 'Archive/A.txt'
    bFile = 'Archive/b.txt'
    nCols = 100000
    maxL = 1e-3
    maxLerror = 1e-2

    (aLists, b) = pr.read_data(aFile, bFile)
    vals = aLists[2]
    rows = aLists[0]
    cols = aLists[1]
    nRows = len(b)
    A = (vals, (rows, cols))
    myShape = (nRows, nCols)

    aSparses = []
    aSparses.append(sp.csr_matrix(A, shape = myShape))
    aSparses.append(sp.coo_matrix(A, shape = myShape))
    aSparses.append(sp.csc_matrix(A, shape = myShape))
    #aSparses.append(sp.dok_matrix(aSparses[0]))
    
    times = []
    for aSparse in aSparses:
        print("Class " + str(type(aSparse)))
        t0 = tm.time()
        res = la.lsqr(aSparse, b, atol = 1e-6, btol = 1e-7, iter_lim = 150)
        t1 = tm.time()
        L = pr.cost(pr.difference_vector(aLists, res[0], b))
        Lemp = 0.5 * res[3] * res[3]
        if np.abs(L - Lemp) / Lemp > maxLerror or L > 1e-3:
            print("ERROR: wrong cost for class " + str(type(aSparse)))
        times.append(t1 - t0)

    mp.plot(range(len(times)), times)
    mp.show()
