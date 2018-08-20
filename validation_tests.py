import scipy.sparse as sp
import time as tm
import numpy as np
import lsqr as l
import matplotlib.pyplot as mp
import random as rd

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
    (aLists,b) = l.read_data(aFile, bFile)
    # Convert to a sparse matrix
    aSparse = l.convert_to_sparse_matrix(aLists, len(b), nCols)

    # Try LSQR
    t0 = tm.time()
    r = sp.linalg.lsqr(aSparse, b, atol = aTol, btol = bTol, show = False)
    t1 = tm.time()
    print('LSQR: ' + str(t1 - t0) + ' s')
    x = r[0]
    dv = l.difference_vector(aLists, x, b)
    L = l.cost(dv)
    res = 0.5 * r[3] * r[3]
    if np.abs(L - res) / res > maxErrorL or L > maxL:
        print('ERROR: wrong cost for LSQR')
    
    # Try LSMR
    t0 = tm.time()
    r = sp.linalg.lsmr(aSparse, b, atol = aTol, btol = bTol, show = False)
    t1 = tm.time()
    print('LSMR: ' + str(t1 - t0) + ' s')
    x = r[0]
    dv = l.difference_vector(aLists, x, b)
    L = l.cost(dv)
    res = 0.5 * r[3] * r[3]
    if np.abs(L - res) / res > maxErrorL or L > maxL:
        print('ERROR: wrong cost for LSMR')

    # Now try with fake data
    (xs, bs) = gen_rand_b(aLists, np.array(b), nCols, nFakeTests, maxXValue, multFac)
    lsqrTimes = []
    lsmrTimes = []
    for i in range(len(bs)):
        t0 = tm.time()
        r = sp.linalg.lsmr(aSparse, bs[i], atol = aTol, btol = bTol)
        t1 = tm.time()
        lsmrTimes.append(t1 - t0)
        x = r[0]
        dv = l.difference_vector(aLists, x, bs[i])
        L = l.cost(dv)
        res = 0.5 * r[3] * r[3]
        if np.abs(L - res) / res > maxErrorL or L > maxL:
            print('ERROR: wrong cost for LSMR')
        dx = np.array([(x[j] - xs[i][j]) for j in range(len(x))])
        if np.sqrt(np.sum(dx * dx)) / np.sqrt(np.sum(xs[i] * xs[i])) > maxErrorL:
            print('ERROR: wrong x result for LSMR')
        
        t0 = tm.time()
        r = sp.linalg.lsqr(aSparse, bs[i], atol = aTol, btol = bTol)
        t1 = tm.time()
        lsqrTimes.append(t1 - t0)
        x = r[0]
        dv = l.difference_vector(aLists, x, bs[i])
        L = l.cost(dv)
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


# Try different kinds of spare matrix classes
# CSR and CSC seem roughly equivalent; COO is a bit slower; DOK is too slow
def find_matrix_class():
    aFile = 'Archive/A.txt'
    bFile = 'Archive/b.txt'
    nCols = 100000
    maxL = 1e-3
    maxLerror = 1e-2

    (aLists, b) = l.read_data(aFile, bFile)
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
        res = sp.linalg.lsqr(aSparse, b, atol = 1e-6, btol = 1e-7, iter_lim = 150)
        t1 = tm.time()
        L = l.cost(l.difference_vector(aLists, res[0], b))
        Lemp = 0.5 * res[3] * res[3]
        if np.abs(L - Lemp) / Lemp > maxLerror or L > 1e-3:
            print("ERROR: wrong cost for class " + str(type(aSparse)))
        times.append(t1 - t0)

    mp.plot(range(len(times)), times)
    mp.show()


# Find the optimal number of iterations
def find_iters():
    aFile = 'Archive/A.txt'
    bFile = 'Archive/b.txt'
    nCols = 100000
    maxL = 1e-3
    maxLerror = 1e-2

    iterLims = [50, 100, 150, 200, 250, 300, 350, None]

    (aLists, b) = l.read_data(aFile, bFile)
    aSparse = l.convert_to_sparse_matrix(aLists, len(b), nCols)
    
    Ls = []
    for iterLim in iterLims:
        print("Up to " + str(iterLim) + " iterations...")
        res = sp.linalg.lsqr(aSparse, b, atol = 1e-6, btol = 1e-7, iter_lim = iterLim)
        L = l.cost(l.difference_vector(aLists, res[0], b))
        Lemp = 0.5 * res[3] * res[3]
        if np.abs(L - Lemp) / Lemp > maxLerror:
            print("ERROR: wrong cost for " + str(iterLim) + " iterations")
        Ls.append(L)
    
    mp.semilogy(range(len(Ls)), Ls)
    mp.show()

# Try LSQR with multiple b vectors
def lsqr_test(aTol, bTol, nTests):
    lenX = 100000
    maxL = 1e-3
    maxFracDiff = 1e-2
    maxXValue = 10
    multFac = 1. / bTol

    print("Reading the real data...")
    aAndRealB = l.read_data('Archive/a.txt', 'Archive/b.txt')
    A = aAndRealB[0]
    realB = np.array(aAndRealB[1])

    print('Getting random b vectors...')
    (xs, bs) = gen_rand_b(A, realB, lenX, nTests, maxXValue, multFac)

    # For each b vector, find x using lsqr
    print("Converting to sparse matrix...")
    aSparse = l.convert_to_sparse_matrix(A, len(realB), lenX)
    predictedXs = []
    iters = []
    costs = []
    fracDiffs = []
    for i in range(len(bs)):
        print("Test " + str(i))
        b = bs[i]
        (x, reason, nIters, r) = l.lsqr_single_sparse(aSparse, b, aTol, bTol, False, None)
        predictedXs.append(x)
        iters.append(nIters)

        # Compare the x obtained with the one used in the input using program.py
        xFracDiff = compare_x_arrays(x, xs[i])[2]
        if xFracDiff > maxFracDiff:
            print('ERROR: wrong x result for iteration ' + str(i))

        # Make sure the cost was less than maxL in all cases
        L = l.cost(l.difference_vector(A, x, b))
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

# Compare a "real" x and the predicted one
def compare_x_arrays(x, realX):
        if len(realX) != len(x):
            print("Different numbers of elements in real and predicted x!")
            return -1
        xSq = 0
        diffX = 0
        normPredictedX = 0
        for i in range(len(realX)):
            if x[i] == 0:
                continue
            xSq += (realX[i] * realX[i])
            normPredictedX += (x[i] * x[i])
            diffX += ((realX[i] - x[i]) * (realX[i] - x[i]))
        normRealX = np.sqrt(xSq)
        normPredictedX = np.sqrt(normPredictedX)
        fracDiff = np.sqrt(diffX) / normRealX
        return (normPredictedX, normRealX, fracDiff)

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

def test_tol():
    aT = 1e-6
    bT = 1e-6

    (x, L, reason, itn, A, b, aSparse) = l.lsqr('Archive/A.txt', 'Archive/b.txt', 100000, aT, bT)
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

# Generate random bs and save them
def save_random_bs():
    aFile = 'Archive/A.txt'
    bFile = 'Archive/b.txt'
    lenX = 100000
    nTests = 100
    xMax = 10
    sc = 1e6

    (A,b) = l.read_data(aFile, bFile)
    (xs, bs) = gen_rand_b(A, np.array(b), lenX, nTests, xMax, sc)
    
    for i in range(len(xs)):
        with open('fakes/b' + str(i) + '.txt', 'w') as outb:
            for b in bs[i]:
                outb.write(str(b) + '\n')
        with open('fakes/x' + str(i) + '.txt', 'w') as outx:
            for x in xs[i]:
                outx.write(str(x) + '\n')

# Test random bs
def test_random_bs():
    nTests = 100
    aFile = 'Archive/A.txt'
    maxL = 1e-3
    nCols = 100000

    for i in range(nTests):
        print('Test ' + str(i))
        bFile = 'fakes/b' + str(i) + '.txt'
        xFile = 'fakes/x' + str(i) + '.txt'
        (A,b) = l.read_data(aFile, bFile)
        (x, L, reason, nIters, aLists, b, aSparse) = l.lsqr(aFile, bFile, nCols, verbose = False)
        myCost = l.cost(l.difference_vector(A, x, b))
        if myCost > maxL:
            print('Error in file ' + str(i))

# Find any rows that have no entries
def find_empty_rows():
    aFile = open('Archive/A.txt', 'r')
    dict = {}
    for line in aFile:
        elements = line.split()
        row = int(elements[0])
        dict[row] = 1
    aFile.close()

    i = 0
    bFile = open('Archive/b.txt', 'r')
    with open('bRed.txt', 'w') as bOutFile:
        for line in bFile:
            bVal = float(line)
            if ((not i in dict) or (dict[i] != 1)) and bVal != 0:
                print("Row " + str(i) + " is empty. B value is " + line)
            i = i + 1
    bFile.close()

def generate_high_b():
    with open('fakes/b0.txt', 'r') as inf:
        with open('fakes/bLarge.txt', 'w') as outf:
            for line in inf:
                outf.write(str(float(line) * 100) + '\n')

    
