import scipy.sparse as sp
import time as tm
import program as pr
import numpy as np
import lsqr as l
import matplotlib.pyplot as mp

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
    aSparse = l.convert_to_sparse_matrix(aLists, len(b), nCols)

    # Try LSQR
    t0 = tm.time()
    r = sp.linalg.lsqr(aSparse, b, atol = aTol, btol = bTol, show = False)
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
    r = sp.linalg.lsmr(aSparse, b, atol = aTol, btol = bTol, show = False)
    t1 = tm.time()
    print('LSMR: ' + str(t1 - t0) + ' s')
    x = r[0]
    dv = pr.difference_vector(aLists, x, b)
    L = pr.cost(dv)
    res = 0.5 * r[3] * r[3]
    if np.abs(L - res) / res > maxErrorL or L > maxL:
        print('ERROR: wrong cost for LSMR')

    # Now try with fake data
    (xs, bs) = l.gen_rand_b(aLists, np.array(b), nCols, nFakeTests, maxXValue, multFac)
    lsqrTimes = []
    lsmrTimes = []
    for i in range(len(bs)):
        t0 = tm.time()
        r = sp.linalg.lsmr(aSparse, bs[i], atol = aTol, btol = bTol)
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
        r = sp.linalg.lsqr(aSparse, bs[i], atol = aTol, btol = bTol)
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
        res = sp.linalg.lsqr(aSparse, b, atol = 1e-6, btol = 1e-7, iter_lim = 150)
        t1 = tm.time()
        L = pr.cost(pr.difference_vector(aLists, res[0], b))
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

    (aLists, b) = pr.read_data(aFile, bFile)
    aSparse = l.convert_to_sparse_matrix(aLists, len(b), nCols)
    
    Ls = []
    for iterLim in iterLims:
        print("Up to " + str(iterLim) + " iterations...")
        res = sp.linalg.lsqr(aSparse, b, atol = 1e-6, btol = 1e-7, iter_lim = iterLim)
        L = pr.cost(pr.difference_vector(aLists, res[0], b))
        Lemp = 0.5 * res[3] * res[3]
        if np.abs(L - Lemp) / Lemp > maxLerror:
            print("ERROR: wrong cost for " + str(iterLim) + " iterations")
        Ls.append(L)
    
    mp.semilogy(range(len(Ls)), Ls)
    mp.show()
