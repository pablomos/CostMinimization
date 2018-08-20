import program as pr
import numpy as np
import lsqr as l
import scipy.sparse.linalg as la
import not_for_submission as nfs

def test_find_smallest_row_column():
    pr.find_smallest_row_column()

def test_find_smallest_el():
    pr.find_smallest_el()

def test_find_empty_rows():
    pr.find_empty_rows()

def test_read_data():
    (A, b) = pr.read_data('Archive/A.txt', 'Archive/b.txt')
    success = len(A) == 3 and len(A[0]) == len(A[1]) and len(A[0]) == len(A[2]) and len(A) < len(b) and len(b) > 0
    if not success:
        print('ERROR!!! in read data')
    
def test_difference_vector():
    rows = [0, 1, 2]
    cols = [0, 1, 2]
    vals = [1, 3, 5]
    x = [7, 11, 13]
    b = [17, 19, 21]
    dv = pr.difference_vector((rows, cols, vals), x, b)
    success = dv[0] == -10 and dv[1] == 14 and dv[2] == 44
    if not success:
        print('ERROR!!! in difference vector')

def test_cost():
    dv = [1, 3, 5]
    L = pr.cost(dv)
    success = L == 17.5
    if not success:
        print('ERROR!!! in cost')

def test_compare_x_arrays():
    x1 = [1, 3, 5]
    x2 = [7, 11, 13]
    (nx1, nx2, fd) = nfs.compare_x_arrays(x1, x2)
    success = nx1 == np.sqrt(1+9+25) and nx2 == np.sqrt(49+121+169) and fd == np.sqrt(np.sum((np.array(x1) - np.array(x2)) * (np.array(x1) - np.array(x2)))) / nx2
    if not success:
        print('ERROR!!! in compare x arrays')

def test_convert_to_sparse_matrix():
    rows = [0, 1, 2]
    cols = [0, 1, 2]
    vals = [1, 3, 5]
    mal = (rows, cols, vals)
    nRows = 3
    nCols = 3
    sparse = l.convert_to_sparse_matrix(mal, nRows, nCols)
    success = la.norm(sparse) == np.sqrt(1 + 9 + 25)
    if not success:
        print('ERROR!!! in convert to sparse matrix')

def test_create_mock_sparse_matrix():
    sparse = l.create_mock_sparse_matrix()
    if sparse == None:
        print('ERROR!!! in create mock sparse matrix')

def test_mock_lsqr():
    (x,L) = l.mock_lsqr()
    if len(x) < 2 or L < 0:
        print('ERROR!!! in mock LSQR')

def test_lsqr_single():
    A = ([0,1,2], [0,1,2], [1,3,5])
    b = [7,11,13]
    nCols = 3
    aTol = 1e-1
    bTol = 1e-2
    verbose = False
    maxIter = None
    (sparse, res) = l.lsqr_single(A, b, nCols, aTol, bTol, verbose, maxIter)
    success = la.norm(sparse) == np.sqrt(1+9+25) and res != None
    if not success:
        print('ERROR!!! in LSQR single')

def test_lsqr_single_sparse():
    A = l.convert_to_sparse_matrix(([0,1,2], [0,1,2], [1,3,5]), 3, 3)
    b = [7,11,13]
    aTol = 1e-1
    bTol = 1e-2
    verbose = False
    maxIter = None
    res = l.lsqr_single_sparse(A, b, aTol, bTol, verbose, maxIter)
    success = np.all(np.array([res[0][i] - [7, 11./3, 13./5][i] for i in range(len(res[0]))])) and res[1] == 1
    if not success:
        print('ERROR!!! in LSQR single sparse')

def test_lsqr():
    aFile = 'Archive/A.txt'
    bFile = 'Archive/b.txt'
    nCols = 100000
    aTol = 1e-6
    bTol = 1e-7
    verbose = False
    maxIter = None
    (x, L, reason, nIters, aLists, b, aSparse) = l.lsqr(aFile,bFile,nCols,aTol,bTol,verbose,maxIter)
    success = len(x) == nCols and reason == 1
    if not success:
        print('ERROR!!! in LSQR')
        print(len(res[0]))
        print(nCols)
        print(res[2])
        
def test_gen_rand_b():
    A = ([0,1,2], [0,1,2], [1,3,5])
    b = np.array([7,11,13])
    nCols = 3
    nTests = 10
    xMax = 1
    sc = 1e6
    (xs, bs) = nfs.gen_rand_b(A, b, nCols, nTests, xMax, sc)
    success = len(xs) == len(bs) and len(xs) == nTests and len(xs[0]) == nCols and len(bs[0]) == len(b)
    if not success:
        print('ERROR!!! in gen rand b')
    for i in range(len(bs)):
        realB = np.sqrt(np.sum(np.array(b) * np.array(b)))
        fakeB = np.sqrt(np.sum(bs[i] * bs[i]))
        if np.abs(realB - fakeB) / realB > 0.01:
            print('ERROR!!! in gen rand b in the modes of b')
            print(realB)
            print(fakeB)

def test_lsqr_test():
    aTol = 1e-6
    bTol = 1e-7
    nTests = 1
    nfs.lsqr_test(aTol, bTol, nTests)

def test_test_tol():
    nfs.test_tol()

def test_compare_algos():
    nfs.compare_algos()

def test_find_iters():
    nfs.find_iters()

def test_find_matrix_class():
    nfs.find_matrix_class()

def all_tests():
    test_find_smallest_row_column()
    test_find_smallest_el()
    test_find_empty_rows()
    test_read_data()
    test_difference_vector()
    test_cost()
    test_compare_x_arrays()
    test_convert_to_sparse_matrix()
    test_create_mock_sparse_matrix()
    test_mock_lsqr()
    test_lsqr_single()
    test_lsqr_single_sparse()
    test_lsqr()
    test_gen_rand_b()
    test_lsqr_test()
    test_test_tol()
    test_compare_algos()
    test_find_iters()
    test_find_matrix_class()
