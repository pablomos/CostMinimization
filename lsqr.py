import scipy.sparse as sp
import scipy.sparse.linalg as la
import sys

'''
Create CSR matrix from lists of rows, columns and values
There are various sparse matrix classes available in the package. 
This one was the fastest in validation_tests.find_matrix_class()
'''
def convert_to_sparse_matrix(A, nRows, nCols):
    return sp.csr_matrix((A[2], (A[0], A[1])), shape = (nRows, nCols))

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
    AS = create_mock_sparse_matrix()
    b = [11, 13, 17]
    (x, reason, nIters, r) = lsqr_single_sparse(AS, b, 1e-8, 1e-8, False, None)
    return (x, 0.5 * r * r)

# Compute LSQR for a given A,b
# A is of the form (rows, cols, vals)
def lsqr_single(A, b, nCols, aTol, bTol, verbose, maxIter):
    AS = convert_to_sparse_matrix(A, len(b), nCols)
    return (AS, lsqr_single_sparse(AS, b, aTol, bTol, verbose, maxIter))

'''
Compute LSQR for a given AS, b
AS is a Python sparse matrix
This method seems to implement the LSQR algorithm presented in 
Paige & Saunders (1982): 
https://web.stanford.edu/class/cme324/paige-saunders2.pdf
We also tried LSMR, featured in the same package, but it was slower as found by
validation_tests.compare_algos()
'''
def lsqr_single_sparse(AS, b, aTol, bTol, verbose, maxIter):
    return la.lsqr(AS, b, atol = aTol, btol = bTol, show = verbose, iter_lim = maxIter)[:4]

# Try LSQR with the entire data
def lsqr(aFile, bFile, nCols, aTol, bTol, maxIter, verbose = False):
    # Read the data
    (A,b) = read_data(aFile, bFile)
    (AS, (x, reason, nIters, r)) = lsqr_single(A, b, nCols, aTol, bTol, verbose, maxIter)
    L = 0.5 * r * r
    return (x, L, reason, nIters, A, b, AS)

# Read the data from the input files and create data structures
def read_data(aFile, bFile):
    aFile = open(aFile, 'r')
    rows = []
    cols = []
    vals = []
    for line in aFile:
        elements = line.split()
        row = int(elements[0])
        col = int(elements[1])
        val = float(elements[2])
        rows.append(row)
        cols.append(col)
        vals.append(val)
    aFile.close()

    bFile = open(bFile, 'r')
    b = []
    for line in bFile:
        b.append(float(line))
    bFile.close()

    return [[rows, cols, vals], b]

# Return Ax-b given A, x, b; A is of the form (rows, columns, values)
def difference_vector(A, x, b):
    res = [-el for el in b]
    for i in range(len(A[0])):
        res[A[0][i]] += (A[2][i] * x[A[1][i]])
    return res

# Return the cost function given the difference vector Ax-b
def cost(diffVec):
    theCost = 0
    for el in diffVec:
        theCost += (el * el)
    return 0.5 * theCost

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("ERROR: wrong number of arguments")
        print("Usage: " + str(sys.argv[0]) + " <A file> <B file>")
        exit(-1)

    # Stated in the problem description:
    nCols = 100000
    outFile = 'out.txt'
    # ----------------------------------

    # aTol and bTol represent the actual error in the input data
    aTol = 1e-6
    bTol = 1e-7
    # The optimal number of maxIter found in validation_tests.find_iters()
    maxIter = 150

    (x, L, reason, nIters, A, b, AS) = lsqr(sys.argv[1], sys.argv[2], nCols, aTol, bTol, maxIter)
    with open('out.txt', 'w') as outf:
        for val in x:
            outf.write(str(val) + '\n')
    print("Saved output in " + outFile)
