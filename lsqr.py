import scipy.sparse as sp
import scipy.sparse.linalg as la
import sys

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
    return la.lsqr(A, b, atol = aTol, btol = bTol, show = verbose, iter_lim = maxIter)[:4]

# Try LSQR with the entire data
# aTol and bTol represent the actual error in the input data
# The optimal number of maxIter found in find_iters()
def lsqr(aFile, bFile, nCols, aTol = 1e-6, bTol = 1e-7, verbose = True, maxIter = 150):
    # Read the data
    (aLists,b) = read_data(aFile, bFile)
    (aSparse, (x, reason, nIters, r)) = lsqr_single(aLists, b, nCols, aTol, bTol, verbose, maxIter)
    L = 0.5 * r * r
    return (x, L, reason, nIters, aLists, b, aSparse)

# Find the smallest row and column numbers in A
def find_smallest_row_column():
    smallestRow = 1e6
    smallestCol = 1e6
    aFile = open('Archive/A.txt', 'r')
    with open('aRed.txt', 'w') as aOutFile:
        aLines = 0
        for line in aFile:
            aLines = aLines + 1
            if aLines % 100000 == 0:
                print(aLines)
            elements = line.split()
            row = int(elements[0])
            col = int(elements[1])
            if row < smallestRow:
                smallestRow = row
            if col < smallestCol:
                smallestCol = col
    aFile.close()
    print("Smallest row " + str(smallestRow) + ", col " + str(smallestCol))

# Find the element closest to the 0,0 slot
def find_smallest_el():
    smallestEl = 1e6
    aFile = open('Archive/A.txt', 'r')
    with open('aRed.txt', 'w') as aOutFile:
        aLines = 0
        for line in aFile:
            aLines = aLines + 1
            if aLines % 100000 == 0:
                print(aLines)
            elements = line.split()
            row = int(elements[0])
            col = int(elements[1])
            el = max(row, col)
            if el < smallestEl:
                smallestEl = el
    aFile.close()
    print("Smallest element " + str(smallestEl))

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

# Return Ax-b given A, x, b
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

    (x, L, reason, nIters, aLists, b, aSparse) = lsqr(sys.argv[1], sys.argv[2], nCols, verbose = False)
    with open('out.txt', 'w') as outf:
        for val in x:
            outf.write(str(val) + '\n')
    print("Saved output in " + outFile)
