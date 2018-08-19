import random as rd
import numpy as np

# Given the input data, reduce to a smaller size
def reduce_data(nRows, nCols):
    aFile = open('Archive/A.txt', 'r')
    aOutFileLines = 0
    rows = []
    cols = []
    vals = []
    with open('aRed.txt', 'w') as aOutFile:
        for line in aFile:
            elements = line.split()
            row = int(elements[0])
            col = int(elements[1])
            val = float(elements[2])
            if row >= nRows or col >= nCols:
                continue
            aOutFile.write(line)
            rows.append(row)
            cols.append(col)
            vals.append(val)
            aOutFileLines += 1
    aFile.close()
    print("Out file has " + str(aOutFileLines) + " lines")

    # Now generate a random x vector of length nCols
    x = [rd.randint(0, 100)/100. for j in range(nCols)]
    # Save it for book-keeping
    with open('xRed.txt', 'w') as xOutFile:
        for el in x:
            xOutFile.write(str(el) + "\n")

    # Now generate b by multiplying A by x
    b = np.zeros(nRows)
    for i in range(len(rows)):
        b[rows[i]] += (x[cols[i]] * vals[i])
    nonZeroEls = 0
    with open('bRed.txt', 'w') as bOutFile:
        for el in b:
            if el != 0:
                nonZeroEls += 1
            bOutFile.write(str(el) + "\n")
    print("There are " + str(nonZeroEls) + " non-zero elements in b")
    
    A = (rows, cols, vals)
    return (A, b)

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
                print(line)
            if col < smallestCol:
                smallestCol = col
                print(line)
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
                print(line)
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

def difference_vector_dict_and_cost(A, x, b):
    res = []
    theCost = 0
    for i in range(len(b)):
        res.append(-b[i])
        if i in A:
            for col in A[i].keys():
                res[-1] += A[i][col] * x[col]
        else:
            print("Warning: row " + str(i) + " not in A")
        theCost += (res[-1] * res[-1])
    return (res, 0.5 * theCost)

# Return the cost function given the difference vector Ax-b
def cost(diffVec):
    theCost = 0
    for el in diffVec:
        theCost += (el * el)
    return 0.5 * theCost

# Perform a single gradient descent step
def gradient_descent_step(x, learningRate, A, diffVec):
    outX = x.copy()
    for i in range(len(A[0])):
        outX[A[1][i]] -= (learningRate * A[2][i] * diffVec[A[0][i]])
    return outX

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

# Compare a "real" x and the predicted one
def compare_x(x, xFile):
        theXFile = open(xFile, 'r')
        realX = []
        for line in theXFile:
            realX.append(float(line))
        theXFile.close()
        print(compare_x_arrays(x, realX))

# Perform gradient descent on some data
def gradient_descent(aFile, bFile, nCols, learningRate, xFile, maxIter, adaptive, adaptiveFactor):
    # Read the data
    (A,b) = read_data(aFile, bFile)
    # Initialise x
    x = [1 for i in range(nCols)]
    # Perform gradient descent until the stopping condition is satisfied
    L = 1e9
    nTries = 0
    oldL = L
    oldX = x
    oldDiffVec = []
    while nTries < maxIter:
        diffVec = difference_vector(A, x, b)
        L = cost(diffVec)
        if L < 1e-3:
            break
        
        # If the cost worsened, go back and decrease the learning rate
        realL = L
        if oldL < L and nTries > 10 and adaptive:
            x = oldX
            diffVec = oldDiffVec
            learningRate /= (adaptiveFactor * adaptiveFactor)
            L = oldL
        elif oldL > L and nTries > 10 and adaptive and L < 10:
            learningRate *= adaptiveFactor
        oldL = L
        print("Cost = " + str(realL) + ", learning rate " + str(learningRate) + ", iteration " + str(nTries))

        oldX = x
        oldDiffVec = diffVec
        x = gradient_descent_step(x, learningRate, A, diffVec)
        nTries += 1
    print("Number of iterations needed: " + str(nTries))

    # If xFile is provided, compare the original x with the predicted x
    if xFile != "":
        compare_x(x, xFile)

    # Save the result
    with open("xOut.txt", "w") as xOutFile:
        for el in x:
            xOutFile.write(str(el) + "\n")
