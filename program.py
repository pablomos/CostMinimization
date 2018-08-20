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
