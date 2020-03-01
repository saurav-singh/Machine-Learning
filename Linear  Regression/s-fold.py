import numpy
import pandas


def seperateData(dataMatrix):
    X = dataMatrix[:, 1:]
    Y = dataMatrix[:, :1]
    return X, Y


def mergeMatrix(matrix):
    merged = matrix.pop(0)
    for mat in matrix:
        merged = numpy.vstack((merged, mat))
    return merged


def createSFolds(dataMatrix, S):
    try:
        foldMatrx = numpy.vsplit(dataMatrix, S)
        return foldMatrx
    except:
        return createSFolds(dataMatrix, S-1)


def closedLinearRegression(dataMatrix):
    # Split X (features) and Y (output)
    X, Y = seperateData(dataMatrix)

    # Add a column of ones : X = [1 X]
    ones = numpy.ones((X.shape[0], 1), numpy.int8)
    X = numpy.hstack((ones, X))

    # Compute Weights
    A = numpy.linalg.inv(X.T.dot(X))
    B = A.dot(X.T)
    Weights = B.dot(Y)

    return Weights


def testModel(testData, weights):
    # Split X (features) and Y (output)
    X, Y = seperateData(testData)

    # Add a column of ones : X = [1 X]
    ones = numpy.ones((X.shape[0], 1), numpy.int8)
    X = numpy.hstack((ones, X))

    sqError = []

    for i in range(len(X)):
        Ypredict = X[i].dot(weights)
        sqError.append((Y[i][0] - Ypredict[0]) ** 2)

    return sqError


def SFold(dataMAtrix, S):

    foldMatrix = createSFolds(dataMatrix, S)

    # Cross Validation using S-folds
    SqErrors = []
    RMSE = []

    for I in range(20):

        # Shuffly Data Matrix
        numpy.random.seed(I)
        numpy.random.shuffle(dataMatrix)

        for i in range(S):

            # Initialize test and train data
            testData = foldMatrix[i]
            trainData = []

            # Select and merge t raining data
            for j in range(S):
                if i != j:
                    trainData.append(foldMatrix[j])
            trainData = mergeMatrix(trainData)

            # Compute model weights
            weight = closedLinearRegression(trainData)

            # Compute Sqyared error
            SqErrors += testModel(testData, weight)

        # Compute RMSE
        RMSE.append(numpy.sqrt(numpy.mean(SqErrors)))

    # Results
    mean = numpy.mean(RMSE)
    std = numpy.std(RMSE)

    print("for S =", str(S))
    print("Mean =", mean)
    print("Std =", std)


if __name__ == "__main__":

    # Retrieve and parse data
    dataFile = "x06Simple.csv"
    rawData = pandas.read_csv(dataFile)
    dataMatrix = numpy.array(rawData)[:, 1:]

    S_values = [2, 4, 22, len(dataMatrix)]

    for S in S_values:
        SFold(dataFile, S)
        print()
