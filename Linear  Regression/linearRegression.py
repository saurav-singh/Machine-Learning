import numpy
import pandas


def seperateData(dataMatrix):
    X = dataMatrix[:, 1:]
    Y = dataMatrix[:, :1]
    return X, Y


if __name__ == "__main__":

    # Retrieve and parse data
    dataFile = "x06Simple.csv"
    rawData = pandas.read_csv(dataFile)
    dataMatrix = numpy.array(rawData)[:, 1:]

    # Shuffly Data Matrix
    numpy.random.seed(0)
    numpy.random.shuffle(dataMatrix)

    # Split data for training and testing
    N = numpy.shape(dataMatrix)[0]
    splitIndex = round((2 * N) / 3)
    trainData = dataMatrix[:splitIndex, :]
    testData = dataMatrix[splitIndex:, :]

    # Closed Form Linear Regression
    X, Y = seperateData(trainData)

    # Add a column of ones : X = [1 X]
    ones = numpy.ones((X.shape[0], 1), numpy.int8)
    X = numpy.hstack((ones, X))

    # Compute Weights
    A = numpy.linalg.inv(X.T.dot(X))
    B = A.dot(X.T)
    Weights = B.dot(Y)

    # Test computed weights
    _X, _Y = seperateData(testData)

    # Add a column of ones : X = [1 X]
    ones = numpy.ones((_X.shape[0], 1), numpy.int8)
    _X = numpy.hstack((ones, _X))

    # Compute Root Mean Squared Error (RMSE)
    sqError = []

    for i in range(len(_X)):
        Ypredict = _X[i].dot(Weights)
        sqError.append((_Y[i][0] - Ypredict[0]) ** 2)
    
    weights = Weights.flatten()
    
    # Display Results
    RMSE = str(numpy.sqrt(numpy.mean(sqError)))
    MODEL = str(round(weights[0], 3))
    for i in range(1, len(weights)):
        MODEL += " + " + str(round(weights[i], 2)) + " x" + str(i)

    print("RSME = ", RMSE)
    print("MODEL =", MODEL)
