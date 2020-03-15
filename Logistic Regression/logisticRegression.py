import pandas
import numpy


class LogisticRegression:

    def __init__(self, learningRate=0.1):
        self.learningRate = learningRate
        self.epsilion = pow(2, -16)
        self.theta = None
        self.classes = None

    def sigmoid(self, x, theta):
        exponent = -1 * numpy.dot(x, theta)
        return 1 / (1 + numpy.exp(exponent))

    def fit(self, X, Y):
        self.Nobservation, self.Nfeatures = X.shape
        self.theta = numpy.random.uniform(-1, 1, (self.Nfeatures, 1))
        Lbefore = 100
        change = 100

        # Gradient Ascent
        while change > self.epsilion:
            # Update Theta
            logistic = self.sigmoid(X, self.theta)
            updateTheta = (self.learningRate /
                           self.Nobservation) * numpy.dot(X.T, (Y-logistic))
            self.theta += updateTheta

            # Compute Likelyhood
            logistic = self.sigmoid(X, self.theta)
            Lafter = (1/self.Nobservation) * numpy.dot(X.T, (Y-logistic))

            # Compute change in likelyhood
            change = numpy.sum(numpy.abs(Lafter-Lbefore))
            Lbefore = Lafter

    def predict(self, X):
        probability = self.sigmoid(X, self.theta)
        predict = 1 if probability >= 0.5 else 0
        return predict


# Data Processing Helper Functions


def seperateData(dataMatrix):
    X = dataMatrix[:, :-1]
    Y = dataMatrix[:, -1:]
    return X, Y


def getMeanStd(dataMatrix):
    mean = numpy.mean(dataMatrix, axis=0)
    std = numpy.std(dataMatrix, axis=0, ddof=1)
    return mean, std


def standardize(dataMatrix, mean, std):
    return (dataMatrix - mean) / std


if __name__ == "__main__":

    # ----- Data Processing
    # ---------------------

    # Read Data
    rawData = pandas.read_csv("spambase.data")
    dataMatrix = numpy.array(rawData)

    # Randomize Data
    numpy.random.seed(0)
    numpy.random.shuffle(dataMatrix)

    # Split train-test set
    N = numpy.shape(dataMatrix)[0]
    splitIndex = round((2 * N) / 3)
    trainData = dataMatrix[:splitIndex, :]
    testData = dataMatrix[splitIndex:, :]

    # ----- Training Model
    # ---------------------

    # Seperate feature X and output Y
    X, Y = seperateData(trainData)

    # Standardie features X
    mean, std = getMeanStd(X)
    X = standardize(X, mean, std)

    # Add a bias column : X = [1 X]
    ones = numpy.ones((X.shape[0], 1), numpy.int8)
    X = numpy.hstack((ones, X))

    # Train Naive Bayes Model
    model = LogisticRegression()
    model.fit(X, Y)

    # ----- Testing Model
    # ---------------------

    # Seperate feature X and output Y
    X, Y = seperateData(testData)

    # Standardie features X (mean & std from training data)
    X = standardize(X, mean, std)

    # Add a bias column : X = [1 X]
    ones = numpy.ones((X.shape[0], 1), numpy.int8)
    X = numpy.hstack((ones, X))

    # Compute classifications
    TP = FN = FP = TN = 0

    for i in range(len(X)):
        Yp = model.predict(X[i])

        # True Positive
        if (Y[i] == 1 and Yp == 1):
            TP += 1
        # False Negative
        if (Y[i] == 1 and Yp == 0):
            FN += 1
        # False Positive
        if (Y[i] == 0 and Yp == 1):
            FP += 1
        # True Negative
        if (Y[i] == 0 and Yp == 0):
            TN += 1

    # Calculate Precision, Recall, F-measure, Accuracy
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    fmeasure = (2 * precision * recall) / (precision + recall)
    accuracy = (TP + TN) / (TP + FN + FP + TN)

    # Format Results
    precision = round(precision * 100, 3)
    recall = round((recall * 100), 3)
    fmeasure = round((fmeasure*100), 3)
    accuracy = round((accuracy*100), 3)

    # Display Results
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F-measure:", fmeasure)
