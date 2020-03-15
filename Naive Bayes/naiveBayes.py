import pandas
import numpy


class NaiveBayes:

    def fit(self, X, Y):
        self.classes = numpy.unique(Y)
        self.Nobservation, self.Nfeatures = X.shape

        # Seperate data for each class
        classData = {c: [] for c in self.classes}
        for i in range(self.Nobservation):
            classData[Y[i][0]].append(X[i])

        # Initialize Priros, Mean, Std
        self.priors = {}
        self.mean = {}
        self.std = {}

        # Calculate Priors, Mean, Std
        for c in self.classes:
            self.priors[c] = len(classData[c])
            self.mean[c] = numpy.mean(classData[c], axis=0)
            self.std[c] = numpy.std(classData[c], axis=0, ddof=1)

    def predict(self, x):
        posteriors = []
        for c in self.classes:
            prob = numpy.log(self.priors[c])
            for i in range(self.Nfeatures):
                prob += self.gaussianModel(x, i, c)
            posteriors.append(prob)

        classIndex = numpy.argmax(posteriors)
        return self.classes[classIndex]

    def gaussianModel(self, x, i, c):
        mean = self.mean[c][i]
        std = self.std[c][i]
        x = x[i]
        numerator = numpy.exp(- (x-mean)**2 / (2 * std**2))
        denominator = std * numpy.sqrt(2*numpy.pi)
        prob = numerator/denominator
        prob = -9999 if prob == 0 else numpy.log(prob)
        return prob

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
    numpy.random.seed(1)
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

    # Train Naive Bayes Model
    model = NaiveBayes()
    model.fit(X, Y)

    # ----- Testing Model
    # ---------------------

    # Seperate feature X and output Y
    X, Y = seperateData(testData)

    # Standardie features X (mean & std from training data)
    X = standardize(X, mean, std)

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
