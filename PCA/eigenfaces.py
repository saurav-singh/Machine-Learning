from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy
from cv2 import cv2


def processImage(imageSource):
    image = Image.open(imageSource)
    image = image.resize((40, 40))
    image = numpy.array(image)
    return image.flatten()


def determineStat(dataMatrix):
    mean = numpy.mean(dataMatrix, axis=0)
    std = numpy.std(dataMatrix, axis=0, ddof=1)
    return mean, std


def standardize(dataMatrix, mean, std):
    return (dataMatrix - mean) / std


def unstandardize(dataMatrix, mean, std):
    return (dataMatrix * std) + mean


def determineEigen(dataMatrix):
    features = dataMatrix.T
    covarianceMatrix = numpy.cov(features)
    return numpy.linalg.eig(covarianceMatrix)


def projectData(dataMatrix, eigenVectors, K):
    dataMatrix = numpy.array(dataMatrix)
    reducedData = []
    for i in range(K):
        projection = dataMatrix.dot(eigenVectors.T[i])
        reducedData.append(projection)
    return reducedData


def reconstruct(dataMatrix, eigenVectors, K):
    dataMatrix = numpy.array(dataMatrix)
    reconstruction = dataMatrix[K] * eigenVectors.T[:, K]
    return reconstruction


def refineImage(imageMatrix):
    index = imageMatrix > 255
    imageMatrix[index] = 255
    indexLow = imageMatrix < 0
    imageMatrix[indexLow] = 0
    return imageMatrix


if __name__ == "__main__":

    # Retrieve list of image files
    data_directory = "yalefaces"
    data_files = os.listdir(data_directory)
    data_files.remove("Readme.txt")
    data_matrix = []

    # Process the images into data matrix
    for imageSource in data_files:
        data_row = processImage(data_directory + "/" + imageSource)
        data_matrix.append(data_row)

    # Standardize data matrix
    data_matrix = numpy.array(data_matrix)
    mean, std = determineStat(data_matrix)
    #data_matrix = standardize(data_matrix, mean, std)

    # Compute PCA
    eigenValues, eigenVectors = determineEigen(data_matrix)

    # Project subject into K most important PCA
    subject = data_directory+"/subject02.centerlight"
    subject_data = processImage(subject)
    subject_data = standardize(subject_data, mean, std)

    for i in range(1600):
        pca = eigenVectors[:, :i+1]
        A = subject_data.dot(pca)
        B = A.dot(pca.T)
        B = unstandardize(B, mean, std)
        B = numpy.real(B)
        B = refineImage(B)
        B = B.reshape(40, 40)
        img = Image.fromarray(B.astype("uint8"))
        img.save("images/"+str(i)+".jpg")

    # Make Video
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    videoWriter = cv2.VideoWriter("output.avi", fourcc, 25, (40, 40))

    images = os.listdir("images")
    for image in images:
        frame = cv2.imread("images/" + image)
        videoWriter.write(frame)

    videoWriter.release()
