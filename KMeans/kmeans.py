import numpy
import pandas
import random
from cv2 import cv2
from sklearn import decomposition
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#-----------------------------------------
# K-means function
#-----------------------------------------

def myKMeans(X, Y, K):
    # Limit Value of K between 1 - 7
    K = 7 if K > 7 else K
    K = 1 if K < 1 else K

    # Reduce dimensionality to <= 3
    if X.shape[1] > 3:
        pca = decomposition.PCA(3)
        X = pca.fit_transform(X)

    # Total Observations
    totalObs = X.shape[0]

    # Random Seed
    random.seed(0)

    # Randomly select K centroids
    centroids = {}
    for i in range(K):
        randomIndex = random.randint(0, totalObs - 1)
        centroids[i] = X[randomIndex]

    # Terminal tolerance condition
    epsilon = pow(2, -23)

    # Graph Image Frames for video
    imageFrames = []

    # Cluster using K-means
    for I in range(500):

        # Initialize clusters and output matrix
        clusters = {}
        clusterY = {}
        for i in range(K):
            clusters[i] = []
            clusterY[i] = []

        # Cluster each feature based on similarity
        for i in range(totalObs):
            # Compute Distance between each observation and centroids
            distances = []
            for centroid in centroids:
                distances.append(distance.euclidean(X[i], centroids[centroid]))
            # Retrieve Cluster index
            clusterIndex = numpy.argmin(distances)
            # Put correspoinding observation and result into their cluster
            clusters[clusterIndex].append(X[i])
            clusterY[clusterIndex].append(Y[i][0])

        # Compute Purity
        purity = computePurity(clusterY)

        # Plot the data and store image array
        imageData = plot(clusters, centroids, I, purity)
        imageFrames.append(imageData)

        # Save previous centroid
        prevCentroids = dict(centroids)

        # Update Centroid
        for cluster in clusters:
            centroids[cluster] = numpy.mean(clusters[cluster], axis=0)

        # Check Terminal Process
        change = 0
        for c in centroids:
            change += distance.cityblock(centroids[c], prevCentroids[c])
        if change < epsilon:
            break

    # Generate Video
    generateVideo(imageFrames)

    return

#-----------------------------------------
# Computes avg purity of a cluster
#-----------------------------------------

def computePurity(clusterY):
    purity = []

    for cluster in clusterY:
        N = len(clusterY[cluster])
        posCount = clusterY[cluster].count(1)
        negCount = clusterY[cluster].count(-1)
        maxCount = max(posCount, negCount)
        P = maxCount/N
        purity.append(P)

    purity = sum(purity)/(len(purity))

    return round(purity, 5)

#-----------------------------------------
# Plots clusters and returns image array
#-----------------------------------------

def plot(clusters, centroids, iter, purity):
    dimension = centroids[0].shape[0]
    info = "Iteration " + str(iter + 1) + " Purity = " + str(purity)
    color = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    graph = plt.figure()

    # 2D - Graph
    if dimension == 2:
        # Plot Clusters
        for cluster in clusters:
            currCluster = clusters[cluster]
            for feature in currCluster:
                X, Y = feature[0], feature[1]
                plt.scatter(X, Y, marker='x',
                            c=color[cluster], s=5, linewidth=0.3)
        # Plot Centroids
        for centroid in centroids:
            X, Y = centroids[centroid][0], centroids[centroid][1]
            plt.scatter(X, Y, marker='o', c=color[centroid])

    # 3D - Graph
    if dimension == 3:
        ax = graph.add_subplot(111, projection="3d")
        # Plot Clusters
        for cluster in clusters:
            currCluster = clusters[cluster]
            for feature in currCluster:
                X, Y, Z = feature[0], feature[1], feature[2]
                ax.scatter(X, Y, Z, marker='x',
                           c=color[cluster], s=5, linewidth=0.3)
        # Plot Centroids
        for centroid in centroids:
            X = centroids[centroid][0]
            Y = centroids[centroid][1]
            Z = centroids[centroid][2]
            ax.scatter(X, Y, Z, marker='o', c=color[centroid])

    # Title
    plt.title(info)
    
    # Parse to image array
    graph.canvas.draw()
    imageData = numpy.frombuffer(
        graph.canvas.tostring_rgb(), dtype=numpy.uint8)
    imageData = imageData.reshape(graph.canvas.get_width_height()[::-1] + (3,))
    plt.clf()

    return imageData

#-----------------------------------------
# Function to generate video
#-----------------------------------------

def generateVideo(imageFrames):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = 3
    videoWriter = cv2.VideoWriter("output.avi", fourcc, fps, (640, 480))
    for frame in imageFrames:
        videoWriter.write(frame)
    videoWriter.release()

#-----------------------------------------
# Helper Function for Data Manipulation
#-----------------------------------------

def getMeanStd(dataMatrix):
    mean = numpy.mean(dataMatrix, axis=0)
    std = numpy.std(dataMatrix, axis=0, ddof=1)
    return mean, std


def seperateData(dataMatrix):
    X = dataMatrix[:, 1:]
    Y = dataMatrix[:, :1]
    return X, Y


def standardize(dataMatrix, mean, std):
    return (dataMatrix - mean) / std


def unstandardize(dataMatrix, mean, std):
    return (dataMatrix * std) + mean

#-----------------------------------------
# Main
#-----------------------------------------

if __name__ == "__main__":

    # Read Data
    dataFile = "diabetes.csv"
    rawData = pandas.read_csv(dataFile)
    dataMatrix = numpy.array(rawData)

    # Seperate class label from observable data
    X, Y = seperateData(dataMatrix)

    # Standaradzie Data
    mean, std = getMeanStd(X)
    X = standardize(X, mean, std)

    # K-means
    myKMeans(X, Y, 3)