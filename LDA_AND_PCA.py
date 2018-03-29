import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
import seaborn as sns

def PCA(data):

    #plot initial given data

    fig = pyplot.figure()
    ax = Axes3D(fig)
    sequence_containing_x_vals = list(data[:, 0])
    sequence_containing_y_vals = list(data[:, 1])
    sequence_containing_z_vals = list(data[:, 2])
    ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
    pyplot.show()


    mean=np.mean(data,axis=0)  # find the mean of the data

    Z= data-mean #Center the Data
    covariance=(np.dot(np.transpose(Z),Z))/8   #Find the Covariance matrix
    eigen_value,eigen_vector=np.linalg.eig(covariance)  # find eigen values and eigen vectors

    eigen_value=abs(eigen_value)



    eig_pair = [(eigen_value[i], eigen_vector[:, i]) for i in range(len(eigen_value))]
    eig_pair.sort(key=lambda x: x[0], reverse=True)  # sort eigen values and eigen vectors

    projection=data.dot(eig_pair[0][1]) #reduce the dimensions of the data
    plot(projection)



def plot(projection):
    y=np.zeros(8) #making a 0 vector
    plt.scatter(projection,y) # plot new data
    plt.xlabel("PCA Component")
    plt.ylabel("Y Axis")
    plt.title("PCA Result")
    plt.show()


def LDA(negative_data,postive_data):


    fig = pyplot.figure()
    ax = Axes3D(fig)
    #plot positive class
    sequence_containing_x_vals = list(postive_data[:, 0])
    sequence_containing_y_vals = list(postive_data[:, 1])
    sequence_containing_z_vals = list(postive_data[:, 2])
    ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)

    #plot negative class
    sequence_containing_x_vals = list(negative_data[:, 0])
    sequence_containing_y_vals = list(negative_data[:, 1])
    sequence_containing_z_vals = list(negative_data[:, 2])
    ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)

    pyplot.show()

    mean_positive=np.mean(postive_data,axis=0) #find mean of positive class
    mean_negative = np.mean(negative_data, axis=0) # find mean of negative class

    S_positive = (np.transpose(postive_data-mean_positive).dot(postive_data-mean_positive)) #covariance matrix of +ve class
    S_negative = (np.transpose(negative_data - mean_negative).dot(negative_data - mean_negative)) #covariance matrix of -ve class
    Sw=(S_positive)+(S_negative) #within class scatter matrix

    Sb=(mean_positive.reshape(3,1)-mean_negative.reshape(3,1)).dot(np.transpose(mean_positive.reshape(3,1)-mean_negative.reshape(3,1))) #between class scatter matrix
    S=np.linalg.inv(Sw).dot(Sb) #solving Generalized eigen value problem
    eigen_value, eigen_vector = np.linalg.eig(S)  # find eigen values and eigen vectors
    eigen_value=abs(eigen_value)


    eig_pair = [(eigen_value[i], eigen_vector[:, i]) for i in range(len(eigen_value))]
    eig_pair.sort(key=lambda x: x[0], reverse=True)  # sort eigen values and eigen vectors

    projection_positive = postive_data.dot((eig_pair[0][1])) #projection of +class

    projection_negative=negative_data.dot((eig_pair[0][1])) #projection of -ve class


    plot_lda(projection_positive,projection_negative)

def plot_lda(projection_positive,projection_negative):
     #making a 0 vectors
    y = np.zeros(8)
    y1 = np.zeros(5)
    plt.scatter(projection_positive,y,c='b') # plot new data
    plt.scatter(projection_negative, y1,c='orange')
    plt.xlabel("LD1")
    plt.ylabel("Y-axis")
    plt.title("LDA Result")
    plt.show()

if __name__ == "__main__":

    data=np.array([[-2,9,0],[1,3,7],[4,2,-5],[6,-1,3],[5,-4,2],[3,-2,3],[6,-4,4],[2,5,6]]) #data for PCA
    data_lda_positive=np.array([[4,1,0],[2,4,1],[2,3,1],[3,6,0],[4,4,-1],[6,2,0],[3,2,1],[8,3,0]])
    data_lda_negative=np.array([[9,10,1],[6,8,0],[9,5,0],[8,7,1],[10,8,-1]])


    PCA(data)
    LDA(data_lda_negative,data_lda_positive)
