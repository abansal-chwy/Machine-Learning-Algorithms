import scipy.io
import numpy as np
from scipy.interpolate import *
import numpy.core.numeric as NX
import matplotlib.pyplot as plt



#to find the new output y using cofficients
def find_y2(a,x):
    p = NX.asarray(a)

    x = NX.asarray(x)
    y = NX.zeros_like(x)
    for i in range(len(p)):
        y = y * x + p[i]
    return y

#to find the cofficients for degree 1
def Regression_degree_1(x,y):
    vanderX=np.vander(x,2) #get vandemonde matrix
    a = np.linalg.inv(np.transpose(vanderX).dot(vanderX)).dot(np.transpose(vanderX).dot(y)) # get the cofficients
    y2 = find_y2(a, x) #get the expected y
    print("Cofficients for degree 1 are :", a)
    print(" ")
    print("Error for degree 1 is",1-(sum(pow(y-y2,2)))/sum(pow(y-np.mean(y),2)))
    print("")
    plt.scatter(x, y)
    plt.scatter(x, y2)
    plt.show()

#to find the cofficients for degree 3
def Regression_degree_3(x,y):
    vanderX=np.vander(x,4) #get vandemonde matrix
    a = np.linalg.inv(np.transpose(vanderX).dot(vanderX)).dot(np.transpose(vanderX).dot(y)) # get the cofficients
    y2 = find_y2(a, x) #get the expected y
    print(y2)
    print("Cofficients for degree 3 are :", a)
    print("")
    print("Error for degree 3 is",1-(sum(pow(y-y2,2)))/sum(pow(y-np.mean(y),2)))
    print("")
    plt.scatter(x, y)
    plt.scatter(x, y2)
    plt.show()

#to find the cofficients for degree 5
def Regression_degree_5(x,y):
    vanderX=np.vander(x,6) #get vandemonde matrix
    a = np.linalg.inv(np.transpose(vanderX).dot(vanderX)).dot(np.transpose(vanderX).dot(y)) # get the cofficients
    y2 = find_y2(a, x) #get the expected y
    print("Cofficients for degree 5 are :", a)
    print("")
    print("Error for degree 5 is",1-(sum(pow(y-y2,2)))/sum(pow(y-np.mean(y),2)))
    print("")
    plt.scatter(x, y)
    plt.scatter(x, y2)
    plt.show()

#to find the cofficients for degree 7
def Regression_degree_7(x,y):
    vanderX=np.vander(x,8) #get vandemonde matrix
    a = np.linalg.inv(np.transpose(vanderX).dot(vanderX)).dot(np.transpose(vanderX).dot(y)) # get the cofficients
    y2 = find_y2(a, x) #get the expected y
    print("Cofficients for degree 7 are :", a)
    print("Error for degree 7 is ",1-(sum(pow(y-y2,2)))/sum(pow(y-np.mean(y),2)))
    plt.scatter(x, y)
    plt.scatter(x, y2)
    plt.show()




if __name__ == "__main__":
    data = scipy.io.loadmat("data.mat")
    x = (data['x']).ravel()  #change the shape to  1-D
    y = (data['y']).ravel()
    x = NX.asarray(x) + 0.0
    y = NX.asarray(y) + 0.0
    Regression_degree_1(x,y)
    Regression_degree_3(x, y)
    Regression_degree_5(x, y)
    Regression_degree_7(x, y)