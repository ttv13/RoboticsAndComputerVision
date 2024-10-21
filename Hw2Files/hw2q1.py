import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import transform

def findHomography(sidePoints, flatPoints):
    #Create the A matrix and find the homography by SVD
    A = []
    for i in range(len(sidePoints)):
        x, y = sidePoints[i][0], sidePoints[i][1]
        u, v = flatPoints[i][0], flatPoints[i][1]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])

    A = np.array(A)
    U, S, VT = np.linalg.svd(A)
    L = VT[-1, :] / VT[-1, -1]
    Homography = L.reshape(3, 3)
    return Homography

 
def warpImage(image, homography, output_shape):
    #Warp the image
    return transform.warp(image, np.linalg.inv(homography), output_shape=output_shape)

def main ():
    #import side view of image and select 4 points
    sideImage = cv2.imread('sideImage.jpg')
    sideImagegrey = cv2.cvtColor(sideImage, cv2.COLOR_BGR2GRAY)

    plt.imshow(sideImagegrey, cmap = 'gray')
    plt.title('Select 4 points')
    sidePoints = plt.ginput(4)
    plt.close()


    #Getting size of image and selecting coresponding points in front view
    height ,width = sideImagegrey.shape

    
    flatPoints = np.array([[0,0], [width,0], [width,height], [0,height]])

    #Estimate the homography
    homography = findHomography(sidePoints, flatPoints)
    print("Homography Matrix: \n",homography)

    #Warp the image
    warped_image = warpImage(sideImagegrey, homography, (height, width))

    #Plot the original and warped images
    plt.subplot(1, 2, 1)
    plt.title('Original Side View')
    plt.imshow(sideImagegrey,cmap = 'gray')
    plt.subplot(1, 2, 2)
    plt.title('Warped Frontal View')
    plt.imshow(warped_image,cmap='gray')
    plt.show()

if __name__ == "__main__": 
    main()





