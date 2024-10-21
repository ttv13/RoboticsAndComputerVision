#problem 1 

import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Termination criteria for cornerSubPix function for corner refinement 
criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)


# Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0)
#  the size of the checkerboard num of inner corners
checkerboard_size = (7, 5)
objp = np.zeros((np.prod(checkerboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)


# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.


# Load the images
images = glob.glob('calibrationobj/*.jpg')
print(f'Loaded {len(images)} images.')


for fname in images:
    img = cv2.imread(fname)

    print('Processing image %s...' % fname)
    # resizeImg = cv2.resize(img, (640, 480))
    # gray = cv2.cvtColor(resizeImg, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)



 # If found, add object points, image points (after refining them)
    if ret:
        print(f"Found corners in {fname}")
        objpoints.append(objp)

        # Refine the corners
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)


        # img = cv2.drawChessboardCorners(resizeImg, checkerboard_size, corners2, ret)
        img = cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
        cv2.imshow("img", img)
        cv2.waitKey(100)
    else:
        print(f"No corners found in {fname}")

cv2.destroyAllWindows()


#calibrate camera
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


print("Intrinsic matrix (K):\n", K)
print ("r and t ", rvecs[1])
print("\nR matrices and t vectors for 3 of the camera views:\n")

for i in range(3):
    print(f"View {i+1}:")
    #rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvecs[i])

    print("R:\n", R)
    print("t:\n", tvecs[i],"\n")

# Plot the camera locations
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#plotting checkerboard
Cbplot = np.array([[0,0,0], [checkerboard_size[0],0,0], [checkerboard_size[0],checkerboard_size[1],0], [0,checkerboard_size[1],0]])
ax.plot(Cbplot[:, 0], Cbplot[:, 1], Cbplot[:, 2], 'r-')

for i in range(len(rvecs)):
    #rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvecs[i])

    #camera location (with respect to world coordinate system)
    wtc = -np.matrix(R).T * np.matrix(tvecs[i])
    wtc = np.array(wtc).flatten()
    ax.scatter(wtc[0], wtc[1], -wtc[2], c='b', marker='o')
    ax.text(wtc[0], wtc[1], -wtc[2], f'{i+1}', size=10, zorder=1, color='k')  # Adding labels

    #camera orientation
    camera_direction = R.T @ np.array([0, 0, 1]).reshape(-1, 1)
    camera_direction = camera_direction.flatten()
    ax.quiver(wtc[0], wtc[1], -wtc[2], camera_direction[0], camera_direction[1], -camera_direction[2], color='r')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Camera Locations in World Coordinates')
plt.show()
