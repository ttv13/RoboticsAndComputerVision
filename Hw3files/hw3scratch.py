import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

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
    resizeImg = cv2.resize(img, (640, 480))
    gray = cv2.cvtColor(resizeImg, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)



 # If found, add object points, image points (after refining them)
    if ret:
        print(f"Found corners in {fname}")
        objpoints.append(objp)

        # Refine the corners
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)


        img = cv2.drawChessboardCorners(resizeImg, checkerboard_size, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(1000)
    else:
        print(f"No corners found in {fname}")

cv2.destroyAllWindows()


#calibrate camera
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


print("Camera matrix (K):\n", K)
# print("\nR matrices and t vectors for 3 of the camera views:\n")

# for i in range(3):
#     print(f"View {i+1}:")
#     R, _ = cv2.Rodrigues(rvecs[i])
#     print("R:\n", R)
#     print("t:\n", tvecs[i])
#     print()

# # Plot the camera locations
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# for i in range(len(rvecs)):
#     R, _ = cv2.Rodrigues(rvecs[i])
#     camera_location = -np.matrix(R).T * np.matrix(tvecs[i])
#     ax.scatter(camera_location[0], camera_location[1], camera_location[2], c='b', marker='o')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.title('Camera Locations in World Coordinates')
# plt.show()
