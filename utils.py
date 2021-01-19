import cv2
import numpy as np
import math
from sklearn.cluster import MeanShift, estimate_bandwidth,KMeans
def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1/l
    rot_2 = col_2/l
    translation = col_3/l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def keypoint_clustering(kp):
    x = np.array([p.pt for p in kp])
    #bandwidth = estimate_bandwidth(x, quantile=0.1, n_samples=10)
    ms = KMeans(2)#MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False)
    ms.fit(x)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    #
    print(np.unique(labels,return_counts=True)[1])
    n_cluster = len(np.unique(labels))
    return n_cluster,labels

def draw_axis(img, R, t, K):
    # unit is mm
    #rotV, _ = cv2.Rodrigues(R)
    points = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(-1, 3)
    print(R.shape)
    print(points.shape)
    axisPoints = 1000*R.dot(points.T)#cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    for i in range(4):
        axisPoints[:,i]-=t
    img = cv2.line(img, tuple(axisPoints[:,3].ravel().astype('int')[:2]), tuple(axisPoints[:,0].ravel().astype('int')[:2]), (255,0,0), 3)
    img = cv2.line(img, tuple(axisPoints[:,3].ravel().astype('int')[:2]), tuple(axisPoints[:,1].ravel().astype('int')[:2]), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[:,3].ravel().astype('int')[:2]), tuple(axisPoints[:,2].ravel().astype('int')[:2]), (0,0,255), 3)
    return img