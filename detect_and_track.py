import cv2
import argparse
import numpy as np
import math
from utils import *
import tensorflow.keras
from PIL import Image, ImageOps
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import json

class Detector:
    def __init__(self, model_path=None, camera_matrix_path=None):
        # load model for detect icon:
        if model_path is None:
            self.icon_model = tensorflow.keras.models.load_model('keras_model.h5')
        else: 
            self.icon_model = tensorflow.keras.models.load_model(model_path)
        # data array for icon:
        self.icon_data = np.ndarray(shape = (1, 224, 224, 3), dtype=np.float32)
        self.icon_size = (224,224)
        #
        if camera_matrix_path is None:
            self.camera_matrix =  np.array([[519.89300222,   0.,         307.20316512],
                                            [  0.,         517.02897669, 227.13200295],
                                            [  0.,           0.,          1.        ]])
        else: 
            with open("./camera_parameters.json", 'r') as f:
                self.camera_matrix = np.array(json.load(f))
        #
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params,search_params)
        self.orb = cv2.xfeatures2d.SIFT_create()
        self.threshold = 350
        self.bandwidth = None
        self.track = False
        self.counter = 0
        self.show = False
        self.refPt = [] 
        self.temp = []
        self.frame = None
        self.gray = None
        self.k = None
        self.img = None
        self.kp1 = None
        self.des1 = None
        self.kp2 = None
        self.des2 = None
        self.rvecs = None
        self.tvecs = None
        self.label = 0

    def detect_icon(self, cut_image):
        image_icon = Image.fromarray(cut_image)
        image_icon = ImageOps.fit(image_icon, self.icon_size, Image.ANTIALIAS)
        #turn the image into a numpy array
        image_array = np.asarray(image_icon)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # Load the image into the array
        self.icon_data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        #resize the image to a 224x224 with the same strategy as in TM2:
        #resizing the image to be at least 224x224 and then cropping from the center
        self.icon_data[0] = normalized_image_array
        # run the inference
        label = self.icon_model.predict(self.icon_data)
        # print("label_detected", np.argmax(label))
        return label

    def click_and_crop(self, event, x, y, flags, param):
        # grab references to the global variables
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt.append((x, y))
            self.temp.append((x,y))
        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            self.refPt.append((x, y))
            # draw a rectangle around the region of interest
            cv2.rectangle(self.frame, self.refPt[0], self.refPt[1], (0, 255, 0), 2)
            cv2.imshow("frame", self.frame)
            self.img = self.gray[self.temp[0][1]:self.temp[1][1],self.temp[0][0]:self.temp[1][0]]
            # test_text = cv2.resize(test_text,(self.img.shape[1],self.img.shape[0]))
            self.temp = []
            self.kp1, self.des1 = self.orb.detectAndCompute(self.img,None)
            self.track = True
            cv2.imshow('img',self.img)
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            try:
                self.temp[1] = (x,y)
            except:
                self.temp.append((x,y))

    def find_rvec_tvec(self,homography):
        #def projection_matrix(camera_parameters, homography):
        """
        From the camera calibration matrix and the estimated homography
        compute the 3D projection matrix
        """
        # Compute rotation along the x and y axis as well as the translation
        homography = homography * (-1)
        rot_and_transl = np.dot(np.linalg.inv(self.camera_matrix), homography)
        col_1 = rot_and_transl[:, 0]
        col_2 = rot_and_transl[:, 1]
        col_3 = rot_and_transl[:, 2]
        # normalise vectors
        l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
        rot_1 = col_1 / l
        rot_2 = col_2 / l
        tvec = col_3 / l
        tvec = tvec / -260.0 # Hardcode
        # compute the orthonormal basis
        c = rot_1 + rot_2
        p = np.cross(rot_1, rot_2)
        d = np.cross(c, p)
        rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
        rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
        rot_3 = np.cross(rot_1, rot_2)
        R = np.stack((rot_1,rot_2,rot_3)).T
        rvec = cv2.Rodrigues(R)[0]
        return rvec,tvec

    def main_detect(self, frame):
        #flip the frame coming from webcam
        self.frame = cv2.flip(frame, 1)
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        #set callback to control the mouse
        cv2.setMouseCallback("frame", self.click_and_crop)

        if len(self.temp) == 2:
            cv2.rectangle(self.frame, self.temp[0], self.temp[1], (0, 255, 0), 2)
        if self.track:
            cv2.destroyAllWindows()
            matchExist = True
            while matchExist:
                matchExist = False
                self.kp2, self.des2 = self.orb.detectAndCompute(self.gray,None)
                matches_ = self.flann.knnMatch(self.des1,self.des2,k=2)
                n,lbls = keypoint_clustering([self.kp2[m.trainIdx] for m,_ in matches_])
                for i in range(n):
                    matches = np.array(matches_)[lbls==i].tolist()
                    matches = sorted(matches,key=lambda x: x[0].distance)
                    matches = [match for match in matches if match[0].distance<self.threshold]
                    #matches = matches[:20]
                    if len(matches) > (0.75*len(matches_)/n):
                        src_pts = np.float32([self.kp1[m.queryIdx].pt for m,_ in matches[:50]]).reshape(-1,1,2)
                        dst_pts = np.float32([self.kp2[m.trainIdx].pt for m,_ in matches[:50]]).reshape(-1,1,2)
                        M,mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
                        self.rvecs , self.tvecs = self.find_rvec_tvec(M)
                        h,w = self.img.shape
                        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
                        dst = cv2.perspectiveTransform(pts,M)
                        # projection = projection_matrix(self.camera_matrix,M)
                        self.frame = cv2.polylines(self.frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
                        cv2.fillPoly(self.gray,[np.int32(dst)],255)
                    else: 
                        matchExist = False

            # cv2.imshow('frame',self.frame)
            self.tvecs.shape = (3,1)
            return [[self.rvecs, self.tvecs, self.label]]
        else:
            cv2.imshow('frame',self.frame)
            return 0
        
        
