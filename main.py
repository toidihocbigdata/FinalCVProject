from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import cv2
import sys
from PIL import Image
import numpy as np
from webcam import Webcam
from glyphs.constants import *
from objloader import *

us = True
if us is not True:
    from glyphs.glyphs import Glyphs
else:
    from detect_and_track import *
  
class AR:
    # constants
    INVERSE_MATRIX = np.array([[ 1.0, 1.0, 1.0, 1.0],
                               [-1.0,-1.0,-1.0,-1.0],
                               [-1.0,-1.0,-1.0,-1.0],
                               [ 1.0, 1.0, 1.0, 1.0]])
  
    def __init__(self):
        # initialise webcam and start thread
        self.webcam = Webcam()
        self.webcam.start()
  
        # initialise detector
        if us is not True:
            self.glyphs = Glyphs()
        else:
            self.detector = Detector(   model_path="new_icon_detector_model/keras_model.h5", 
                                        camera_matrix_path="camera_parameters.json")
  
        # initialise shapes
        self.cone = None
        self.sphere = None
        self.isShapeSwitch = False;

        # initialise texture
        self.texture_background = None
 
        # initialise view matrix
        self.view_matrix = np.array([])

        # refine arg
        self.deltaX = 0.5
        self.deltaY = 0.3
        self.deltaZ = 0.0
 
    def initGL(self, Width, Height):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(33.7, 1.3, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

        # assign shapes
        # self.cone       = OBJ('resources/basic_object/cone.obj')
        # self.sphere     = OBJ('resources/basic_object/sphere.obj')
        self.pikachu    = OBJ('resources/Pikachu_B/Pikachu_B.obj')
        self.skull      = OBJ('resources/Skull/12140_Skull_v3_L2.obj')
  
        # assign texture
        glEnable(GL_TEXTURE_2D)
        self.texture_background = glGenTextures(1)
        self.texture_object = glGenTextures(1)

    def buildViewMatrix(self, label, rvec, tvec):
        # build view matrix
        rmtx = cv2.Rodrigues(rvec)[0]

        if label == 2: # skull
            self.deltaX = 0.315
            self.deltaY = 0.3
            tvec = tvec + np.array([[self.deltaX], [self.deltaY], [self.deltaZ]])
        elif label == 1:
            self.deltaX = 0.5
            self.deltaY = 0.3
            tvec = tvec + np.array([[self.deltaX], [self.deltaY], [self.deltaZ]])
        else:
            pass
        # TODO 
        self.view_matrix  = np.array(   [[rmtx[0][0],   rmtx[0][1],     rmtx[0][2],     tvec[0]],
                                        [rmtx[1][0],    rmtx[1][1],     rmtx[1][2],     tvec[1]],
                                        [rmtx[2][0],    rmtx[2][1],     rmtx[2][2],     tvec[2]],
                                        [0.0       ,    0.0       ,     0.0       ,     1.0     ]])

        self.view_matrix = self.view_matrix * self.INVERSE_MATRIX

        self.view_matrix = np.transpose(self.view_matrix)
  
    def drawScene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
  
        # get image from webcam
        image = self.webcam.get_current_frame()
  
        # convert image to OpenGL texture format
        bg_image = cv2.flip(image, 0)
        bg_image = cv2.flip(bg_image, 1)
        bg_image = Image.fromarray(bg_image)     
        ix = bg_image.size[0]
        iy = bg_image.size[1]
        bg_image = bg_image.tobytes("raw", "BGRX", 0, -1)

        # create background texture
        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_image)

        # draw background
        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glPushMatrix()
        glTranslatef(0.0,0.0,-10.0)
        self.drawBackground()
        glPopMatrix()

        # handle glyphs
        self.handleImage(image)

        glutSwapBuffers()

    def init_object_texture(self,image_filepath):
        #THE COMMENTS FOR THE FOLLOWING STEPS ARE SAME AS IN THE ABOVE FUNCTION, THE ONLY DIFFERENCE IS, HERE INSTEAD OF USING CAMERA FRAME WE USE IMAGE FILES
        tex = cv2.imread(image_filepath)
        tex = cv2.flip(tex, 0)
        tex = Image.fromarray(tex)
        ix = tex.size[0]
        iy = tex.size[1]
        tex = tex.tobytes("raw","BGRX", 0, -1)
        # if self.texture_object is None:
        #     self.texture_object = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_object)
        glEnable(GL_TEXTURE_2D)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex)
        return None

    def render3Dobj(self, label):
        if label == 0:
            self.init_object_texture("resources/basic_object/wave-textures.png")
            glRotate(180, 0, 1, 0)
            glutSolidTeapot(0.3) #HARCODE
        elif label == 1:
            glRotate(190, 0, 1, 0)
            glCallList(self.pikachu.gl_list)
        elif label == 2:
            glScalef(0.02, 0.02, 0.02)
            glRotate(90, 0, 0, 1)
            glRotate(90, 1, 0, 0)
            glRotate(90, 0, 1, 0)
            glCallList(self.skull.gl_list)
        else:
            pass

    def handleImage(self, image):

        # attempt to detect glyphs
        results = []

        try:
            if us is not True:
                # results = self.glyphs.detect(image) #TODO create detector
                results = self.glyphs.detect(cv2.flip(image, 1)) #TODO create detector
            else:
                results = self.detector.main_detect(image)
        except Exception as ex:
            print("Exception !: ")
            print(ex)

        if not results: 
            return

        for ret in results:
            rvec, tvec, label = ret
            
            # print(tvec)
            # build view matrix
            tvec = tvec + np.array([[self.deltaX], [self.deltaY], [self.deltaZ]])

            self.buildViewMatrix(label, rvec, tvec)

            # load view matrix and draw shape
            glPushMatrix()
            glLoadMatrixf(self.view_matrix)
            self.render3Dobj(label) #TODO
            glPopMatrix()

    def drawBackground(self):
        # draw background
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex3f(-4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex3f( 4.0, -3.0, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex3f( 4.0,  3.0, 0.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-4.0,  3.0, 0.0)
        glEnd()

    def main(self):
        # setup and run OpenGL
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(640, 480)
        glutInitWindowPosition(800, 400)
        self.window_id = glutCreateWindow("OpenGL Window")
        glutDisplayFunc(self.drawScene)
        glutIdleFunc(self.drawScene)
        self.initGL(640, 480)
        glutMainLoop()

# run an instance of AR
ar = AR()
ar.main()