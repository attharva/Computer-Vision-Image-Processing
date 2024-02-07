from matplotlib import image
import numpy as np
from typing import List, Tuple
import cv2

from cv2 import cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners
from regex import R
from sympy import LM, im

'''
Please do Not change or add any imports. 
'''

#task1

def findRot_xyz2XYZ(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles along x, y and z axis respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from xyz to XYZ.

    '''
    
    rot_xyz2XYZ = np.eye(3).astype(float)
    alpha= np.deg2rad(alpha)
    beta= np.deg2rad(beta)
    gamma= np.deg2rad(gamma)

    Rz = np.matrix([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0,0,1] ])
    
    Rx = np.matrix([[1,0,0], [0, np.cos(beta), -np.sin(beta)], [0,np.sin(beta),np.cos(beta)] ])
    
    Rz1 = np.matrix([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0,0,1] ])
  

    rot_xyz2XYZ =  Rz1 @ Rx @ Rz
    return rot_xyz2XYZ


def findRot_XYZ2xyz(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles of the 3 step respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from XYZ to xyz.

    '''
    

    # Your implementation
  
    alpha= -np.deg2rad(alpha)
    beta= -np.deg2rad(beta)
    gamma= -np.deg2rad(gamma)

    Rz = np.matrix([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0,0,1] ])
    
    Rx = np.matrix([[1,0,0], [0, np.cos(beta), -np.sin(beta)], [0,np.sin(beta),np.cos(beta)] ])
    
    Rz1 = np.matrix([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0,0,1] ])
  
    rot_XYZ2xyz =  Rz @ Rx @ Rz1 
    return rot_XYZ2xyz

"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above "findRot_xyz2XYZ()" and "findRot_XYZ2xyz()" functions are the only 2 function that will be called in task1.py.
"""

# Your functions for task1






#--------------------------------------------------------------------------------------------------------------
# task2:

def find_corner_img_coord(image: np.ndarray) -> np.ndarray:
    '''
    Args: 
        image: Input image of size MxNx3. M is the height of the image. N is the width of the image. 3 is the channel of the image.
    Return:
        A numpy array of size 32x2 that represents the 32 checkerboard corners' pixel coordinates. 
        The pixel coordinate is defined such that the of top-left corner is (0, 0) and the bottom-right corner of the image is (N, M). 
    '''
    img_coord = np.zeros([32, 2], dtype=float)

    # Your implementation


    TrFa, givenCornersInCheckboard = cv2.findChessboardCorners(image, (9,4))
   
    givenCornersInCheckboard = np.squeeze(givenCornersInCheckboard)
    


    givenCornersInCheckboard = np.delete(givenCornersInCheckboard, 4,axis=0)
    givenCornersInCheckboard = np.delete(givenCornersInCheckboard, 12,axis=0)
    givenCornersInCheckboard = np.delete(givenCornersInCheckboard, 20,axis=0)
    givenCornersInCheckboard = np.delete(givenCornersInCheckboard, 28,axis=0)

    # print(TrFa)
    # print(givenCornersInCheckboard)

    return (givenCornersInCheckboard)






       


def find_corner_world_coord(img_coord: np.ndarray) -> np.ndarray:
    '''
    You can output the world coord manually or through some algorithms you design. Your output should be the same order with img_coord.
    Args: 
        img_coord: The image coordinate of the corners. Note that you do not required to use this as input, 
        as long as your output is in the same order with img_coord.
    Return:
        A numpy array of size 32x3 that represents the 32 checkerboard corners' pixel coordinates. 
        The world coordinate or each point should be in form of (x, y, z). 
        The axis of the world coordinate system are given in the image. The output results should be in milimeters.
    '''
    # world_coord = np.zeros([32, 3], dtype=float)

    # Your implementation

   



    world_coord = [
    [0, 40, 40],
    [40,  0, 10],
    [30,  0, 10],
    [20,  0, 10],
    [10,  0, 10],
    [ 0, 10, 10],
    [ 0, 20, 10],
    [ 0, 30, 10],
    [ 0, 40, 10],
    [40,  0, 20],
    [30,  0, 20],
    [20,  0, 20],
    [10,  0, 20],
    [ 0, 10, 20],
    [ 0, 20, 20],
    [ 0, 30, 20],
    [ 0, 40, 20],
    [40,  0, 30],
    [30,  0, 30],
    [20,  0, 30],
    [10,  0, 30],
    [ 0, 10, 30],
    [ 0, 20, 30],
    [ 0, 30, 30],
    [ 0, 40, 30],
    [40,  0, 40],
    [30,  0, 40],
    [20,  0, 40],
    [10,  0, 40],
    [ 0, 10, 40],
    [ 0, 20, 40],
    [ 0, 30, 40],
    [ 0, 40, 40]
    ]


    return np.array(world_coord)


def find_intrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[float, float, float, float]:
    '''
    Use the image coordinates and world coordinates of the 32 point to calculate the intrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        fx, fy: Focal length. 
        (cx, cy): Principal point of the camera (in pixel coordinate).
    '''

    fx: float = 0
    fy: float = 0
    cx: float = 0
    cy: float = 0

    mat = []
    for i in range(0,32):
        j = [world_coord[i][0], world_coord[i][1], world_coord[i][2], 1,0,0,0,0,(-1 * img_coord[i][0]* world_coord[i][0]),(-1 * img_coord[i][0]* world_coord[i][1]), (-1 * img_coord[i][0]* world_coord[i][2]),(-1 * img_coord[i][0])]
        mat.append(j)
        k = [0,0,0,0, world_coord[i][0], world_coord[i][1], world_coord[i][2],1,(-1 * img_coord[i][1]* world_coord[i][0]),(-1 * img_coord[i][1]* world_coord[i][1]), (-1 * img_coord[i][1]* world_coord[i][2]),(-1 * img_coord[i][1])]
        mat.append(k)
    matArray = np.array(mat)

    u,s,v = np.linalg.svd(matArray)

    matr = v[-1].reshape(3,4)
    x = matr[:,0:-1]
    _, inv = np.linalg.qr(np.linalg.inv(x))
    im = np.linalg.inv(inv)
    im = im * 1000000 * -1
    fx = im[0][0]
    fy = im[1][1]
    cx = im[0][2]
    cy = im[1][2]



    # Your implementation

    return fx, fy, cx, cy


def find_extrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Use the image coordinates, world coordinates of the 32 point and the intrinsic parameters to calculate the extrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        R: The rotation matrix of the extrinsic parameters. It is a 3x3 numpy array.
        T: The translation matrix of the extrinsic parameters. It is a 1-dimensional numpy array with length of 3.
    '''

    R = np.eye(3).astype(float)
    T = np.zeros(3, dtype=float)

    # Your implementation

    return R, T


"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above 4 functions are the only ones that will be called in task2.py.
"""

# Your functions for task2







#---------------------------------------------------------------------------------------------------------------------