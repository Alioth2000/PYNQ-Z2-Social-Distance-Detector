# 双目相机标定参数

import cv2
import numpy as np

left_camera_matrix = np.array([[499.1876, 1.2895, 342.7303],
                               [0., 497.7724, 255.7097],
                               [0., 0., 1.]])
# left_distortion = np.array([[0.0560, 0.4234, -0.0056, 0.0031, -1.3149]])
left_distortion = np.array([[0.0, 0., -0., 0.0, -0.]])
right_camera_matrix = np.array([[498.6942, 1.0367, 289.8898],
                                [0., 497.0368, 254.0299],
                                [0., 0., 1.]])
# right_distortion = np.array([[0.0349, 0.6468, -0.0051, 0.0018, -2.2025]])
right_distortion = np.array([[0.0, 0., -0., 0.0, -0.]])
R = np.matrix([
    [0.999962591997135, 0.000604400029751327, -0.00862840118305652],
    [-0.000599080887536805, 0.999999628945507, 0.000619040336142804],
    [0.00862877212944707, -0.000613848068841369, 0.999962583041029],
])

# print(R)

T = np.array([[-59.6858], [-0.0519], [-0.8964]])  # 平移关系向量

size = (640, 480)  # 图像尺寸

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
