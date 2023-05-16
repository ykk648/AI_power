# -- coding: utf-8 --
# @Time : 2022/11/11
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import numpy as np


def inverse_cv_affine(mat):
    """
    similar to mat_rev = cv2.invertAffineTransform(mat)
    Args:
        mat:
    Returns:
    """
    # inverse the Affine transformation matrix
    mat_rev = np.zeros([2, 3])
    div1 = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
    mat_rev[0][0] = mat[1][1] / div1
    mat_rev[0][1] = -mat[0][1] / div1
    mat_rev[0][2] = -(mat[0][2] * mat[1][1] - mat[0][1] * mat[1][2]) / div1
    div2 = mat[0][1] * mat[1][0] - mat[0][0] * mat[1][1]
    mat_rev[1][0] = mat[1][0] / div2
    mat_rev[1][1] = -mat[0][0] / div2
    mat_rev[1][2] = -(mat[0][2] * mat[1][0] - mat[0][0] * mat[1][2]) / div2
