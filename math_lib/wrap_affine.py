# -- coding: utf-8 --
# @Time : 2022/12/28
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import cv2
import numpy as np
from cv2box import CVImage, MyTimer
import cvcuda
import nvcv
import torch

# opencv default version
img_p = ''
img = CVImage(img_p).bgr
crop_size = 256
mat_ = np.array([[1.77893761e-01, 2.47390154e-03, -9.42742635e+01], [-2.47390154e-03, 1.77893761e-01, -3.40511541e+01]])
mat_rev = cv2.invertAffineTransform(mat_)
with MyTimer() as mfc:
    for i in range(10000):  # 31.5 fps
        warped = cv2.warpAffine(img, mat_, (crop_size, crop_size), borderValue=0.0)
CVImage(warped).show()


# opencv cuda version
image_tensors = torch.tensor(CVImage(img_p).bgr).unsqueeze(0).cuda()
# print(image_tensors)
print(image_tensors.size())
cvcuda_input_tensor = cvcuda.as_tensor(image_tensors, "NHWC")
cvcuda_output_tensor = cvcuda.Tensor([1, 256, 256, 3], np.uint8, "NHWC")
print(cvcuda_input_tensor.shape)
cvcuda_affine_tensor = cvcuda.warp_affine_into(src=cvcuda_input_tensor, dst=cvcuda_output_tensor, xform=mat_rev,
                                               flags=cvcuda.Interp.LINEAR, border_mode=cvcuda.Border.CONSTANT,
                                               border_value=[0])
print(cvcuda_affine_tensor.shape)
print(type(cvcuda_affine_tensor))
print(np.array(cvcuda_affine_tensor))

# torch.tensor(cvcuda_output_tensor.cuda()).data_ptr()
img_out = cvcuda.as_image(cvcuda_output_tensor.cuda(), format=cvcuda.Format.BGR8)
img_out = img_out.cpu()
print(img_out.shape)
CVImage(img_out).show()
