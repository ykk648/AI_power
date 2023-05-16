# -- coding: utf-8 --
# @Time : 2021/11/18
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
from face_lib.face_detect_and_align import FaceAlignment, LandmarksType
from skimage import io
import matplotlib.pyplot as plt
import collections

face_detector_kwargs = {
    "path_to_detector": './pretrain_models/face_detect/face_alignment_1adrianb/s3fd-619a316812.pth',
    "filter_threshold": 0.5,
}

# input RGB image or image path
fa = FaceAlignment(LandmarksType._2D, device='cuda', flip_input=False, face_detector='sfd',face_detector_kwargs=face_detector_kwargs)


input = io.imread('./test_img/test2.jpg')
preds = fa.get_landmarks(rgb_image_or_path=input)[-1]
print(preds)

# 2D-Plot
plot_style = dict(marker='o',
                  markersize=4,
                  linestyle='-',
                  lw=2)

pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
              'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
              'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
              'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
              'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
              'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
              'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
              'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
              'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
              }

fig = plt.figure(figsize=plt.figaspect(.5))
ax = fig.add_subplot(1, 2, 1)
ax.imshow(input)

for pred_type in pred_types.values():
    ax.plot(preds[pred_type.slice, 0],
            preds[pred_type.slice, 1],
            color=pred_type.color, **plot_style)

ax.axis('off')
plt.show()
