# Caution
Considering some sensitive codes, stop update and move to private, will reopen after some period of time.

## Convenience API for your AI power

All pretrain models and conda env can be accessed by my share link:

[BaiDuPan](https://pan.baidu.com/s/18MegZnMQn1oQR1jJPpWJxQ) pwd: ibgg

### Mocap Libs

Body & hand detector, hand mesh, visualize tools etc.

refer [mocap_lib](./mocap_lib)

### Face Libs

Face detect & align & parsing & embedding & 3D & restore etc.

refer [face_lib](./face_lib)

### Model Libs

Onnx & tensorrt model template, model convert.

refer [model_lib](./model_lib)

### Install Notes

Using pip/conda/gcc/... install the repo all u needs.

Mind the linux/python/cuda/cudnn/tensorrt/opencv/... versions and compatible.

```shell
# torch
pip install torch==1.11+cu113 torchvision==0.12+cu113 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
# tensorrt
pip install nvidia-tensorrt==8.* --index-url https://pypi.ngc.nvidia.com
```
