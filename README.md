## convenience API for your AI power

all prtrain models can be accessed by BaiDuPan:

[link](https://pan.baidu.com/s/18MegZnMQn1oQR1jJPpWJxQ) pwd: ibgg

### Face Embedding

- Arcface from [SimSwap](https://github.com/neuralchen/SimSwap) (mod to jit model)
- [CurricularFace](https://github.com/HuangYG123/CurricularFace) (mod to jit model)

### Face Restore 

- dfdnet Gpen ESRGAN etc. TODO

### Face Quality

- based on [mmclassification](https://github.com/open-mmlab/mmclassification), follow [install guide](https://github.com/open-mmlab/mmclassification/blob/master/docs/install.md)
- supply pretrained model trained by private dataset, mind the face should be aligned first

### Face Detect & Align

- ffhq align method
- mtcnn from [FaceShifter](https://github.com/taotaonice/FaceShifter)
- scrfd from [SimSwap](https://github.com/neuralchen/SimSwap)

### Dataset Preprocess

- count imgs mean & std
- generate img names from dir to txt

### Model Convert

- torch2jit torch2onnx etc.
- onnx2onnx-sim2tensorrt
- onnx2tensorrt