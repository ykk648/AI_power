### Face Attribution

- based on [mmclassification](https://github.com/open-mmlab/mmclassification), already convert to onnx.
- supply pretrained model trained by private dataset, mind the face should be aligned first.


### Face Detect & Align

- ffhq align method
- mtcnn from [mtcnn-pytorch](https://github.com/TropComplique/mtcnn-pytorch)
- scrfd from [insightface](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
- [face alignment](https://github.com/1adrianb/face-alignment) from 1adrianb
- conform multi similarity align methods


### Face Embedding

- Arcface from [SimSwap](https://github.com/neuralchen/SimSwap) (mod to jit model)
- [CurricularFace](https://github.com/HuangYG123/CurricularFace) (mod to jit model)

### Face Parsing

- [face-parsing.PyTorch](./face_parsing/face_parsing_api.py), onnx converted
- [DFL Xseg](./face_parsing/dfl_xseg_net.py), based on opensource, self trained (private data)

### Face Restore 

- [Gpen](https://github.com/yangxy/GPEN)
- [DFDNet](https://github.com/csxmli2016/DFDNet) (add batch parallel support)
- [GFPGAN](https://github.com/TencentARC/GFPGAN)
- ESRGAN etc. TODO

### Face Swap

- [Hififace](https://johann.wang/HifiFace/)
- [InSwapper](https://github.com/deepinsight/insightface)