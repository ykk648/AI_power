## Convenience API for your AI power

All pretrain models and supply materials can be accessed by my share link:

[BaiDuPan](https://pan.baidu.com/s/18MegZnMQn1oQR1jJPpWJxQ) pwd: ibgg

### Projects of mine

- [cv2box](https://github.com/ykk648/cv2box):   A gather of tools or functions frequently using in my work.
- [AI_power](https://github.com/ykk648/AI_power): Convenience API for your AI power.
- [apstone](https://github.com/ykk648/apstone): Base stone of AI_power, maintain all inference of AI_Power models.


### Model&Method Zoo

'*' means self-trained model



Supported face related models&methods:

more details refer [face_lib](./face_lib)

<details open>
<summary>(click to collapse)</summary>

- [x] [EyeOpenDetect](https://github.com/abhilb/Open-eye-closed-eye-classification) (eye close detect)
- [x] [IrisLandmark](https://github.com/Kazuhito00/iris-detection-using-py-mediapipe) 
- [x] [Deep3DFace](https://github.com/microsoft/Deep3DFaceReconstruction) (face 3d)
- [x] [*MobileNetV3](https://github.com/open-mmlab/mmclassification/tree/master/configs/mobilenet_v3) (face attitrbute) 
- [x] [MTCNN](https://github.com/taotaonice/FaceShifter/blob/master/face_modules/mtcnn.py) (face detect)
- [x] [FAN](https://github.com/1adrianb/2D-and-3D-face-alignment)
- [x] [S3FD](https://github.com/iperov/DeepFaceLive/blob/master/modelhub/onnx/S3FD/S3FD.py)
- [x] [SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd) 
- [x] [ArcFace](https://github.com/neuralchen/SimSwap/blob/01a8d6d0a6fd7e7b0052a5832328fba33f2b8414/models/fs_model.py#L63) (face embedding)
- [x] [CurricularFace](https://github.com/HuangYG123/CurricularFace)
- [x] [InsightFace](https://github.com/deepinsight/insightface/tree/master/model_zoo) 
- [x] [PFPLD](https://github.com/hanson-young/nniefacelib/tree/master/PFPLD/models/onnx) (face landmark)
- [x] [*XsegNet](./face_lib#face-parsing) (face parsing)
- [x] [face-parsing.PyTorch](./face_lib#face-parsing)
- [x] [GPEN](https://github.com/yangxy/GPEN) (face restore)
- [x] [DFDNet](https://github.com/csxmli2016/DFDNet) 
- [x] [GFPGAN](https://github.com/TencentARC/GFPGAN) 
- [x] [RestoreFormer](https://github.com/wzhouxiff/RestoreFormer) 
- [x] [CodeFormer](https://github.com/sczhou/CodeFormer) 
- [x] [*HifiFace](https://johann.wang/HifiFace/) (face swap)



Supported body related models&methods:

more details refer [body_lib](./body_lib)

<details open>
<summary>(click to collapse)</summary>

- [x] [yolox_tiny/yolox_s](https://github.com/open-mmlab/mmdetection) (mmdetection) (body bbox)
- [x] [hrnetv2w32](https://modelscope.cn/models/damo/cv_hrnetv2w32_body-2d-keypoints_image/summary)  (modelscope) (body keypoints)
- [x] [BlazePose](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/053_BlazePose)
- [x] [lightweight-human-pose-estimation](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)
- [x] [MoveNet](https://tfhub.dev/s?q=movenet)
- [x] [kapao](https://github.com/wmcnally/kapao)



Supported hand related models&methods:

more details refer [hand_lib](./hand_lib)

<details open>
<summary>(click to collapse)</summary>

- [x] [yolox_tiny](https://github.com/open-mmlab/mmdetection) (mmdetection) (hand 21 keypoints detect) 
- [x] [hand_detector.d2](https://github.com/ddshan/hand_detector.d2) (hand bbox)
- [x] [mediapipe hands](https://google.github.io/mediapipe/solutions/hands)
- [x] [*yolox](https://github.com/open-mmlab/mmdetection/blob/master/configs/yolox)
- [x] [minimal-hand](https://github.com/CalciferZh/minimal-hand) (hand mesh)
- [x] [frankmocap hand regressor](https://github.com/facebookresearch/frankmocap) (hand pose)



Supported mocap related models&methods:

more details refer [mocap_lib](./mocap_lib)

<details open>
<summary>(click to collapse)</summary>

- [x] [SPIN](https://github.com/nkolot/SPIN) (body regress) 
- [x] [r50/vipnas_mbv3_dark/hrnet_w48_384_dark etc.](https://github.com/open-mmlab/mmpose) (mmpose) (whole body keypoints)
- [x] [mediapipe holistic](https://google.github.io/mediapipe/solutions/holistic.html)
- [x] [Calibration](./mocap_lib#calibration)
- [x] [Smooth Filter](./mocap_lib#smooth-filter)
- [x] [Triangulate](./mocap_lib#triangulation)



Supported segmentation related models&methods:

more details refer [seg_lib](./seg_lib)

<details open>
<summary>(click to collapse)</summary>

- [x] [u2net](./seg_lib#u2net) (object segmentation) 
- [x] [ppmattingv2](./seg_lib#ppmattingv2)
- [x] [carvekit](./seg_lib#carvekit)
- [x] [cihp_pgn](./seg_lib#cihp_pgn)


Supported AIGC related models&methods:

more details refer [art_lib](./seg_lib)

<details open>
<summary>(click to collapse)</summary>

- [x] [DCTNet](https://www.modelscope.cn/models/damo/cv_unet_person-image-cartoon_compound-models/summary) (style transfer) 
- [x] [TPSMM](https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model) (talking head) 


Supported GPT related models&methods:

more details refer [gpt_lib](./seg_lib)

<details open>
<summary>(click to collapse)</summary>

- [x] [ChatGLM6B finetune](https://github.com/mymusise/ChatGLM-Tuning)
- [x] [OPENAI API]()
- [x] [langchain]()
- [x] [lora finetune]()
- [x] [text splitter](https://www.modelscope.cn/models/damo/nlp_bert_document-segmentation_chinese-base/summary)


Supported audio related models&methods:

more details refer [audio_lib](./audio_lib)

<details open>
<summary>(click to collapse)</summary>

- [x] [sovits](https://github.com/voicepaw/so-vits-svc-fork) (svc) 
- [x] [bark](https://github.com/suno-ai/bark) (tts) 


Supported OCR related models&methods:

more details refer [ocr_lib](./audio_lib)

<details open>
<summary>(click to collapse)</summary>

- [x] [paddleocr](https://github.com/PaddlePaddle/PaddleOCR) 

