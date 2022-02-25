## 3D Estimation

### Hand Detector

- hand detector d2, borrow from [hand_detector.d2](https://github.com/ddshan/hand_detector.d2)
- hand detector mediapipe , using [mediapipe](https://github.com/google/mediapipe) to replace hand_detector.d2
- hand detector yolox , yolox model trained by [100DOH](https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/) and convert2onnx, box AP 94, fps 130.

### Hand pose estimator

- minimal hands, borrow from [minimal-hand](https://github.com/CalciferZh/minimal-hand), model converted to onnx
