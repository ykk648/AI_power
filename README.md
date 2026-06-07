# AI_power

> **This repo is no longer maintained.**
>
> Face, audio, and LLM related APIs have been moved to dedicated repos:
> - [face_power](https://github.com/ykk648/face_power) - Face detection, parsing, restore, swap, etc.
> - [audio_power](https://github.com/ykk648/audio_power) - ASR, TTS, voice conversion, audio translation
> - [llm_power](https://github.com/ykk648/llm_power) - LLM fine-tuning, RAG, LangChain, ChatGLM

Convenience API collection for computer vision, motion capture, segmentation, AIGC, and Stable Diffusion inference.

Pretrained models: [BaiDuPan](https://pan.baidu.com/s/18MegZnMQn1oQR1jJPpWJxQ) pwd: `ibgg`

## Related Projects

| Project | Description |
|---------|-------------|
| [cv2box](https://github.com/ykk648/cv2box) | CV utility functions used across all projects |
| [apstone](https://github.com/ykk648/apstone) | Base inference engine for all model wrappers |

## Module Overview

### body_lib - Body Keypoint Detection

| Model | Source | Function |
|-------|--------|----------|
| YOLOX-tiny/s | [MMDetection](https://github.com/open-mmlab/mmdetection) | Body bbox |
| HRNetV2-w32 | [ModelScope](https://modelscope.cn/models/damo/cv_hrnetv2w32_body-2d-keypoints_image/summary) | Body keypoints |
| BlazePose | [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/053_BlazePose) | Body keypoints |
| Lightweight-Pose | [lightweight-human-pose-estimation](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) | Body keypoints |
| MoveNet | [TFHub](https://tfhub.dev/s?q=movenet) | Body keypoints |
| KAPAO | [kapao](https://github.com/wmcnally/kapao) | Body keypoints |

### hand_lib - Hand Detection & Mesh

| Model | Source | Function |
|-------|--------|----------|
| YOLOX-tiny | [MMDetection](https://github.com/open-mmlab/mmdetection) | Hand 21 keypoints |
| hand_detector.d2 | [hand_detector.d2](https://github.com/ddshan/hand_detector.d2) | Hand bbox |
| MediaPipe Hands | [MediaPipe](https://google.github.io/mediapipe/solutions/hands) | Hand landmarks |
| YOLOX* | [MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/configs/yolox) | Hand detection |
| Minimal-Hand | [minimal-hand](https://github.com/CalciferZh/minimal-hand) | Hand mesh |
| FrankMoCap | [frankmocap](https://github.com/facebookresearch/frankmocap) | Hand pose regressor |

### mocap_lib - Motion Capture Pipeline

| Module | Description |
|--------|-------------|
| [SPIN](https://github.com/nkolot/SPIN) | Body shape regress |
| [MMPose](https://github.com/open-mmlab/mmpose) | Whole body keypoints (r50, hrnet_w48_384_dark, etc.) |
| [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic.html) | Whole body holistic |
| [Calibration](./mocap_lib#calibration) | Camera calibration |
| [Smooth Filter](./mocap_lib#smooth-filter) | Temporal smoothing |
| [Triangulate](./mocap_lib#triangulation) | Multi-view triangulation |

### seg_lib - Segmentation & Matting

| Module | Description |
|--------|-------------|
| [CarveKit](./seg_lib#carvekit) | Cloth segmentation |
| [CIHP-PGN](./seg_lib#cihp_pgn) | Human parsing |
| [U2Net](./seg_lib#u2net) | Object segmentation / saliency |
| [PPMattingV2](./seg_lib#ppmattingv2) | Portrait matting |
| [Green Screen Matting](./seg_lib/green_screen_matting) | Chroma-key video matting (BackgroundMattingV2) |
| [SegFormer B2](./seg_lib/segformer_b2_clothes) | Cloth segmentation |
| [MODNet](./seg_lib/modnet) | Portrait matting |
| [RAFT](./seg_lib#raft) | Optical flow |

### art_lib - AIGC & Video Effects

| Model | Function |
|-------|----------|
| [DCTNet](./art_lib/style_transfer/dct_net) | Style transfer |
| [LaMa](./art_lib/inpainting/lama.py) | Image inpainting |
| [TPSMM](./art_lib/talking_head/tpsmm) | Talking head synthesis |
| [SadTalker](./art_lib/talking_head/sadtalker) | Talking head synthesis |
| [Wav2Lip](./art_lib/talking_head/wav2lip) | Lip sync |
| [DINet](./art_lib/talking_head/dinet_boost) | Talking head synthesis |

### sd_lib - Stable Diffusion Tools

| Module | Description |
|--------|-------------|
| [ControlNet](./sd_lib/controlnet) | ControlNet inference |
| [IP-Adapter](./sd_lib/ip_adapter) | IP-Adapter for image-prompted generation |
| [Prompt2Prompt](./sd_lib/prompt2prompt) | Prompt-based image editing |
| [CLIP Encoder](./sd_lib/clip_encoder.py) | Text encoding |
| [DDIM Inversion](./sd_lib/inversion_api.py) | Image inversion |
| [Tagger](./sd_lib/tagger) | Image tagging/WD14 |

### ocr_lib - OCR

| Model | Description |
|-------|-------------|
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | General-purpose OCR |

### data_lib - Dataset Tools

COCO format conversion, dataset preprocessing, visualization utilities.

### math_lib / utils - Utilities

Affine transforms, Gaussian filters, K-means, timing tools, path helpers.
