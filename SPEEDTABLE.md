### Notes

- '-' means use last result from top
- empty means no test results
- onnx convert based on [mmdeploy](https://github.com/open-mmlab/mmdeploy)
- model infer fps does not contain pre/post-process time cost
- trt & trt16 based on onnxruntime tensorrt EP
- input array located on cpu (io-binding done by onnxruntime itself)
- more info check [onnx test codes]()



### Environment

| Name | Attr                                      |
| ---- | ----------------------------------------- |
| Sys  | Ubuntu 20.04                              |
| GPU  | NVIDIA GeForce RTX 3080  10G              |
| CPU  | Intel® Core™ i9-10850K CPU @ 3.60GHz × 20 |
| MEM  | 32G                                       |
| Libs | onnxruntime-gpu=1.13                      |



### Pose Detect



| MMPose                                                       | input shape      | size   | cpu infer fps | gpu infer fps | trt infer fps | trt16 infer fps |
| ------------------------------------------------------------ | ---------------- | ------ | ------------- | ------------- | ------------- | --------------- |
| [pvtv2-b2_static_coco](https://github.com/open-mmlab/mmpose/tree/master/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/pvtv2-b2_coco_256x192.py) | [1, 3, 256, 192] | 116.3m | 4.9           | 73            | 184           | 257             |
|                                                              | [4, 3, 256, 192] | -      | 2.5           | 47            | 106           | 178             |
|                                                              |                  |        |               |               |               |                 |
| [hrnet_w48_dark+_dynamic_cocowholebody](https://github.com/open-mmlab/mmpose/tree/master/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py) | [4, 3, 384, 288] | 254m   | 2.9           | 31            | 39            | 83              |
|                                                              |                  |        |               |               |               |                 |
| **MMPose Post-process**                                      |                  |        |               |               |               |                 |
| gaussian_blur_k17                                            | [4, 133, 96, 72] | 0.19m  | 7.9           | 119           | 147           | 142             |
|                                                              |                  |        |               |               |               |                 |

