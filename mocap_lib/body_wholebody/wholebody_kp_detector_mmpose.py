# -- coding: utf-8 --
# @Time : 2022/8/17
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

from cv2box import CVImage, MyFpsCounter
from apstone.wrappers.mmlab_wrapper import KpDetectorBase

MODEL_ZOO = {
    # API 62fps trt16 154fps
    'r50': {
        'model_path': 'pretrain_models/mocap_lib/coco_whole_body/res50_coco_wholebody_256x192-9e37ed88_20201004_remove_initializer.onnx',
        'model_input_size': (192, 256)
    },  # w h
    # 195fps
    'r50_trt': {
        'model_path': 'pretrain_models/mocap_lib/coco_whole_body/res50_coco_wholebody_256x192-9e37ed88_20201004.engine',
        'model_input_size': (192, 256)
    },
    # API 34fps
    'vipnas_mbv3_dark': {
        'model_path': 'pretrain_models/mocap_lib/coco_whole_body/vipnas_mbv3_coco_wholebody_256x192_dark-e2158108_20211205_remove_initializer.onnx',
        'model_input_size': (192, 256)
    },
    # API 38fps
    'vipnas_r50_dark': {
        'model_path': 'pretrain_models/mocap_lib/coco_whole_body/vipnas_res50_wholebody_256x192_dark-67c0ce35_20211112_remove_initializer.onnx',
        'model_input_size': (192, 256)
    },
    # trt16 50fps
    'hrnet_w48_384_dark': {
        'model_path': 'pretrain_models/mocap_lib/coco_whole_body/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918_remove_initializer.onnx',
        'model_input_size': (288, 384),
        'kernel': 17},
    'hrnet_w48_384_dark_dynamic': {
        'model_path': 'pretrain_models/mocap_lib/coco_whole_body/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918_dynamic.onnx',
        'model_input_size': (288, 384),
        'input_dynamic_shape': (4, 3, 288, 384),
        'kernel': 17},
    # 48fps
    'hrnet_w48_384_dark_trt': {
        'model_path': 'pretrain_models/mocap_lib/coco_whole_body/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.engine',
        'model_input_size': (288, 384),
        'kernel': 17},
}

# 用于镜像翻转的pair对
flip_pairs = [
    [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 20], [18, 21], [19, 22]
]
for i in range(91, 112):
    flip_pairs.append([i, i + 21])


class BodyWholebodyDetector(KpDetectorBase):
    def __init__(self, model_type='r50', provider='gpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.dark_flag = model_type.find('dark') > 0

    def forward(self, image_in_, bbox_, show=False, mirror_test=False):
        if len(bbox_) == 0:
            return [[0, 0, 0]] * 133

        outputs = self.model.forward(self.preprocess(image_in_, bbox_))

        if mirror_test:
            outputs_mirror = self.model.forward(self.preprocess(image_in_, bbox_, mirror=mirror_test))
            kp_results = self.postprocess_mirror(outputs, outputs_mirror, flip_pairs)
        else:
            kp_results = self.postprocess(outputs)

        if show:
            self.show(image_in_, kp_results)

        return kp_results


if __name__ == '__main__':
    image_path = 'resource/for_pose/t_pose_1080p.jpeg'
    image_in = CVImage(image_path).bgr
    bbox = [493, 75, 1427, 1044]

    bwd = BodyWholebodyDetector(model_type='hrnet_w48_384_dark_dynamic', provider='trt')
    kps = bwd.forward(image_in, bbox, show=True, mirror_test=False)
    # print(kps)

    with MyFpsCounter('model forward 10 times fps: ') as mfc:
        for i in range(10):
            kps = bwd.forward(image_in, bbox)

    # # for video
    # from cv2box import CVVideoLoader
    # from tqdm import tqdm
    #
    # with CVVideoLoader('') as cvvl:
    #     for _ in tqdm(range(len(cvvl))):
    #         _, frame = cvvl.get()
    #         kps = bwd.forward(image_in, bbox, show=True, mirror_test=False)
