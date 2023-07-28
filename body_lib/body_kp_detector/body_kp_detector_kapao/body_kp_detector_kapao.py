# -- coding: utf-8 --
# @Time : 2022/7/29
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
"""
model from https://github.com/wmcnally/kapao
"""
import torch
from cv2box import CVImage, MyFpsCounter, mfc
import numpy as np
from torchvision import transforms

from apstone.wrappers.mmlab_wrapper import KpDetectorBase
from body_lib.body_kp_detector.body_kp_detector_kapao.utils import non_max_suppression_kp, post_process_batch, letterbox

MODEL_ZOO = {
    # gpu 30fps trt 39fps trt16 43fps
    # input_name:['actual_input_1'], shape:[[1, 3, 768, 1280]]
    # output_name:['output1'], shape:[[1, 61200, 57]]
    'kapao_s_coco_1080': {
        'model_path': 'pretrain_models/body_lib/body_kp_detector/kapao/kapao_s_coco_static_1280x768.onnx',
        'model_input_size': (1280, 768)
    },
}


class KaPao(KpDetectorBase):
    def __init__(self, model, provider):
        super().__init__(MODEL_ZOO[model], provider)
        self.origin_shape = None

    def preprocess(self, image_in_, bbox_, mirror=False):
        image_in_ = CVImage(image_in_).bgr
        # Padded resize
        image_in_ = letterbox(image_in_, self.model_input_size[::-1], stride=64)[0]

        if self.model_type == 'trt':
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            image_in_ = dict(input=CVImage(image_in_.astype(np.float32)).tensor(transform).cuda())
        else:
            # HWC -> CHW BGR -> RGB
            image_in_ = image_in_.astype(np.float32).transpose(2, 0, 1)[::-1][np.newaxis, :]/255

        return image_in_

    @mfc('postprocess')
    def postprocess(self, model_results):
        kp_flip = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
        data = {'num_coords': 34, 'use_kp_dets': True, 'conf_thres_kp_person': 0.3,
                'overwrite_tol': 50, 'count_fused': False}
        # lazy to rewrite torch nms to numpy
        model_results = torch.Tensor(model_results[0])
        person_dets = non_max_suppression_kp(model_results, conf_thres=0.7, iou_thres=0.45,
                                             classes=[0],
                                             num_coords=34)
        kp_dets = non_max_suppression_kp(model_results, conf_thres=0.5, iou_thres=0.45,
                                         classes=list(range(1, 1 + len(kp_flip))),
                                         num_coords=34)
        _, poses, _, _, _ = post_process_batch(data, self.model_input_size, self.origin_shape, person_dets, kp_dets)

        return poses[0]

    def forward(self, image_in_, show=False, max_bbox_num=1):
        self.origin_shape = image_in_.shape
        model_results = self.model.forward(self.preprocess(image_in_, None))
        results_after = self.postprocess(model_results)
        if show:
            self.show(image_in_, results_after)
        return results_after


if __name__ == '__main__':

    image_path = 'resources/for_pose/t_pose_1080p.jpeg'
    image_in = CVImage(image_path).bgr
    kp = KaPao(model='kapao_s_coco_1080', provider='gpu')

    kps = kp.forward(image_in, show=True, max_bbox_num=3)

