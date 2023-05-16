from apstone import ONNXModel
from cv2box import CVImage, MyFpsCounter
from apstone.wrappers.mmlab_wrapper.utils import _taylor, _gaussian_blur, _get_max_preds
import numpy as np
import cv2

HRNET_W18 = 'pretrain_models/hand_lib/hand_detector_21kp/hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908_remove_initializer.onnx'
MOBILENETV2 = 'pretrain_models/hand_lib/hand_detector_21kp/mobilenetv2_coco_wholebody_hand_256x256-06b8c877_20210909_dynamic.onnx'
MOBILENETV2_TRT = 'pretrain_models/hand_lib/hand_detector_21kp/mobilenetv2_coco_wholebody_hand_256x256-06b8c877_20210909.engine'

model_input_size = {
    # w h
    'hrnet_w18': (256, 256),
    'mbfv2': (256, 256),
    'mbfv2_trt': (256, 256),
}


# flip_pairs = [
#     [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 20], [18, 21], [19, 22]
# ]
# for i in range(91, 112):
#     flip_pairs.append([i, i + 21])


class HandWholebodyDetector:
    def __init__(self, model_type='hrnet_w18', provider='gpu'):
        self.model_type = model_type
        if self.model_type == 'hrnet_w18':
            self.model = ONNXModel(HRNET_W18, provider=provider)
        elif self.model_type == 'mbfv2':
            # dynamic model needs to aim the input shape
            self.model = ONNXModel(MOBILENETV2, provider=provider, input_dynamic_shape=(1, 3, 256, 256))
        elif self.model_type == 'mbfv2_trt':
            from apstone.wrappers.trt_wrapper import TRTWrapper
            self.model = TRTWrapper(MOBILENETV2_TRT)

    def trans_pred(self, preds, ratio, left, top, maxvals, H):
        kp_results = []
        for index, kp in enumerate(preds[0]):
            x, y = preds[0][index]
            # 4 / ratio
            new_ratio = (model_input_size[self.model_type][1] // H) / ratio
            new_y = y * new_ratio + top
            new_x = x * new_ratio + left
            kp_results.append([new_x, new_y, float(maxvals[0][index][0])])
        # print(kp_results)
        return kp_results

    def post_process_default(self, heatmaps, ratio, left, top):
        # postprocess
        N, K, H, W = heatmaps.shape
        preds, maxvals = _get_max_preds(heatmaps)
        for n in range(N):
            for k in range(K):
                heatmap = heatmaps[n][k]
                px = int(preds[n][k][0])
                py = int(preds[n][k][1])
                if 1 < px < W - 1 and 1 < py < H - 1:
                    diff = np.array([
                        heatmap[py][px + 1] - heatmap[py][px - 1],
                        heatmap[py + 1][px] - heatmap[py - 1][px]
                    ])
                    preds[n][k] += np.sign(diff) * .25
        return self.trans_pred(preds, ratio, left, top, maxvals, H)

    def post_process_unbiased(self, heatmaps, ratio, left, top, kernel=11):
        # apply Gaussian distribution modulation.
        N, K, H, W = heatmaps.shape
        preds, maxvals = _get_max_preds(heatmaps)
        heatmaps = np.log(
            np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
        for n in range(N):
            for k in range(K):
                preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])

        return self.trans_pred(preds, ratio, left, top, maxvals, H)

    def forward(self, image_in_, bbox_, show=False, mirror_test=False):
        if len(bbox_) == 0:
            return [[0, 0, 0]] * 21
        kp_results = []
        img_resize, ratio, left, top = CVImage(image_in_).crop_keep_ratio(bbox_, model_input_size[self.model_type])
        if self.model_type.find('trt') > 0:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            blob = CVImage(img_resize).set_transform(transform).tensor().cuda()

            heatmaps = self.model.forward(dict(input=blob))['output'].cpu().numpy()
        else:
            blob = CVImage(img_resize) \
                .innormal(mean=[123.675, 116.28, 103.53], std=[58.395, 57.120000000000005, 57.375])
            heatmaps = self.model.forward(blob, trans=True)[0]
        if mirror_test:
            blob_mirror = CVImage(cv2.flip(img_resize, 1)) \
                .innormal(mean=[123.675, 116.28, 103.53], std=[58.395, 57.120000000000005, 57.375])
            heatmaps_mirror = self.model.forward(blob_mirror, trans=True)[0]
            # heatmaps_mirror_back = heatmaps_mirror.copy()
            # # # Swap left-right parts
            # for left_id, right_id in flip_pairs:
            #     heatmaps_mirror_back[:, left_id, ...] = heatmaps_mirror[:, right_id, ...]
            #     heatmaps_mirror_back[:, right_id, ...] = heatmaps_mirror[:, left_id, ...]
            # heatmaps_mirror_back = heatmaps_mirror_back[..., ::-1]
            heatmaps = (heatmaps + heatmaps_mirror[..., ::-1]) * 0.5
        if self.model_type in ['mbfv2', 'mbfv2_trt']:
            kp_results = self.post_process_default(heatmaps, ratio, left, top)
        if self.model_type in ['hrnet_w18']:
            kp_results = self.post_process_unbiased(heatmaps, ratio, left, top)

        if show:
            image_base = CVImage(image_in_).bgr
            for kp in kp_results:
                cv2.circle(image_base, (int(kp[0]), int(kp[1])), 4, color=(0, 255, 0))
            CVImage(image_base).show()

        return kp_results

    def batch_forward(self, image_in_, bbox_list_):
        if len(bbox_list_) == 0:
            return [[0, 0, 0]] * 21
        ratio_list = []
        left_list = []
        top_list = []
        blob_list = []
        kp_results = []
        for index, box_in_ in enumerate(bbox_list_):
            img_resize, ratio, left, top = CVImage(image_in_).crop_keep_ratio(box_in_[:4],
                                                                              model_input_size[self.model_type])
            ratio_list.append(ratio)
            left_list.append(left)
            top_list.append(top)

            blob = CVImage(img_resize) \
                .innormal(mean=[123.675, 116.28, 103.53], std=[58.395, 57.120000000000005, 57.375])
            blob_list.append(blob)
        heatmaps = self.model.batch_forward(np.array(blob_list), trans=True)[0]

        for index, heatmap in enumerate(heatmaps):
            kp_result = self.post_process_default(heatmap[np.newaxis, :], ratio_list[index], left_list[index],
                                                  top_list[index])
            kp_result = np.array(kp_result)
            kp_result[:, 2] = kp_result[:, 2] * bbox_list_[index][4]
            kp_results.append(kp_result)
        return kp_results


if __name__ == '__main__':
    image_path = 'resource/t_pose.jpeg'
    image_in = CVImage(image_path).bgr
    bbox = [100, 365, 242, 422]

    """
    hrnet_w18: trt8 fail trt16 136fps trt 125fps gpu 51fps
    mbfv2 dynamic: trt8 fail trt16 445fps trt 329fps gpu 123fps
    mbfv2_trt: trt16 325fps
    """
    bwd = HandWholebodyDetector(model_type='mbfv2', provider='trt16')  # hrnet_w18 mbfv2 mbfv2_trt
    kps = bwd.forward(image_in, bbox, show=True, mirror_test=False)
    print(kps)

    with MyFpsCounter('model forward 10 times fps: ') as mfc:
        for i in range(10):
            kps = bwd.forward(image_in, bbox)
