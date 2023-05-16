# -- coding: utf-8 --
# @Time : 2022/8/17
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

# os.environ['CV_MULTI_MODE'] = 'torch-process'  # multi-thread multi-process torch-process
import numpy as np
import queue

from cv2box import CVImage
from cv2box.cv_gears import Linker, Consumer, Queue, CVMultiVideoThread

from apstone import ModelBase
from apstone.wrappers.mmlab_wrapper.utils import _taylor_numba, _get_max_preds
from apstone.wrappers.mmlab_wrapper import KpDetectorBase
from mocap_lib.triangulate import EasyMocapTriangulate
from mocap_lib.bbox_tracking import BboxTracking
from mocap_lib.skeleton_transfer import cocowb_2_openpose
from mocap_lib.smooth_filter import SmoothFilter
from body_lib import BodyBboxDetector

MODEL_ZOO = {
    # trt16 50fps
    'hrnet_w48_384_dark': {
        'model_path': 'pretrain_models/mocap_lib/coco_whole_body/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918_remove_initializer.onnx',
        'model_input_size': (288, 384),
        'kernel': 17, },
    'hrnet_w48_384_dark_dynamic': {
        'model_path': 'pretrain_models/mocap_lib/coco_whole_body/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918_dynamic.onnx',
        'model_input_size': (288, 384),
        'batch_size': 4,
        'kernel': 17},
    'hrnet_w48_384_dark_dynamic_picklable': {
        'model_path': 'pretrain_models/mocap_lib/coco_whole_body/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918_dynamic.onnx',
        'model_input_size': (288, 384),
        'batch_size': 4,
        'picklable': True,  # for multiprocess infer
        'kernel': 17},
    # 48fps
    'hrnet_w48_384_dark_trt': {
        'model_path': 'pretrain_models/mocap_lib/coco_whole_body/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.engine',
        'model_input_size': (288, 384),
        'kernel': 17},
    'gaussian_133_k17': {
        'model_path': 'pretrain_models/math_lib/batch_gaussian_blur_4*133*96*72_k17.onnx',
    },
    'gaussian_133_k17_pickable': {
        'model_path': 'pretrain_models/math_lib/batch_gaussian_blur_4*133*96*72_k17.onnx',
        'picklable': True,  # for multiprocess infer
    },
}


class PreprocessThread(Linker):
    def __init__(self, queue_list: list, frame_shape_, fps_counter=True):
        super().__init__(queue_list, fps_counter)
        self.bbd = BodyBboxDetector(model='yolox_tiny_trt16', threshold=0.5, provider='')
        self.bt = BboxTracking(frame_shape_, batch=4)
        self.first = True
        self.count = 0
        self.bbox_ = None

    def forward_func(self, something):
        image_in_list = something
        if self.first:
            self.bbox_ = self.bt.forward(None)
            self.first = False
        else:
            if self.count % 5 == 0:
                self.bbox_ = self.bt.forward(self.queue_list[2].get(), margin=0., area_limit=1000)
            elif self.count % 90 == 0:
                self.bbox_ = []
                for image_in in image_in_list:
                    self.bbox_.append(self.bbd.forward(image_in, max_bbox_num=1)[0])

        img_resize_ = []
        ratio_ = []
        left_ = []
        top_ = []
        for i in range(4):
            img_resize, ratio, left, top = CVImage(image_in_list[i]).crop_keep_ratio(self.bbox_[i], (288, 384))
            # CVImage(img_resize).show(1, '{}'.format(i))
            blob = CVImage(img_resize).innormal(mean=[123.675, 116.28, 103.53],
                                                std=[58.395, 57.1200, 57.375]).transpose(2, 0, 1)
            img_resize_.append(blob)
            ratio_.append(ratio)
            left_.append(left)
            top_.append(top)
        self.count += 1
        return np.array(img_resize_), ratio_, left_, top_


class ModelForwardThread(KpDetectorBase, Linker):
    def __init__(self, queue_list, model_type='hrnet_w48_384_dark_dynamic', provider='trt16'):
        KpDetectorBase.__init__(self, MODEL_ZOO[model_type], provider)
        Linker.__init__(self, queue_list, fps_counter=True)
        self.dark_flag = model_type.find('dark') > 0

    def forward_func(self, something_in):
        inputs, ratio_, left_, top_ = something_in
        outputs = self.model.forward(inputs)
        return [outputs[0], ratio_, left_, top_]


class PostProcess0Thread(Linker):
    def __init__(self, queue_list: list, fps_counter=True):
        super().__init__(queue_list, fps_counter)

    def forward_func(self, something):
        heatmaps, ratio_, left_, top_ = something
        N, K, H, W = heatmaps.shape
        preds, maxvals = _get_max_preds(heatmaps)
        return [heatmaps, N, K, H, preds, maxvals, ratio_, left_, top_]


class PostProcess1Thread(Linker, ModelBase):
    def __init__(self, queue_list: list, fps_counter=True):
        Linker.__init__(self, queue_list, fps_counter)
        ModelBase.__init__(self, MODEL_ZOO['gaussian_133_k17'], 'trt16')
        # self.Gaussian_blur = GaussianLayer(133, 17).cuda().eval()

    def forward_func(self, something):
        heatmaps, N, K, H, preds, maxvals, ratio_, left_, top_ = something
        # heatmaps = _gaussian_blur(heatmaps, kernel=17)
        # with torch.no_grad():
        #     heatmaps = self.Gaussian_blur.forward(torch.from_numpy(heatmaps).cuda()).cpu().numpy()
        heatmaps = self.model.forward(heatmaps)[0]
        return [heatmaps, N, K, H, preds, maxvals, ratio_, left_, top_]


class PostProcess2Thread(Linker):
    def __init__(self, queue_list: list, frame_shape_, fps_counter=True):
        super().__init__(queue_list, fps_counter)
        self.frame_shape_ = frame_shape_

    def forward_func(self, something):

        heatmaps, N, K, H, preds, maxvals, ratio_, left_, top_ = something
        heatmaps = np.log(np.maximum(heatmaps, 1e-10))
        kp_results = []
        for n in range(N):
            for k in range(K):
                coord, offset_flag, offset = _taylor_numba(heatmaps[n][k], preds[n][k])
                if offset_flag:
                    offset = np.squeeze(np.array(offset.T), axis=0)
                    coord += offset
                preds[n][k] = coord
            kp_results_this_cam = []
            for index, kp in enumerate(preds[n]):
                x, y = preds[n][index]
                # 4 / ratio
                new_ratio = (384 // H) / ratio_[n]
                new_y = y * new_ratio + top_[n]
                new_x = x * new_ratio + left_[n]
                kp_results_this_cam.append([new_x, new_y, float(maxvals[0][index][0])])

            left_hand_kps, right_hand_kps, openpose_25_kps = cocowb_2_openpose(kp_results_this_cam)
            whole_kps = np.concatenate((openpose_25_kps, left_hand_kps, right_hand_kps), 0)

            kp_results.append(whole_kps)
        try:
            self.queue_list[2].put_nowait(kp_results)
        except queue.Full:
            _ = self.queue_list[2].get_nowait()
            self.queue_list[2].put_nowait(kp_results)

        return kp_results


class TriangulateThread(Consumer):
    def __init__(self, pkl_path_, queue_list: list, fps_counter):
        super(TriangulateThread, self).__init__(queue_list, fps_counter=fps_counter)
        self.emt = EasyMocapTriangulate(pkl_path_, pkl_mode_='anipose', max_repro_error=50)
        # self.at = AniposeTriangulate(pkl_path_, pkl_mode_='anipose')
        self.sf = SmoothFilter(filter_type='one_euro')
        self.four_list = []
        self.count = 0

    def forward_func(self, something_in):
        self.four_list = np.array(something_in)

        kpt_thre = 0.5
        ignore_idxs = np.where(self.four_list[:, :, 2] < kpt_thre)
        self.four_list[ignore_idxs[0], ignore_idxs[1], :] = 0.

        kps_3d, kps_repro = self.emt.triangulate(self.four_list)
        # kps_3d = self.at.triangulate(np.array(self.four_list))

        kps_3d = kps_3d[:, :3]
        zero_index = np.where(kps_3d == 0.)
        kps_3d = self.sf.forward(kps_3d)
        kps_3d[zero_index] = 0.

        # # 坐标转换
        # Z = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0]).reshape((3, 3))
        # kps_3d = kps_3d[:, :, :3] @ Z.T

        # CVFile('./results/2d_kp_four_list/{}.pkl'.format(self.count)).pickle_write(self.four_list)

        # CVFile('./results/3d_kp/{}.pkl'.format(self.count)).pickle_write(kps_3d)

        self.four_list = []
        self.count += 1
        return kps_3d


if __name__ == '__main__':
    # for onnx multiprocess infer
    # import multiprocessing
    # # method = multiprocessing.get_start_method()
    # # print(method)
    # multiprocessing.set_start_method('spawn')  # fork spawn

    video_parent_p = ''
    source_list = [
        video_parent_p + '/268.mp4',
        video_parent_p + '/617.mp4',
        video_parent_p + '/728.mp4',
        video_parent_p + '/886.mp4',
    ]

    pkl_path = video_parent_p + '/front_4_0809_window_1080_bundle_adjust_cgroup.pkl'
    frame_shape = (1920, 1080)

    q1 = Queue(2)
    q1_2 = Queue(2)
    q_circle = Queue(2)
    q2 = Queue(2)
    q3 = Queue(2)
    q4 = Queue(2)
    q5 = Queue(2)
    q6 = Queue(2)

    cmvt = CVMultiVideoThread(source_list, [q1], multi_stream_offset=False, silent=False, block=True, fps_counter=False)
    pt = PreprocessThread([q1, q1_2, q_circle], frame_shape)
    mf = ModelForwardThread([q1_2, q2])
    pp0 = PostProcess0Thread([q2, q3])
    pp1 = PostProcess1Thread([q3, q4])
    pp2 = PostProcess2Thread([q4, q5, q_circle], frame_shape)
    t1 = TriangulateThread(pkl_path, [q5, q6], fps_counter=True)

    cmvt.start()
    pt.start()
    mf.start()
    pp0.start()
    pp1.start()
    pp2.start()
    t1.start()

    cmvt.join()
    pt.join()
    mf.join()
    pp0.join()
    pp1.join()
    pp2.join()
    t1.join()
