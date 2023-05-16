# -- coding: utf-8 --
# @Time : 2022/6/22
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import numpy as np
from cv2box import CVFile, CVVideoLoader, CVImage, MyFpsCounter
from apstone import ModelBase
from tqdm import tqdm

# from frankmocap https://github.com/facebookresearch/frankmocap
# input_name:['input'], shape:[[1, 3, 224, 224]]
# output_name:['output'], shape:[[1, 61]]
MODEL_ZOO = {
    # gpu 67fps trt16 85fps
    'fmc': {
        'model_path': 'pretrain_models/hand_lib/frankmocap/hand_regressor.onnx',
        'model_input_size': (224, 224), },
}

hand_mean = np.array([0, 0, 0, 0.11167872, -0.04289217, 0.41644184, 0.10881133, 0.06598568,
                      0.75622001, -0.09639297, 0.09091566, 0.18845929, -0.11809504,
                      -0.05094385, 0.5295845, -0.14369841, -0.0552417, 0.70485714,
                      -0.01918292, 0.09233685, 0.33791352, -0.45703298, 0.19628395,
                      0.62545753, -0.21465238, 0.06599829, 0.50689421, -0.36972436,
                      0.06034463, 0.07949023, -0.14186969, 0.08585263, 0.63552826,
                      -0.30334159, 0.05788098, 0.63138921, -0.17612089, 0.13209308,
                      0.37335458, 0.85096428, -0.27692274, 0.09154807, -0.49983944,
                      -0.02655647, -0.05288088, 0.53555915, -0.04596104, 0.27735802])


class HandRegressorFmc(ModelBase):
    def __init__(self, model='fmc', provider='gpu'):
        super().__init__(MODEL_ZOO[model], provider)
        self.input_std = self.input_mean = 127.5

    def forward(self, image_in_, hand_box, left_hand=False, show=False):
        """
        Args:
            image_in_:
            hand_box: [x1,y1,x2,y2]
            show:
            left_hand:
        Returns:
        """
        hand_img = CVImage(image_in_).crop_margin(hand_box, margin_ratio=0.3)
        hand_img, _, _, _ = CVImage(hand_img).resize_keep_ratio((224, 224))

        if show:
            CVImage(hand_img).show(wait_time=1, window_name='left_hand' if left_hand else 'right_hand')
        if left_hand:
            hand_img = np.ascontiguousarray(hand_img[:, ::-1, :], hand_img.dtype)

        blob = CVImage(hand_img).set_blob(self.input_std
                                          , self.input_mean, (224, 224)).blob_in(rgb=False)
        results_ = self.model.forward(blob, trans=True)[0]

        # pred_cam_params = results[:, :3]
        # (1, 48): (1, 3) for hand rotation, (1, 45) for finger pose.
        pred_pose_params = results_[:, 3: (3 + 48)] + hand_mean
        # pred_shape_params = results[:, (3 + 48):]
        if left_hand:
            pred_pose_params[:, 1::3] *= -1
            pred_pose_params[:, 2::3] *= -1
        # if show:
        #     print(pred_cam_params)
        #     print(pred_pose_params)
        #     print(pred_shape_params)
        return pred_pose_params


class HandRegressorVideo(HandRegressorFmc):
    def __init__(self, box_file, video_path):
        super().__init__()
        self.box_file = box_file
        self.video_path = video_path

    def crop_video_hand(self):
        hand_box_list = CVFile(self.box_file).data
        with CVVideoLoader(self.video_path) as cvvl:
            for index, _ in enumerate(tqdm(range(len(cvvl)))):
                _, frame = cvvl.get()
                for box in hand_box_list[index]:
                    hand_img = CVImage(frame).crop_margin(box, margin_ratio=0.3)
                    hand_img, _, _, _ = CVImage(hand_img).resize_keep_ratio((224, 224))

                    CVImage(hand_img).show(wait_time=0)
                    results = self.model.forward(hand_img.astype(np.float32), trans=True)[0]
                    # print(self.model.forward(hand_img.astype(np.float32), trans=True))

                    pred_cam_params = results[:, :3]
                    # (1, 48): (1, 3) for hand rotation, (1, 45) for finger pose.
                    pred_pose_params = results[:, 3: (3 + 48)]
                    pred_shape_params = results[:, (3 + 48):]
                    print(pred_cam_params)
                    print(pred_pose_params)
                    print(pred_shape_params)


if __name__ == '__main__':
    image_p = 'resource/for_pose/t_pose_1080p.jpeg'
    box_l = [1306, 259, 1432, 305]
    box_r = [494, 260, 609, 303]
    hrf = HandRegressorFmc(model='fmc', provider='gpu')
    results = hrf.forward(image_p, box_l,  left_hand=True, show=True)

    with MyFpsCounter('model forward 10 times fps:') as mfc:
        for i in range(10):
            results = hrf.forward(image_p, box_l, left_hand=True)

    # bbox_file = ''
    # video_p = ''
    # hrf = HandRegressorFmc(video_p, bbox_file)
    # hrf.crop_video_hand()
