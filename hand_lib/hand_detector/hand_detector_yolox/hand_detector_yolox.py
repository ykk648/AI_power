# -- coding: utf-8 --
# @Time : 2022/1/7
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
from cv2box import CVImage, MyFpsCounter, CVVideoLoader, CVVideoMaker, CVFile
from apstone.mmlab_wrapper import BboxDetectorBase
from tqdm import tqdm

# input 1*3*640*640 output 1*N*5 1*N
MODEL_ZOO = {
    # gpu 68fps
    'yolox_s': {
        'model_path': 'private_models/hand_lib/hand_detector_yolox/yolox_s_100DOH_epoch90_mmdeploy_dynamic.onnx',
        'input_dynamic_shape': (1, 3, 640, 640),
        'model_input_size': (640, 640),
        'label': 1,
    },
    'yolox_s_local': {
        'model_path': 'private_models/hand_lib/hand_detector_yolox/yolox_s_2dataset_epoch127_0922_dynamic.onnx',
        'input_dynamic_shape': (1, 3, 640, 640),
        'model_input_size': (640, 640),
        'label': 0,
    },
    # 260 fps
    'yolox_s_trt16': {
        'model_path': 'private_models/hand_lib/hand_detector_yolox/yolox_s_100DOH_epoch90_mmdeploy_static.engine',
        'input_dynamic_shape': (1, 3, 640, 640),
        'model_input_size': (640, 640),
        'label': 1,
    },
}


class HandDetectorYolox(BboxDetectorBase):
    def __init__(self, model='yolox_s', threshold=0.5, provider='gpu'):
        super().__init__(MODEL_ZOO[model], provider)
        self.threshold = threshold
        self.label = MODEL_ZOO[model]['label']

    def forward(self, image_in_, show=False):
        model_results = self.model.forward(self.preprocess(image_in_))
        results_after = self.postprocess(model_results, self.threshold, label=self.label, max_bbox_num=5)
        if show:
            _ = self.show(image_in_, results_after)
        return results_after


if __name__ == '__main__':
    # image_p = 'resource/for_pose/t_pose_1080p.jpeg'
    # img_bgr = CVImage(image_p).bgr
    # hd = HandDetectorYolox(model='yolox_s_trt16', threshold=0.5, provider='gpu')  # yolox_s_trt16
    # hd_result = hd.forward(img_bgr, show=True)
    # print(hd_result)
    #
    # with MyFpsCounter('model forward 10 times fps:') as mfc:
    #     for i in range(10):
    #         bboxes = hd.forward(img_bgr)

    # # video detect and show
    # hd = HandDetectorYolox(model='yolox_s_local', threshold=0.5)
    # with CVVideoLoader('') as cvvl:
    #     for _ in tqdm(range(len(cvvl))):
    #         _, img = cvvl.get()
    #         hd_result = hd.forward(img, show=True)

    # video detect and show to video
    hd = HandDetectorYolox(model='yolox_s_local', threshold=0.5)
    count = 0
    with CVVideoLoader('') as cvvl:
        for _ in tqdm(range(len(cvvl))):
            _, img = cvvl.get()
            hd_result = hd.forward(img, show=False)
            out_img = hd.show(img, hd_result)
            CVImage(out_img).save(f'./cache/hand_out/{count}.jpg', create_path=True)
            count += 1

    # # video 2 pkl
    # from cv2box import CVFile
    #
    # for video_name in ['268', '617', '728', '886']:
    #     result_list = []
    #     video_p = '/{}.mp4'.format(
    #         video_name)
    #     cap = cv2.VideoCapture(video_p)
    #     hd = HandDetectorYolox(0.5)
    #     while True:
    #         success, img = cap.read()
    #         if not success:
    #             break
    #         hd_result = hd.forward(img, show=True)
    #
    #         person_results = []
    #         for bbox in hd_result[0]:
    #             person = {'bbox': np.concatenate([bbox, [1]])}
    #             person_results.append(person)
    #         result_list.append(person_results)
    #
    #         # result_list.append(hd_result[0])
    #
    #     CVFile(video_p.replace('.mp4', '_hand_bbox_out.pkl')).pickle_write(result_list)
