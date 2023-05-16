from body_lib import BodyBboxDetector
from mocap_lib import BodyWholebodyDetector
from mocap_lib.skeleton_transfer.cocowholebody_2_openpose import cocowb_2_openpose
from cv2box import CVImage, CVFile
from cv2box.cv_gears import CVVideoThread, Consumer, Linker, Queue
import numpy as np


class BodyBboxThread(Linker):
    def __init__(self, queue_list: list, fps_counter):
        super().__init__(queue_list, fps_counter=fps_counter)

        self.bbd = BodyBboxDetector(model='yolox_tiny_trt16', threshold=0.5, provider='gpu')

    def forward_func(self, something_in):
        # do your work here.
        image_in = something_in
        something_out = [image_in, self.bbd.forward(image_in, max_bbox_num=1)[0]]
        return something_out


class BodyKpThread(Consumer):
    def __init__(self, queue_list: list, out_pkl_path_, fps_counter):
        super().__init__(queue_list, fps_counter=fps_counter)
        # add init here
        self.bwd = BodyWholebodyDetector(model_type='hrnet_w48_384_dark', provider='gpu')
        self.out_pkl_path = out_pkl_path_
        self.kp_list = []

    def exit_func(self):
        super(BodyKpThread, self).exit_func()
        if self.out_pkl_path:
            CVFile(self.out_pkl_path).pickle_write(np.array(self.kp_list))

    def forward_func(self, something_in):
        # do your work here.
        something_out = self.bwd.forward(something_in[0], something_in[1], show=False, mirror_test=False)

        left_hand_kps, right_hand_kps, openpose_25_kps = cocowb_2_openpose(something_out)
        whole_kps = np.concatenate((openpose_25_kps, left_hand_kps, right_hand_kps), 0)
        if self.out_pkl_path:
            self.kp_list.append(whole_kps)
        # print(whole_kps.shape)
        # print(right_hand_kps)
        # print(openpose_25_kps)


if __name__ == '__main__':
    # 4K 67 fps
    video_p = ''
    out_pkl_path = None

    q1 = Queue(5)
    q2 = Queue(5)
    c1 = CVVideoThread(video_p, [q1], fps_counter=False)
    b1 = BodyBboxThread([q1, q2], fps_counter=True)
    b2 = BodyKpThread([q2], out_pkl_path, fps_counter=True)
    c1.start()
    b1.start()
    b2.start()
