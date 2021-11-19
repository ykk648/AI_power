# -- coding: utf-8 --
# @Time : 2021/11/19
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
from face_restore.dfdnet.options.test_options import DFDTestOptions
from face_detect_and_align import FaceAlignment, LandmarksType
from .models import create_model
from .util.util import tensor2im
from utils import load_img_rgb
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import cv2


# def AddUpSample(img):
#     return img.resize((512, 512), Image.BICUBIC)

# def AddUpSample(img):
#     return cv2.resize(img, (512, 512))

def get_part_location(Landmarks):
    Map_LE = list(np.hstack((range(17, 22), range(36, 42))))
    Map_RE = list(np.hstack((range(22, 27), range(42, 48))))
    Map_NO = list(range(29, 36))
    Map_MO = list(range(48, 68))
    try:
        # left eye
        Mean_LE = np.mean(Landmarks[Map_LE], 0)
        L_LE = np.max((np.max(np.max(Landmarks[Map_LE], 0) - np.min(Landmarks[Map_LE], 0)) / 2, 16))
        Location_LE = np.hstack((Mean_LE - L_LE + 1, Mean_LE + L_LE)).astype(int)
        # right eye
        Mean_RE = np.mean(Landmarks[Map_RE], 0)
        L_RE = np.max((np.max(np.max(Landmarks[Map_RE], 0) - np.min(Landmarks[Map_RE], 0)) / 2, 16))
        Location_RE = np.hstack((Mean_RE - L_RE + 1, Mean_RE + L_RE)).astype(int)
        # nose
        Mean_NO = np.mean(Landmarks[Map_NO], 0)
        L_NO = np.max((np.max(np.max(Landmarks[Map_NO], 0) - np.min(Landmarks[Map_NO], 0)) / 2, 16))
        Location_NO = np.hstack((Mean_NO - L_NO + 1, Mean_NO + L_NO)).astype(int)
        # mouth
        Mean_MO = np.mean(Landmarks[Map_MO], 0)
        L_MO = np.max((np.max(np.max(Landmarks[Map_MO], 0) - np.min(Landmarks[Map_MO], 0)) / 2, 16))
        Location_MO = np.hstack((Mean_MO - L_MO + 1, Mean_MO + L_MO)).astype(int)
    except:
        return 0
    return torch.from_numpy(Location_LE).unsqueeze(0), torch.from_numpy(Location_RE).unsqueeze(0), torch.from_numpy(
        Location_NO).unsqueeze(0), torch.from_numpy(Location_MO).unsqueeze(0)


def obtain_inputs(A, Landmark_path):
    # A_paths = img_path
    # A = Image.open(A_paths).convert('RGB')
    Part_locations = get_part_location(Landmark_path)
    if Part_locations == 0:
        return 0
    C = A
    # A = AddUpSample(A)
    A = cv2.resize(A, (512, 512))
    A = transforms.ToTensor()(A)
    C = transforms.ToTensor()(C)
    A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)  #
    C = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C)  #
    return {'A': A.unsqueeze(0), 'C': C.unsqueeze(0), 'Part_locations': Part_locations}


class DFDNet:
    def __init__(self, use_gpu=True):
        self.gpu = use_gpu

        opt = DFDTestOptions().parse()
        dev = 'cuda:0' if self.gpu else 'cpu'
        self.fa = FaceAlignment(LandmarksType._2D, device=dev, flip_input=False)
        self.model = create_model(opt)
        self.model.setup(opt)

    def forward(self, img_):
        """
        Args:
            img_: BGR image or path
        Returns: BGR image
        """
        if type(img_) == str:
            image_rgb = load_img_rgb(img_)
        else:
            image_rgb = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        try:
            preds_all = self.fa.get_landmarks(img_)
            ins = 0
            if len(preds_all) != 1:
                hights = []
                for l in preds_all:
                    hights.append(l[8, 1] - l[19, 1])
                ins = hights.index(max(hights))
            preds = preds_all[ins]

            data = obtain_inputs(image_rgb, preds[:, 0:2])
            self.model.set_input(data)
            self.model.test()
            visuals = self.model.get_current_visuals()
            im_data = visuals['fake_A']
            image_numpy = tensor2im(im_data)
            # image_numpy = visuals['fake_A'][0].cpu().float().clamp_(-1, 1).numpy().transpose(1,2,0)
            return image_numpy
        except Exception as e:
            print('\t################ Error in enhancing this image: {}'.format(str(e)))
            raise e
