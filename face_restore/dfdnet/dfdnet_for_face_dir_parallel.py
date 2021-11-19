from options.test_options import DFDTestOptions
from models import create_model
from util.visualizer import save_crop
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
from skimage import io
from tqdm import tqdm
from face_detect_and_align import face_alignment_1adrianb
# from multiprocessing import Process, Lock, Queue # multi process
from multiprocessing.dummy import Process, Queue  # multi thread
import time
import queue
from utils.ai_utils import get_path_by_ext
from pathlib import Path

try:
    from cv2box import flush_print as fp
except ModuleNotFoundError:
    fp = print


class Consumer(Process):
    def __init__(self, queue_list: list, opt, block=True, fps_counter=True):
        super().__init__()
        self.queue_list = queue_list
        self.fps_counter = fps_counter
        self.block = block
        self.opt = opt
        # self.pid = os.getpid()

        self.init_func()

        # print('init 1 consumer, pid is {}.'.format(self.pid))

    def init_func(self):
        dev = 'cuda:{}'.format(self.opt.gpu_ids[0])
        # dev = 'cpu'
        self.FD = face_alignment_1adrianb.FaceAlignment(face_alignment_1adrianb.LandmarksType._2D, device=dev,
                                                        flip_input=False)

    def forward_func(self, something_in):

        Img = io.imread(something_in[0])
        try:
            PredsAll = self.FD.get_landmarks(Img)
        except:
            print('\t################ Error in face detection, continue...')
            return None
        if PredsAll is None:
            print('\t################ No face, continue...')
            return None
        ins = 0
        if len(PredsAll) != 1:
            hights = []
            for l in PredsAll:
                hights.append(l[8, 1] - l[19, 1])
            ins = hights.index(max(hights))
        preds = PredsAll[ins]

        return preds[:, 0:2], something_in[0], something_in[1]

    def run(self):

        counter = 0
        queue_full_counter = 0
        start_time = time.time()

        while True:

            something_in = self.queue_list[0].get()
            something_out = self.forward_func(something_in)
            if something_out is not None:
                if self.block:
                    self.queue_list[1].put(something_out)
                else:
                    try:
                        self.queue_list[1].put_nowait(something_out)
                    except queue.Full:
                        # do your judge here, for example
                        queue_full_counter += 1
                        if (time.time() - start_time) > 10:
                            fp('Queue full {} times'.format(queue_full_counter))

                if self.fps_counter:
                    counter += 1
                    if (time.time() - start_time) > 10:
                        fp("Consumer1 FPS: ", counter / (time.time() - start_time))
                        counter = 0
                        start_time = time.time()


class Consumer2(Process):
    def __init__(self, queue_list: list, opt, block=True, fps_counter=True):
        super().__init__()
        self.queue_list = queue_list
        self.fps_counter = fps_counter
        self.block = block
        self.opt = opt
        # self.pid = os.getpid()

        self.init_func()

        # print('init 2 consumer, pid is {}.'.format(self.pid))

    def init_func(self):
        self.model = create_model(self.opt)
        self.model.setup(self.opt)

    def forward_func(self, something_in):
        # torch.cuda.empty_cache()
        data = obtain_inputs(something_in[1], something_in[0])
        if data == 0:
            print('\t################ Error in landmark file, continue...')
            return
        self.model.set_input(data)
        try:
            self.model.test()
            visuals = self.model.get_current_visuals()
            save_crop(visuals, something_in[2])
        except Exception as e:
            print('\t################ Error in enhancing this image: {}'.format(str(e)))
            print('\t################ continue...')
            return

    def run(self):

        counter = 0
        queue_full_counter = 0
        start_time = time.time()

        while True:
            something_in = self.queue_list[0].get()

            # exit condition
            if something_in is None:
                print('subprocess {} exit !'.format(self.pid))
                break

            self.forward_func(something_in)

            if self.fps_counter:
                counter += 1
                if (time.time() - start_time) > 10:
                    fp("Consumer2 FPS: ", counter / (time.time() - start_time))
                    counter = 0
                    start_time = time.time()


def AddUpSample(img):
    return img.resize((512, 512), Image.BICUBIC)


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


def obtain_inputs(A_paths, Landmark_path):
    # A_paths = img_path
    A = Image.open(A_paths).convert('RGB')
    Part_locations = get_part_location(Landmark_path)
    if Part_locations == 0:
        return 0
    C = A
    A = AddUpSample(A)
    A = transforms.ToTensor()(A)
    C = transforms.ToTensor()(C)
    A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)  #
    C = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C)  #
    return {'A': A.unsqueeze(0), 'C': C.unsqueeze(0), 'A_paths': A_paths, 'Part_locations': Part_locations}


class DFDNetParallel:
    def __init__(self, ):
        self.opt = DFDTestOptions().parse()
        self.opt.serial_batches = True  # no shuffle
        self.opt.no_flip = True  # no flip
        self.opt.display_id = -1  # no visdom display
        self.opt.which_epoch = 'latest'  #

    def forward(self, input_dir_path, output_dir_path):
        q1 = Queue(10)
        q2 = Queue(10)
        c1 = Consumer([q1, q2], self.opt, fps_counter=False)
        c2 = Consumer2([q2], self.opt, fps_counter=False)
        c1.start()
        c2.start()

        exts = [".jpg", ".png", ".JPG", ".webp", ".jpeg"]

        for img_path in tqdm(list(get_path_by_ext(input_dir_path, exts))):
            img_save_path = Path(str(img_path.with_suffix('.jpg')).replace(input_dir_path, output_dir_path))
            img_save_path.parent.mkdir(parents=True, exist_ok=True)
            if not img_save_path.exists():
                q1.put([img_path, img_save_path])
