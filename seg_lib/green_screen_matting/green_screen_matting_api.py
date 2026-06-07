#!/usr/bin/env python
# coding=utf-8
"""
Green screen matting inference module.
Based on BackgroundMattingV2 architecture for chroma-key video matting.
Supports frame-by-frame matting with background replacement and video output.
"""
import argparse
import os
import sys
import shutil
import subprocess

import cv2
import torch
import torch as t
import numpy as np
import pandas as pd
import time
import threading
from multiprocessing import Queue
from torch import nn
from torch.nn import functional as F
from threading import Thread
from tqdm import tqdm
import glob

# Resolve paths relative to this module's location
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add module dir to path so model package can be imported
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from model import MattingBase, MattingRefine


# Default model checkpoint path (relative to this module)
_DEFAULT_CHECKPOINT = os.path.join(_MODULE_DIR, 'content', 'pytorch_resnet50.pth')
_DEFAULT_GREEN_SCREEN = os.path.join(_MODULE_DIR, 'data', '绿幕1.png')
_DEFAULT_NEW_BG = os.path.join(_MODULE_DIR, 'data', '0.jpg')


class ModelConfig:
    def __init__(self, device_type, output_path, model_checkpoint=None):
        self.model_type = 'mattingrefine'
        self.model_backbone = 'resnet50'
        self.model_backbone_scale = 0.5
        self.model_checkpoint = model_checkpoint or _DEFAULT_CHECKPOINT
        self.model_refine_mode = 'sampling'
        self.model_refine_sample_pixels = 80_000
        self.model_refine_threshold = 0.7
        self.model_refine_kernel_size = 3
        self.device = device_type
        self.output_dir = output_path
        self.output_types = 'fuse'
        self.batch_size = 16
        self.img_size = (0, 0)
        self.output_format = 'image_sequences'
        self.src_video_path = ''
        self.img_size = ''
        self.new_path = ''
        self.bg_path = ''
        self.video_shape = ''


def get_src_bgr(
        src_path,
        bg_path,
        new_path,
        video_shape,
        output_path,
        device_type,
        process_size=(0, 0),
        batch_size=5,
        num_workers=6):
    opt = ModelConfig(device_type, output_path)
    opt.src_video_path = src_path
    opt.img_size = process_size
    opt.new_path = new_path
    opt.bg_path = bg_path
    opt.video_shape = video_shape
    dataset_src = _FrameLoader(opt)
    return dataset_src


class _FrameLoader:
    def __init__(self, opt, QueueSize=100):
        self.img_size = opt.img_size
        self.video_path = opt.src_video_path
        self.bg_path = opt.bg_path
        self.new_path = opt.new_path
        self.vc = cv2.VideoCapture(self.video_path)
        self.width = int(self.vc.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if max(self.width, self.height) > 3000:
            self.width = int(self.width // 2)
            self.height = int(self.height // 2)
        self.fps = self.vc.get(cv2.CAP_PROP_FPS)
        self.length = self.vc.get(7)
        self.frame_queue_in = Queue(QueueSize)
        self.shape = (self.height, self.width, 3)
        self.thread = threading.Thread(target=self._worker, args=(), )
        self.thread.start()
        self.device = torch.device('cuda')

    def _worker(self):
        if self.vc.isOpened():
            rval = True
            while rval:
                rval, new_img = self.vc.read()
                if rval:
                    if max(new_img.shape) > 3000:
                        new_img = cv2.resize(new_img, (int(new_img.shape[1] // 2), int(new_img.shape[0] // 2)))
                    self.frame_queue_in.put((new_img))

    def join(self):
        self.thread.join()


class _MattingHandler(object):

    def __init__(self, video_path, bg_path, new_path, output_path, video_shape, device_type,
                 model_checkpoint=None):
        self.video_path = video_path
        self.bg_path = bg_path
        self.bg_image = new_path
        self.bg_green = None
        self.process_size = (0, 0)
        self.pha_process_size = (0, 0)
        self.output_size = (0, 0)
        self.device_type = device_type
        self.output_path = output_path
        self.video_shape = video_shape
        self.model_checkpoint = model_checkpoint
        self.q12 = Queue()

    def work(self):
        args = ModelConfig(self.device_type, self.output_path, model_checkpoint=self.model_checkpoint)
        self.device = torch.device(args.device)

        # Load model
        if args.model_type == 'mattingbase':
            self.model = MattingBase(args.model_backbone)
        if args.model_type == 'mattingrefine':
            self.model = MattingRefine(
                args.model_backbone,
                args.model_backbone_scale,
                args.model_refine_mode,
                args.model_refine_sample_pixels,
                args.model_refine_threshold,
                args.model_refine_kernel_size)

        self.model.to(self.device).eval()
        self.model.load_state_dict(
            torch.load(
                args.model_checkpoint,
                map_location=self.device),
            strict=False)

        time_start = time.time()
        tmp = 1
        dataset_src = get_src_bgr(
            self.video_path,
            self.bg_path,
            self.bg_image,
            self.video_shape,
            output_path=self.output_path,
            device_type=self.device_type,
            process_size=self.process_size,
            batch_size=tmp, )

        video_name = os.path.splitext(os.path.basename(self.video_path))[0] + '_lb'
        self.out = cv2.VideoWriter(
            os.path.join(self.output_path, video_name + '.avi'),
            cv2.VideoWriter_fourcc(*'MJPG'),
            dataset_src.fps,
            (dataset_src.width, dataset_src.height))

        bg_img = cv2.imread(self.bg_path)
        green_img = t.from_numpy(bg_img).permute(2, 0, 1).to(self.device).unsqueeze(0).repeat((tmp, 1, 1, 1))
        if max(dataset_src.shape) > 3000:
            green_img = F.interpolate(green_img, (dataset_src.shape[0] // 2, dataset_src.shape[1] // 2))
        green_img = F.interpolate(green_img, (dataset_src.shape[0], dataset_src.shape[1]))
        new_img = cv2.imread(self.bg_image)
        new_img = t.from_numpy(new_img).permute(2, 0, 1).to(self.device).unsqueeze(0).repeat((tmp, 1, 1, 1))
        if max(dataset_src.shape) > 3000:
            new_img = F.interpolate(new_img, (dataset_src.shape[0] // 2, dataset_src.shape[1] // 2))
        new_img = F.interpolate(new_img, (dataset_src.shape[0], dataset_src.shape[1]))
        time_middle = time.time()
        print('加载数据所用总时间:{}s'.format(time_middle - time_start))
        t2 = 0
        count = 0
        with t.no_grad():
            for _ in tqdm(range(100000)):
                count += 1
                if count > dataset_src.length:
                    break

                srcs = dataset_src.frame_queue_in.get()
                srcs = t.from_numpy(srcs).to(self.device).unsqueeze(0).permute(0, 3, 1, 2)
                if t2:
                    pass
                pha, img2sg_t, err = self._run_one_batch(srcs, green_img)
                constrain4_t = self._constrain_vt2(1, [0, 2], srcs, pha, self.device)
                pha = (pha ** 2 * (1 - constrain4_t))
                img2sg_t = self._constrain_vt(1, [0, 2], img2sg_t)
                img2sg_t = (img2sg_t * 255 * pha + (1 - pha) * new_img).permute(0, 2, 3, 1)

                Thread(target=self._write_img, args=(img2sg_t[0],)).start()

        dataset_src.join()
        time_end = time.time()
        print('抠图所用总时间:{}s'.format((dataset_src.length) / (time_end - time_middle)))

    def _write_img(self, frame):
        frame0 = frame.cpu().numpy().astype(np.uint8)
        self.out.write(frame0[:, :, :3])

    def _run_one_batch(self, srcs, bg_green=None):
        b_s = srcs.shape[0]
        if bg_green is None:
            bgr_p4 = self.bg_green.repeat(b_s, 1, 1, 1)
            if bgr_p4.shape[2] != srcs.shape[2] or bgr_p4.shape[3] != srcs.shape[3]:
                bgr_p4 = F.interpolate(bgr_p4, (srcs.shape[2], srcs.shape[3]))
        else:
            bgr_p4 = bg_green

        if self.pha_process_size[0] != 0:
            srcsm = F.interpolate(srcs, self.pha_process_size)
            bgr_p4m = F.interpolate(bgr_p4, self.pha_process_size)
            pham, fgrm, _, _, errm, _ = self.model(srcsm / 255., bgr_p4m / 255.)
            pha = F.interpolate(pham, (srcs.shape[2], srcs.shape[3]))
        elif (srcs.shape[2] % 4 != 0) or (srcs.shape[3] % 4 != 0):
            loc_process_size = (
                srcs.shape[2] - srcs.shape[2] % 4,
                srcs.shape[3] - srcs.shape[3] % 4)
            srcsm = F.interpolate(srcs, loc_process_size)
            bgr_p4m = F.interpolate(bgr_p4, loc_process_size)
            pham, fgrm, _, _, errm, _ = self.model(srcsm / 255., bgr_p4m / 255.)
            pha = F.interpolate(pham, (srcs.shape[2], srcs.shape[3]))
        else:
            pha, fgr, _, _, err, _ = self.model(srcs / 255., bgr_p4 / 255.)

        return pha, fgr, err

    def _constrain_vt2(self, target_ch, other_ch, srcs, pha, device):
        constrain_t = (srcs[:, target_ch, :, :] + 1) / \
                      (srcs[:, other_ch, :, :].float().mean(1) + 1)
        slt_min, slt_max = [1.8, 4.2]
        constrain2_t = t.clip(constrain_t, slt_min, slt_max)
        constrain3_t = (constrain2_t - slt_min) / (slt_max - slt_min)
        constrain3_t = constrain3_t * (pha[:, 0, :, :])
        constrain3_tr = F.interpolate(constrain3_t.unsqueeze(1),
                                      (int(constrain3_t.shape[1] / 4), int(constrain3_t.shape[2] / 4)))
        conv_wt = (t.ones(1, 1, 7, 7) / (7 * 7 * 1)).to(device) * 1.05
        padding = nn.ReflectionPad2d(3)
        constrain3_trconv = F.conv2d(padding(constrain3_tr), conv_wt, padding=0)
        constrain3_t2 = constrain3_trconv * constrain3_tr
        constrain3_t2conv = F.conv2d(padding(constrain3_t2), conv_wt, padding=0)
        constrain3_t2conv = t.clip(constrain3_t2conv, 0, 1.)
        constrain3_t2conv2 = F.interpolate(
            constrain3_t2conv, (int(constrain3_t.shape[1]), int(constrain3_t.shape[2])))

        slt_min2, slt_max2 = [10., 28.]
        img_gray_t = (t.clip(srcs[:, other_ch, :, :, ].float().mean(1),
                             slt_min2, slt_max2) / slt_max2) ** 0.5
        img_gray2_t = ((img_gray_t - 0.59) / (1 - 0.59)) ** 0.2
        img_gray2_t = img_gray2_t.to(device)
        constrain4_t = (constrain3_t2conv2 * img_gray2_t.unsqueeze(1))
        return constrain4_t

    def _constrain_vt(self, target_ch, other_ch, src_p_t):
        constrain_t = (src_p_t[:, target_ch, :, :]) > (src_p_t[:, other_ch, :, :].mean(1))
        constrained_t = src_p_t[:, target_ch, :, :] * (1 - 1 * constrain_t) + \
                        (constrain_t * 1) * src_p_t[:, other_ch, :, :].mean(1)
        src_p_t2 = src_p_t + 0
        src_p_t2[:, target_ch, :, :] = constrained_t
        return src_p_t2


def green_screen_matting(video_path, bg_path, new_path, output_path, video_shape, device_type,
                         model_checkpoint=None):
    """
    Run green screen matting on a video.

    Args:
        video_path: Input video file path.
        bg_path: Green screen reference image path.
        new_path: New background image path.
        output_path: Output directory for result video.
        video_shape: Video shape tuple (height, width, 3).
        device_type: Device string, 'cuda' or 'cpu'.
        model_checkpoint: Optional path to model weights file.
    """
    handler = _MattingHandler(video_path, bg_path, new_path, output_path, video_shape, device_type,
                              model_checkpoint=model_checkpoint)
    handler.work()


def get_video_bps(video_path):
    if not os.path.exists(video_path):
        print('视频不存在%s!!!!' % video_path)
    temp_txt_path = os.path.splitext(video_path)[0] + '.txt'
    os.system('ffmpeg -loglevel info -i %s > %s 2>&1' % (video_path, temp_txt_path))
    max_volume = 0
    try:
        contents = open(temp_txt_path, 'r').readlines()
        for line in contents:
            if 'Duration' in line:
                max_volume = line.split('bitrate:')[1].split('kb/s')[0]
                max_volume = int(max_volume)
                break
    except (FileNotFoundError, IndexError, ValueError):
        pass
    try:
        os.remove(temp_txt_path)
    except OSError:
        pass
    return max_volume


if __name__ == '__main__':
    init_parser = argparse.ArgumentParser(description='Green screen matting inference')
    init_parser.add_argument("--name", help="video name (used for result/<name>/test_video/ path)",
                             default='', type=str)
    init_parser.add_argument("--lb", help="green screen reference image path", default='', type=str)
    init_parser.add_argument("--new_bg", help="new background image path", default='', type=str)

    ar = init_parser.parse_args()
    output_tmp_dir = os.path.join(os.path.dirname(_MODULE_DIR), 'result', ar.name, 'test_video')
    if not os.path.exists(output_tmp_dir):
        print('视频目录不存在，请检查name！！！')
        exit()
    if ar.lb == '' or (not os.path.exists(ar.lb)):
        print('未输入绿幕图片，将会使用默认绿幕图片！！！')
        ar.lb = _DEFAULT_GREEN_SCREEN

    if ar.new_bg == '' or (not os.path.exists(ar.new_bg)):
        print('未输入新背景图片，将会使用默认新背景图片！！！')
        ar.new_bg = _DEFAULT_NEW_BG

    mp4_files = glob.glob(os.path.join(output_tmp_dir, '*.mp4'))
    for f in mp4_files:
        print(f)
        if not os.path.exists(f):
            continue
        videoCapture_temp = cv2.VideoCapture(f)
        bps = get_video_bps(f)
        Frame_Width = videoCapture_temp.get(cv2.CAP_PROP_FRAME_WIDTH)
        Frame_Height = videoCapture_temp.get(cv2.CAP_PROP_FRAME_HEIGHT)
        video_shape = (int(Frame_Height), int(Frame_Width), 3)
        audio_path = os.path.join(output_tmp_dir, os.path.splitext(os.path.basename(f))[0] + '.wav')
        os.system('ffmpeg -y -loglevel quiet -i {} {}'.format(f, audio_path))

        green_screen_matting(
            f,
            ar.lb,
            ar.new_bg,
            output_tmp_dir,
            video_shape,
            'cuda')
        video_name = os.path.splitext(os.path.basename(f))[0] + '_lb'
        avi_path = os.path.join(output_tmp_dir, video_name + '.avi')
        result_mp4_path = os.path.join(output_tmp_dir, video_name + '.mp4')
        if os.path.exists(avi_path):
            os.system('ffmpeg -y -i {} -i {} -loglevel quiet -strict -2 -q:v 1 -b:v {}k {}'.format(
                audio_path, avi_path, bps, result_mp4_path))
