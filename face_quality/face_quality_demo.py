# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from pathlib import Path
from mmcls.apis import inference_model, init_model, show_result_pyplot
import shutil
from tqdm import tqdm
import os
import uuid


def make_random_name(f_name):
    return uuid.uuid4().hex + '.' + f_name.split('.')[-1]


def main():
    parser = ArgumentParser()
    parser.add_argument('--img_dir', default='', help='Image file')
    parser.add_argument('--config', default='config.py', help='Config file')
    parser.add_argument('--checkpoint', default='pretrain_models/face_quality/epoch_23.pth', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    model.CLASSES = ['high', 'low']

    img_dir = Path(args.img_dir)
    with open('', 'a') as f:
        for img_p in tqdm(list(img_dir.rglob('*.jpg'))):
            # test a single image
            img_p_str = str(img_p)
            result = inference_model(model, img_p_str)

            if result['pred_class'] == 'high':
                f.write(img_p_str[53:])
                f.write('\n')

    out_high = ''
    out_low = ''
    for img_p in tqdm(list(img_dir.rglob('*.jpg'))):
        img_p_str = str(img_p)
        result = inference_model(model, img_p_str)
        if result['pred_class'] == 'high':
            shutil.copyfile(img_p, out_high + make_random_name(str(img_p)))
        elif result['pred_class'] == 'low':
            shutil.copyfile(img_p, out_low + make_random_name(str(img_p)))
            pass
        else:
            print(result['pred_class'])


if __name__ == '__main__':
    main()
