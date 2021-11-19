# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from pathlib import Path
from mmcls.apis import inference_model, init_model
import shutil
from tqdm import tqdm
import cv2
from utils.ai_utils import make_random_name


def main():
    parser = ArgumentParser()
    parser.add_argument('--img_dir', default='', help='Image file')
    parser.add_argument('--config', default='config.py', help='Config file')
    parser.add_argument('--checkpoint',
                        default='pretrain_models/gender_classification/gender_classification_epoch_20.pth',
                        help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    model.CLASSES = ['male', 'female']
    img_dir = Path(args.img_dir)

    # with open('test.txt', 'a') as f:
    #     for img_p in tqdm(list(img_dir.rglob('*.jpg'))):
    #         # test a single image
    #         img_p_str = str(img_p)
    #         result = inference_model(model, img_p_str)
    #         # f.write(img_p.strip(root_p) + ' ' + result['pred_class'])
    #         # f.write('\n')
    #
    #         if result['pred_class'] == 'high':
    #             f.write(img_p_str[53:])
    #             f.write('\n')

    out_high = '/workspace/male'
    out_low = '/workspace/female'
    for img_p in tqdm(list(img_dir.rglob('*.jpg'))):
        img_p_str = str(img_p)
        try:
            result = inference_model(model, img_p_str)
        except cv2.error:
            result = None
            print(img_p)

        if result['pred_class'] == 'male':
            shutil.copyfile(img_p, out_high + '/' + make_random_name(str(img_p)))
        elif result['pred_class'] == 'female':
            shutil.copyfile(img_p, out_low + '/' + make_random_name(str(img_p)))
        else:
            print(result['pred_class'])


if __name__ == '__main__':
    main()
