# -- coding: utf-8 --
# @Time : 2023/5/23
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
from paddleocr import PaddleOCR, draw_ocr

# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
img_path = ''
result = ocr.ocr(img_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)

# 显示结果
# 如果本地没有simfang.ttf，可以在doc/fonts目录下下载
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='./pretrain_models/ocr_lib/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result2.jpg')
