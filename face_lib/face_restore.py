# -- coding: utf-8 --
# @Time : 2021/11/17
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

from face_lib.face_restore import FaceRestore
from cv2box import CVImage, CVVideoLoader, get_path_by_ext, MyFpsCounter
from cv2box.cv_ops.cv_video import CVVideoLoader
from tqdm import tqdm
import cv2

if __name__ == '__main__':
    # # === for image ===
    # face_img_p = 'resource/cropped_face/512.jpg'
    # fa = FaceRestore(use_gpu=False, mode='gfpganv4')  # gfpgan gfpganv3 gfpganv4 gpen gpen2048 dfdnet RestoreFormer CodeFormer
    # with MyFpsCounter() as mfc:
    #     face = fa.forward(face_img_p, output_size=512)
    # # CVImage(face, image_format='cv2').save('./gfpgan.jpg')
    # CVImage(face, image_format='cv2').show()

    # === for image dir ===
    face_img_dir = 'resource/test1'
    fa = FaceRestore(use_gpu=False,
                     mode='RestoreFormer')  # gfpgan gfpganv3 gfpganv4 gpen gpen2048 dfdnet RestoreFormer CodeFormer
    for img_p in get_path_by_ext(face_img_dir):
        face = fa.forward(str(img_p), output_size=512)
        CVImage(face, image_format='cv2').save(str(img_p).replace('.', '_RestoreFormer.'))

    # # === for aligned video ===
    # fa = FaceRestore(use_gpu=True, mode='gfpganv4')
    # video_p = ''
    # video_out_p = ''
    #
    # video_writer = cv2.VideoWriter(video_out_p, cv2.VideoWriter_fourcc(*'mp4v'), 25, (256, 256))
    # with CVVideoLoader(video_p) as cvv:
    #     for _ in tqdm(range(len(cvv))):
    #         success, frame = cvv.get()
    #         frame_out = fa.forward(frame, output_size=256)
    #         # CVImage(frame_out).show()
    #         video_writer.write(frame_out)
