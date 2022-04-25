from face_lib.face_restore import FaceRestore
from cv2box import CVImage, CVVideoLoader
from cv2box.cv_ops.cv_video import CVVideoLoader
from tqdm import tqdm
import cv2

if __name__ == '__main__':
    # # === for image ===
    # face_img_p = 'test_img/croped_face/512.jpg'
    # fa = FaceRestore(use_gpu=False, mode='gfpgan')  # gfpgan gpen dfdnet
    # face = fa.forward(face_img_p, output_size=256)
    # CVImage(face, image_format='cv').show()

    # === for aligned video ===
    fa = FaceRestore(use_gpu=True, mode='gpen')  # gfpgan gfpganv3 gpen gpen2048 dfdnet
    video_p = ''
    video_out_p = ''

    video_writer = cv2.VideoWriter(video_out_p, cv2.VideoWriter_fourcc(*'mp4v'), 25, (256, 256))
    with CVVideoLoader(video_p) as cvv:
        for _ in tqdm(range(len(cvv))):
            success, frame = cvv.get()
            frame_out = fa.forward(frame, output_size=256)
            # CVImage(frame_out).show()
            video_writer.write(frame_out)
