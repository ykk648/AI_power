from face_lib.face_restore import FaceRestore
from cv2box import CVImage

if __name__ == '__main__':
    # === for image ===
    face_img_p = 'test_img/croped_face/512.jpg'
    fa = FaceRestore(use_gpu=False, mode='gfpgan')  # gfpgan gpen dfdnet
    face = fa.forward(face_img_p, output_size=256)
    CVImage(face, image_format='cv').show()

    # # === for aligned video ===
    # fa = FaceRestore(use_gpu=True, mode='dfdnet')  # gfpgan gpen dfdnet
    # video_p = ''
    # cap = cv2.VideoCapture(video_p)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # frames_num = cap.get(7)
    # video_writer = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    #
    # for _ in tqdm(range(int(frames_num))):
    #     success, frame = cap.read()
    #     frame_out = fa.forward(frame, output_size=256)
    #     video_writer.write(frame_out)