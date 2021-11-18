from face_restore import FaceRestore
from ai_utils import img_save, img_show

if __name__ == '__main__':
    face_img_p = 'test_img/fake_112.png'
    fa = FaceRestore(mode='gpen')
    # fa = FaceAlign('mtcnn')
    face = fa.forward(face_img_p, output_size=256)
    print(face.shape)
    img_show(face)
