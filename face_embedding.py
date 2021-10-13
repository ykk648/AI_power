from face_feature import FaceEmbedding

if __name__ == '__main__':

    # CurricularFace
    fb = FaceEmbedding(model_type='cur', gpu_ids=[0])
    # # ArcFace
    # fb = FaceEmbedding(model_type='arc', gpu_ids=[0])
    fb.latent_from_image('test_img/rb.png')
