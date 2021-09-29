from face_feature import FaceEmbedding

if __name__ == '__main__':
    fb = FaceEmbedding(model_type='cur', gpu_ids=[0])
    fb.latent_from_image('test_img/rb.png')
