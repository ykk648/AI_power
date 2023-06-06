# -- coding: utf-8 --
# @Time : 2021/11/10
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
from cv2box import CVImage, MyFpsCounter
from cv2box.utils.math import Normalize
from apstone import ModelBase


def down_sample(target_, size):
    import torch.nn.functional as F
    return F.interpolate(target_, size=size, mode='bilinear', align_corners=True)


MODEL_ZOO = {
    # https://github.com/neuralchen/SimSwap/blob/01a8d6d0a6fd7e7b0052a5832328fba33f2b8414/models/fs_model.py#L63
    'arcface_tjm': {
        'model_path': 'pretrain_models/face_lib/face_embedding/ArcFace.tjm'
    },
    # https://github.com/HuangYG123/CurricularFace
    'CurricularFace_tjm': {
        'model_path': 'pretrain_models/face_lib/face_embedding/CurricularFace.tjm'
    },
    'CurricularFace': {
        'model_path': 'pretrain_models/face_lib/face_embedding/CurricularFace.onnx'
    },
    # https://github.com/deepinsight/insightface/tree/master/model_zoo
    # https://github.com/SthPhoenix/InsightFace-REST
    'insightface_mbf': {
        'model_path': 'pretrain_models/face_lib/face_embedding/w600k_mbf.onnx',
    },
}


class FaceEmbedding(ModelBase):
    def __init__(self, model_type='arcface', provider='gpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.model_type = model_type
        if self.model_type == 'arcface_tjm':
            from torchvision import transforms
            self.transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif self.model_type == 'CurricularFace_tjm':
            from torchvision import transforms
            self.transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif model_type in ['insightface_mbf', 'CurricularFace']:
            self.input_std = self.input_mean = 127.5
            self.input_size = (112, 112)

    def forward(self, face_image):
        """
        Args:
            face_image: CVImage acceptable type
        Returns: (512,) or torch.Size([512]) (tjm)
        """
        if self.model_type.find('tjm') > 0:
            face_image = CVImage(face_image).rgb()
            face = self.transformer(face_image)
            face = face.unsqueeze(0)
            if face.shape[2] != 112 or face.shape[3] != 112:
                face = down_sample(face, size=[112, 112])
            face_latent = self.model.forward(face)
            if self.model_type == 'CurricularFace_tjm':
                import torch.nn.functional as F
                face_latent = F.normalize(face_latent, p=2, dim=1)
            return face_latent[0]
        else:
            face = CVImage(face_image).blob(self.input_size, self.input_mean, self.input_std, rgb=True)
            # for batch
            # return Normalize(self.model.forward(face)[0]).batch_norm()
            return Normalize(self.model.forward(face)[0].ravel()).np_norm()


if __name__ == '__main__':
    # CurricularFace
    fb_cur = FaceEmbedding(model_type='CurricularFace', provider='gpu')
    latent_cur = fb_cur.forward('resource/cropped_face/112.png')
    print(latent_cur.shape)
    # print(latent_cur)

    # # ArcFace
    # fe = FaceEmbedding(model_type='arcface_tjm', provider='gpu')
    # with MyFpsCounter() as mfc:
    #     for i in range(10):
    #         latent_arc = fe.forward('resource/cropped_face/112.png')
    # print(latent_arc.shape)
    # print(latent_arc)

    # # insightface MBF
    # fe = FaceEmbedding(model_type='insightface_mbf')
    # latent_mbf_1 = fe.forward('resource/cropped_face/112.png')
    # latent_mbf_2 = fe.forward('resource/cropped_face/112.png')
    # print(latent_mbf_1.shape)
    # print(latent_mbf_1)
    # from cv2box.utils.math import CalDistance
    # print(CalDistance().sim(latent_mbf_1, latent_mbf_2))
