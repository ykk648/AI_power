import torch
from AI_power.face_feature import FaceEmbedding

facenet = FaceEmbedding(model_type='cur', gpu_ids=[0])
face = torch.randn((1, 3, 112, 112)).cuda()


def torch2jit(model_, input_):
    traced_gpu = torch.jit.trace(facenet, face)
    torch.jit.save(traced_gpu, "gpu.tjm")
