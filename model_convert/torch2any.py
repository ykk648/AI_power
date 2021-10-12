import torch
from face_feature import FaceEmbedding
import torch
import torchvision

facenet = FaceEmbedding(model_type='cur', gpu_ids=[0])
face = torch.randn((1, 3, 112, 112)).cuda()

model = torchvision.models.alexnet(pretrained=True).cuda()
dummy_input = torch.randn(10, 3, 224, 224, device='cuda')


def torch2jit(model_, input_):
    traced_gpu = torch.jit.trace(facenet, face)
    torch.jit.save(traced_gpu, "gpu.tjm")


def torch2onnx(model_, input_):
    input_names = ["actual_input_1"]
    output_names = ["output1"]

    torch.onnx.export(model_, input_, "alexnet.onnx", verbose=True, input_names=input_names,
                      output_names=output_names)
