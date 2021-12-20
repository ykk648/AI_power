# -- coding: utf-8 --
# @Time : 2021/12/15
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import os
import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class TRTModel:
    def __init__(self, trt_model_path):

        self.trt_model_path = trt_model_path

        # 加载换脸模型
        # import pycuda.driver as cuda
        # import pycuda.autoinit
        self.cfx = cuda.Device(0).make_context()
        trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), "")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        # f = load_encrypt_model(MODEL_P)

        # device_name = cuda.Device(0).name()

        f = open(self.trt_model_path, "rb")
        self.engine = runtime.deserialize_cuda_engine(f.read())
        # self.engine = runtime.deserialize_cuda_engine(load_encrypt_model(model_p))
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.host_inputs = []
        self.host_outputs = []
        self.host_size = 10
        for i in range(self.host_size):
            self.host_outputs.append([])

        self.cuda_inputs = []

        self.cuda_outputs = []
        self.bindings = []
        self.shape_outputs = []
        self.nbytes_list = []

        # # windows下需要写死，不然报错
        # size_list = [196608, 512, 65536, 196608]

        idx = 0
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            # size = size_list[idx]
            size = trt.volume(shape) * self.engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float16)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                for i in range(self.host_size):
                    self.host_outputs[i].append(host_mem)
                self.cuda_outputs.append(cuda_mem)
                self.shape_outputs.append(shape)
                self.nbytes_list.append(host_mem.nbytes)
            idx += 1
            if idx == 4:
                break
        self.host_idx = 0
        # 将init阶段的context pop掉，不然退出时pycuda会报错
        self.cfx.pop()

    def predict(self, input):  # result gets copied into output
        self.cfx.push()
        # Transfer input data to device
        # import pycuda.driver as cuda
        # import pycuda.autoinit
        for i in range(len(self.cuda_inputs)):
            np.copyto(self.host_inputs[i], input[i].ravel())
            cuda.memcpy_htod_async(self.cuda_inputs[i], self.host_inputs[i], self.stream)
        # Execute model
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        # Transfer predictions back
        output = []
        for i in range(len(self.cuda_outputs)):
            cuda.memcpy_dtoh_async(self.host_outputs[self.host_idx][i], self.cuda_outputs[i], self.stream)
            output.append(self.host_outputs[self.host_idx][i].reshape((-1,) + self.shape_outputs[i][1:]))
        self.host_idx = 0 if self.host_idx + 1 >= self.host_size else self.host_idx + 1
        # Syncronize threads
        self.stream.synchronize()
        self.cfx.pop()
        return output
