import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import os
import time

trt_exec_dict = {
    7234: './TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1/TensorRT-7.2.3.4/targets/x86_64-linux-gnu/bin/trtexec',
    8003: './TensorRT-8.0.0.3/targets/x86_64-linux-gnu/bin/trtexec',
    8206: './TensorRT-8.2.0.6/targets/x86_64-linux-gnu/bin/trtexec'
}
LD_LIBRARY_PATH = {
    7234: './TensorRT-7.2.3.4.Ubuntu-18.04.x86_64-gnu.cuda-11.1.cudnn8.1/TensorRT-7.2.3.4/targets/x86_64-linux-gnu/lib',
    8003: './TensorRT-8.0.0.3/targets/x86_64-linux-gnu/lib',
    8206: './TensorRT-8.2.0.6/targets/x86_64-linux-gnu/lib'
}
trt_exec_version = 8003
os.environ['LD_LIBRARY_PATH'] = LD_LIBRARY_PATH[trt_exec_version]
if __name__ == '__main__':
    BATCH_SIZE = 16
    USE_FP16 = True
    target_dtype = np.float16 if USE_FP16 else np.float32
    trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), "")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

    f = open("test.trt", "rb")
    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    stream = cuda.Stream()

    host_inputs = []
    cuda_inputs = []
    host_outputs = []
    cuda_outputs = []
    bindings = []
    shape_outputs = []
    shape_inputs = []
    for binding in engine:

        shape = engine.get_binding_shape(binding)
        # engine.max_batch_size = BATCH_SIZE
        size = trt.volume(shape)  # * BATCH_SIZE #engine.max_batch_size  #//(4//BATCH_SIZE)
        host_mem = cuda.pagelocked_empty(size, target_dtype)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(cuda_mem))
        if engine.binding_is_input(binding):
            host_inputs.append(host_mem)
            cuda_inputs.append(cuda_mem)
            shape_inputs.append(shape)
        else:
            host_outputs.append(host_mem)
            cuda_outputs.append(cuda_mem)
            shape_outputs.append(shape)
        is_input = engine.binding_is_input(binding)
        op_type = engine.get_binding_dtype(binding)
        print('is input: ', is_input, '  binding name:', binding, '  shape:', shape, 'type: ', op_type)


    def predict(input):  # result gets copied into output
        # Transfer input data to device

        for i in range(len(cuda_inputs)):
            np.copyto(host_inputs[i], input[i].ravel())
            cuda.memcpy_htod_async(cuda_inputs[i], host_inputs[i], stream)
        # Execute model
        context.execute_async_v2(bindings, stream.handle, None)
        # Transfer predictions back
        output = []
        for i in range(len(cuda_outputs)):
            cuda.memcpy_dtoh_async(host_outputs[i], cuda_outputs[i], stream)
            output.append(host_outputs[i].reshape((-1,) + shape_outputs[i][1:]))
        # Syncronize threads
        stream.synchronize()
        return output


    target = np.load('target.npy')
    vsid = np.load('vsid.npy')
    fake = np.load('fake.npy')
    fake_mask = np.load('fake_mask.npy')
    target = np.repeat(target, BATCH_SIZE, axis=0).astype(target_dtype)
    vsid = np.repeat(vsid, BATCH_SIZE, axis=0).astype(target_dtype)
    output = predict([target, vsid])
    for j in range(100):
        start = time.time()
        for i in range(100):
            output = predict([target, vsid])
            # out=output[1]*output[0]+target*(1-output[1])
        print('fps:%d' % (100 / (time.time() - start)))

    for i in output:
        if i.shape[1] == 3:

            out = (np.transpose(i, (0, 2, 3, 1)) / 2 + 0.5).astype(np.float32)[:, :, :, [2, 1, 0]]
            for j in range(BATCH_SIZE):
                cv2.imwrite('{}.jpg'.format(j), (out[j] * 255).astype(np.uint8))
        else:
            mask = i[:, 0]
    print(out.shape)
    fake = (np.transpose(fake, (0, 2, 3, 1)) / 2 + 0.5).astype(np.float32)[:, :, :, [2, 1, 0]]
    fake_mask = fake_mask[:, 0]
    print("Predictor warming up done!")
