import os
from pathlib import Path
from subprocess import Popen
import sys
import time
import os

'''
env config
'''

trt_exec_dict = {
    8003: './TensorRT-8.0.0.3/targets/x86_64-linux-gnu/bin/trtexec'
}
LD_LIBRARY_PATH = {
    8003: './TensorRT-8.0.0.3/targets/x86_64-linux-gnu/lib'
}

'''env config'''
trt_exec_version = 8003
os.environ['LD_LIBRARY_PATH'] = LD_LIBRARY_PATH[trt_exec_version]

if __name__ == '__main__':

    '''config'''
    procesing_onnx_name = 'test.onnx'
    batch_size = 1
    latent_size = 512
    input_size = 256
    '''simple onnx'''
    input_shape = "target:%d,3,%d,%d vsid:%d,%d" % (
    abs(batch_size), input_size, input_size, abs(batch_size), latent_size)
    print('start simplifier onnx')
    processing_simple_onnx_name = procesing_onnx_name.split('.')[
                                      0] + '_sim' + '_bs%d' % batch_size + '_%d.onnx' % trt_exec_version
    if batch_size == -1:
        pb_sim_clt = 'python -m onnxsim  %s  %s  --dynamic-input-shape --input-shape %s ' \
                     % (procesing_onnx_name, processing_simple_onnx_name, input_shape)
    else:
        pb_sim_clt = 'python -m onnxsim  %s  %s --input-shape %s ' \
                     % (procesing_onnx_name, processing_simple_onnx_name, input_shape)

    retcode = Popen(pb_sim_clt, shell=True).wait()

    '''
    sim onnx 2 trt
    '''
    fmt = 16
    workspace = 512 * 20
    iofmt = 16
    print('start convert trt')
    processing_trt_name = processing_simple_onnx_name.split('.')[0] + '_fp%d_ws%d_io%d' % (
    fmt, workspace, iofmt) + '.trt'
    if batch_size == -1:
        if fmt == 16:
            pb_trt_clt = '%s --onnx=%s --saveEngine=%s  --fp%d --workspace=%d --minShapes=target:1x3x256x256,vsid:1x512 --optShapes=target:1x3x256x256,vsid:1x512 --maxShapes=target:1x3x256x256,vsid:1x512' \
                         % (
                         trt_exec_dict[trt_exec_version], processing_simple_onnx_name, processing_trt_name, fmt, 8096)
        else:
            pb_trt_clt = '%s --onnx=%s --saveEngine=%s   --workspace=%d --minShapes=target:1x3x256x256,vsid:1x512 --optShapes=target:8x3x256x256,vsid:8x512 --maxShapes=target:32x3x256x256,vsid:32x512' \
                         % (trt_exec_dict[trt_exec_version], processing_simple_onnx_name, processing_trt_name, 8096)
    else:
        if fmt == 16:
            pb_trt_clt = '%s --onnx=%s --saveEngine=%s --explicitBatch --fp%d --workspace=%d' \
                         % (trt_exec_dict[trt_exec_version], processing_simple_onnx_name, processing_trt_name, fmt,
                            workspace * batch_size)
        else:
            pb_trt_clt = '%s --onnx=%s --saveEngine=%s --explicitBatch  --workspace=%d' \
                         % (trt_exec_dict[trt_exec_version], processing_simple_onnx_name, processing_trt_name,
                            workspace * batch_size)
    if iofmt != 32:
        pb_trt_clt += ' --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw'
    retcode = Popen(pb_trt_clt, shell=True).wait()
